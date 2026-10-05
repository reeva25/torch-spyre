# Cost model 2.0: execution granularity for Phase 1 symbolic shapes

In the symbolic-shapes design the tile size equals the granularity. So each
experiment gives the outermost dimension a concrete size **CG**. The baseline
is `for_each_tile(tile_size=CG)`, which runs a single trip. The model picks an
**EG** that divides CG, which gives `trips = CG / EG`. To study how the
model's choice behaves, vary CG with `--cg`.

## Cost budget

The model gets **one capture** of the baseline compile, taken just before
`_maybe_scratchpad_planning`. There is no compile per candidate and no solver
run; every EG is scored by arithmetic. Recompiling at the picked EG happens
only in the test step, which follows the same pattern as `costmodel/demo_*.py`.

## Files

| File | Role |
|---|---|
| `model.py` | the model. Pure Python. `python3 model.py` runs a toy demo |
| `capture.py` | the pre-solver capture hook, plus test plumbing (recompile, timing) |
| `workloads.py` | `add`, `layernorm` (controls), `softmax`, `mlp` (these overflow LX at CG) |
| `demo_phase1.py` | baseline, then model pick, then one recompile, then compare |
| `calibrate_for_each_tile.py` | one-off EG sweeps that measure the two constants that matter most |

## The model

```
trips      = CG / EG
peak(EG)   = peak_live_bytes(captured buffers, where only `scales` buffers are resized by EG/CG)
loop_us    = trips * PER_ITER_OVERHEAD_US
weights_us = trips * invariant_bytes / WEIGHT_READ_GBS
spill_us   = trips * max(0, peak(EG) - LX) * cores * SPILL_PASSES / SPILL_GBS
pick       = argmin(loop + weights + spill)
```

What changed from v3:

- **Only buffers holding rows of the tiled axis shrink.** The `scales` flag
  comes from the graph's dataflow: a sliced input scales, and so does anything
  computed from it. Weights, and anything computed only from weights, keep
  their size. This needs no loop metadata, so it works on the one-trip
  baseline.
- **New weights term.** `for_each_tile` hands invariants to every iteration
  and nothing keeps them resident, so they are re-read from HBM on every trip.
  v3 had no such term, and the production `_loop_reread_bytes` returns 0.
- **Spill is charged twice (written out, then read back),** at the mixed
  read/write bandwidth.
- Streaming the rows in and out costs the same for every EG, so it is left
  out. It cannot change the pick. In the demo summary, compare *differences*
  between rows rather than absolute times.

## Constants: how much to trust each one

| Constant | Value | Status | Source / problem |
|---|---|---|---|
| `LX_CAPACITY_BYTES` | live | reliable | `_lx_planning_size()` at run time |
| `PER_ITER_OVERHEAD_US` | 0.688 | **placeholder** | v3's slope. Measured on hint-driven coarse tiling with `abs`, not on a `for_each_tile` loop. Sweep A replaces it |
| `WEIGHT_READ_GBS` | 150 | **uncertain, and decides picks** | the production read-only peak. But `cost_model.py` documents that M-split matmul operands are loaded once per core at about 2.3 GB/s per core (about 74 aggregate). In the toy MLP this constant alone flips the pick between 256 and 512. Sweep B measures it |
| `SPILL_GBS`, `SPILL_PASSES` | 105, 2 | reasoned, not calibrated | the production model's measured mixed read/write rate. No sweep yet |
| `NUM_CORES` | 32 | fine | SENCORES default |

Missing penalties:
- core underfill at small EG;
- restickify inside the loop;
- core division changing with EG (the #4233 gap).

The demo's step 3 prints predicted versus actual peak bytes, which shows how
wrong the last of these is.

## Running (Spyre VM)

```bash
python3 costmodel2.0/calibrate_for_each_tile.py        # once; paste results into model.py
python3 costmodel2.0/demo_phase1.py --workload add     # control, expect EG = CG
python3 costmodel2.0/demo_phase1.py --workload mlp
python3 costmodel2.0/demo_phase1.py --workload mlp --cg 1024
python3 costmodel2.0/demo_phase1.py --workload mlp --also-eg 128,256   # grade extra EGs
python3 costmodel2.0/demo_phase1.py --workload mlp --force-eg 256
```

What to check in the output:

- the `scales` column in step 1 (activations `True`, weights `False`);
- predicted versus actual peak in step 3;
- measured time versus the baseline in the summary.

Not run on hardware yet. A failed recompile is reported and the summary still
prints.

## Link to #4233

- The remaining big assumption, that the core split is fixed as EG changes,
  is exactly what a joint decision on tile, cores and LX removes.
- This model is cheap enough (one capture, arithmetic per candidate) to be the
  tile-size term inside that joint search. That matters because compile cost
  is #4233's stated drawback.

## Ideation scope (for later)

1. **Core split as a function of EG.** Query work division for the smaller
   tile without a compile, or take the split from #4233.
2. **A core underfill term.** For example `cost_model.py`'s
   `coarse_underfill_eff`, using the 8-row pass.
3. **Restickify of a transposed weight inside the loop.** Charge it per trip,
   or hoist it out of the loop. This was the dominant cost in your earlier MLP
   run.
4. **A spill calibration case.**
5. **The restickify gap in the buffer list**
   (`costmodel/residency-filter-restickify-gap.md`).
6. **Weight residency (HLD sec. 13).** It would zero the weights term and push
   picks towards smaller EG. The model can price what it's worth.
7. **Where the bridge cuts the loop.** One loop per op versus one around the
   whole chain, and a different EG per loop.
8. **Recommend a CG to the user** when EG = CG keeps winning.
9. **Bands and Phase 2.** Reduction mode adds a per-trip carry-copy term.
