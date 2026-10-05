# LX Tile Cost Model — Explained


## 1. Why do we need this?

### The problem

Spyre has a small, fast, per-core memory called **LX** (about 1.59MB per core, out
of 32 cores). If the data a computation needs at one moment is small enough, it can
live entirely in LX and run fast. If it's too big, it has to live in the much
bigger, much slower main memory (HBM), and the compute has to keep reaching out to
HBM to fetch it , which is slow. 


Torch-spyre compiles some dimensions as **symbolic** , a size that isn't fixed at
compile time, but has to stay valid across a whole range of concrete values at
runtime . The user  declares that
range along with a **granularity**: the step between legal sizes. That granularity
is also, today, literally the size of the one static tile the compiled kernel loops
over.

The granularity the user declares is chosen for external reasons . It has nothing to do with whether a tile of that size
actually fits in LX.

### A concrete example

Say a user declares a granularity of **CG = 100**. At that granularity, one tile of
the computation has shape **`(100,300000 )`**. In fp16, spread across 32 cores, that
tile is too big for LX , it doesn't fit.

Without any intervention, that's just a fact the system lives with: this tile
permanently spills to HBM, every single time, because the granularity the user
picked *is* the tile size, with nothing standing between the two.

**What this cost model does**: it looks at the user's CG (100) and asks : is there
a smaller divisor of 100 that *would* fit? Say **EG = 50** does. The compiler then
internally runs the computation in tiles of shape **`(50, 300000)`** instead, and that shape fits comfortably in LX. The user's declared granularity (100) doesn't have to
change for this to work; the cost model just picks a smaller number, **EG**, that
divides evenly into it, and uses that internally.

In short:

- **CG** ("chosen granularity") = the granularity value the user picked.
- **EG** = a smaller tile size the cost model searches for, always a divisor of CG
  (so the loop count `CG / EG` is always a whole number, no leftovers).
- The cost model's job: given whatever CG the user picked, search its divisors and
  pick the EG that optimises LX Scratchpad usage. 


**Without a cost model, a tile at the user's chosen granularity either permanently overflows LX, or someone has to hand-pick a size that only works once. With a cost model, a smaller EG that divides the user's CG is found automatically, so the (100, 300000) tile that overflows becomes (50, 300000), which fits.**

---

## 2. Prior Art

### 2.1 Work Division Cost Model

Spyre has 32 cores. For a `matmul`, the compiler has to decide which core does
which piece of the work : you can split along rows (M), columns (N), the shared
inner dimension (K), or the batch dimension (B). Every possible combination of
splits is a candidate. The existing work-division cost model computes **9 penalty
scores** and tries every *valid* combination of `b, m, k, n`, picking whichever
scores best.

This is a different question from ours : it decides **how work is divided across
cores** for a *fixed* amount of data. Our cost model decides **how much data exists
at once** in the first place (the tile size). They interact, but they're not the
same search.



### 2.2 LX Solvers

Once you know the list of buffers and their lifetimes, something still has to
decide **where** each one physically sits in LX (or whether it fits at all). That's
the solvers' job.

- **They decide placement, not size.** A solver is handed a fixed list of buffers
  with fixed byte sizes and fixed lifetimes. It never asks "should this buffer be
  smaller?" . It only asks "given these sizes, where (if anywhere) does
  each one go?"

The solver variants:

| Solver | Strategy |
|---|---|
| **Greedy** | Goes in operation order, places each buffer in the first empty space it finds. |
| **FirstFit** | Sorts buffers by ascending lifetime first, then places each in the first gap it finds. |
| **BestFit** | Same sort as FirstFit, but picks the gap that leaves the *least* leftover space. |
| **CP-SAT** | Models all buffers as rectangles (address × capacity) and solves for the best placement of *all of them simultaneously*. |
| **Simulated Annealing** | Currently being worked on. |

**The important caveat, worth repeating on its own line:** a buffer can pass every
structural eligibility check and *still* get rejected by the solver, simply because
there isn't room left for it once other buffers have claimed their space. Passing
the eligibility check is necessary, not sufficient. (Section 4 has a concrete,
measured example of exactly this.)

### 2.3 SDPA / SWA cost model

There is already a hand-built cost model in `decompositions.py` : but it's
**formula-based and narrow**: it calculates live-buffer bytes using a fixed formula
written specifically for two ops, SDPA (scaled dot product attention) and SWA
(sliding-window attention).

The natural next idea is to do the same thing more broadly : hand-derive a buffer
formula per op category, the way SDPA/SWA already has one. **Why that doesn't
generalize**: most use cases aren't single isolated ops. They're chains of 
ops, and the number of ways ops can chain together is effectively unbounded. A
formula per op category doesn't compose automatically into a formula for an
arbitrary fused chain . We cannot hand-derive a formula for every possible op combination.

---

## 3. The cost model algorithm

### 3.1 It considers the entire graph

It would be simpler to ask "is this one buffer too big?" for each buffer in
isolation. That doesn't work because **one op you write is not one buffer**.
Graph lowering decomposes it into several intermediate buffers, each with its own
size and core division, all of which have to be alive on LX together within one
tile:

![One logical op — softmax(x) — decomposes during graph lowering into 4 separate buffers of different sizes (max, exp, sum, divide), all alive together within one tile iteration.](diagrams/graph_decomposition.svg)

So the only way to know whether a candidate `EG` actually works is to look at
every buffer in the graph together at the moment they're most crowded , not to
check buffers one at a time.

### 3.2 Data extraction, and what `peak_live_bytes` records from it

Two real torch-spyre functions do the extraction, called on the compiled graph:

| Function | Returns |
|---|---|
| `calculate_liveness(graph)` | for every buffer, the op indices ("ticks") that touch it — `start_tick` (first use) to `end_tick` (one past the last use). It's "alive" on LX for that whole window. |
| `mem_usage_by_buf(graph)` | for every op-produced buffer, its per-core byte size. |

*Example*: ops `0, 1, 2, 3` in a graph, a buffer with `start_tick=1, end_tick=3` →
ops `1` and `2` use it.

Buffers that would never be allowed onto LX at all ( because they fail the real
allocator's structural `residency_reasons` check, for reasons that have nothing to
do with size ) are filtered out .

With ticks and sizes in hand, `peak_live_bytes` asks: **at the single busiest
tick, how many bytes are alive at once?** Not the *total* of every buffer that
ever existed — the *tallest single moment* of overlap.

![Three buffers with overlapping lifetimes across ticks 0-4. buf_A is alive ticks 0-2, buf_B ticks 1-3, buf_C ticks 2-4. At tick 2 all three overlap, giving peak_live_bytes for the graph.](diagrams/peak_live_bytes.svg)

This is a **per-core** number, and it has to fit under `LX_CAPACITY` (1,589,376
bytes/core) for the working set to be LX-resident.

Today this size is calculated once, at the full untiled size, then scaled down by
`EG/CG`, which assumes core division and buffer size both scale uniformly with
tile size (innacurate assumption , will be adjusted in future steps ).



### 3.3 What the buffer list looks like

Putting the two together (minus anything residency-filtered) gives a plain list
of dicts — the actual shape of the data everything below works from:

```python
{'name': 'buf0', 'full_size_bytes': 32768, 'start_tick': 0, 'end_tick': 1}
```

Name, full per-core byte size, and the tick window it's alive for. Nothing else
about the graph matters to the cost model past this point.

### 3.4 The full pipeline

![Pipeline: the real compiled graph feeds liveness and memory-usage extraction, then residency filtering, into a buffer list. A search loop scores each candidate EG by loop overhead and HBM spill cost, and the lowest-scoring EG is picked.](diagrams/pipeline_flowchart.svg)

### 3.5 What the search is doing

For a chosen `CG`, the search tries the divisors `EG` of `CG` , scales `peak_live_bytes` by `EG/CG` 
, score the two penalties below and adds them, and keeps the `EG`
with the lowest total. 

![Total cost is U-shaped across candidate EG values: loop overhead falls as EG grows, HBM spill cost is zero until the tile stops fitting then rises steeply, and the sum has a minimum — the sweet spot the search picks.](diagrams/tradeoff_curve.svg)

Since the curve almost goes in a U we can go for a greedy search where we stop the search once penalty starts to increase again 

(note that this is just a theoretical curve , testing that there are other factors such as core division that may govern the change in time with a decreased tile size , not just loop count.)



### 3.6 The two penalties

**Penalty 1 : loop overhead.** Every extra tile iteration beyond the one baseline
pass costs a small, fixed amount:

```python
extra_iterations = CG // EG - 1
loop_overhead_us  = extra_iterations * _PER_LOOP_COST_US
```

`_PER_LOOP_COST_US = 0.688` was measured, not guessed: a fixed `3840 × 256` tensor
(chosen so every K in the sweep divides evenly), K swept over `{2, 3, 4, 6, 8, 12,
16, 24, 32}`, real device kernel time measured on hardware, each K repeated, and a
line `time(K) = a·K + b` fit across the results .`a = 0.688` is the slope, taken
as the per-loop cost.

**Penalty 2 : cost of spilling to HBM.** If the (scaled) working set still doesn't
fit in LX, the overflow has to be re-fetched from HBM on every iteration:

```python
overflow    = max(0, peak_live_bytes(EG) - lx_capacity)
iterations  = CG // EG
hbm_spill_us = overflow * iterations / _HBM_BW_GBS
```

Zero whenever the tiled working set fits; otherwise it grows with both how much it
overflows by *and* how many iterations that overflow is paid on. One refinement
raised but not yet built in: **HBM bandwidth derating** : real transfers rarely hit
100% of peak bandwidth, so multiplying by some fraction of peak (rather than
assuming the full rated bandwidth) would make this penalty more realistic.

### 3.7 Future Scope / Initial Assumptions that will gradually be rectified


1. **Assumes every buffer's size scales linearly with `EG/CG`.** There are two
   reasons why this is not necessarily the case. First is that a buffer might not
   map to the entire tile, so a reduction in the tile size might not lead to a
   linear reduction in the buffer. Second is that a reduction in tile size might
   lead to a reduction in the number of cores that the buffer is split across
   instead of a reduction in the per core buffer size.


2. **Treats "eligible" as the same as "will be placed."** This ignores the fact
   that the solver may still not place all eligible buffers on the scratchpad 

3. **No penalty for loop-invariant operands.** Weights (or any input that's the
   *same* on every tile iteration) still have to be reshaped/restickified once per
   iteration if that reshaping sits inside the loop body — the model has zero
   awareness of this cost. This turned out to be the **dominant** cost in one of
   the test cases (§4): tiling correctly shrank the target buffer and got it
   placed in LX, and the run was *still* slower than not tiling at all, because
   redundant per-iteration weight-reshaping outweighed the LX win entirely.
   Notably, the **production** cost model (`cost_model.py`'s `_loop_reread_bytes`)
   already has a term reserved for exactly this — and its own docstring says it is
   currently **inert** (returns 0.0) until an upstream extractor sets a value it
   depends on. This isn't a gap unique to the toy model in this folder; it's a
   known, currently-dormant gap in the real system too.

<!-- 5. **`_PER_LOOP_COST_US` is a single flat constant**, calibrated on one simple,
   weight-free op (plain elementwise `abs`), only validated over `K = 2..32`. It
   cannot capture superlinear cost growth at deeper tile counts (per-tile program
   bytes and backend compile time both grow faster than linearly past a certain
   point) — which is why this model enforces a `MAX_TILES` cap rather than
   trusting the flat constant into unvalidated depth. -->

4. **No HBM-bandwidth derating yet** — `hbm_spill_us` assumes the full rated
   `_HBM_BW_GBS`, which real transfers essentially never sustain at 100%.

---

## 4. Test analysis & findings

### Setup

Both cases below follow the same procedure: compile the function **untiled**
and capture the real, residency-filtered buffer list from the compiler; feed
that buffer list to the cost model to pick the best `K`; recompile the
function **with that `K` applied** (via `spyre_hint(num_tiles_per_dim=...)`);
measure real device time for both compiles and compare.

### Placement of the cost model in this setup

The cost model's only job here is the middle step : pick `K` from the
untiled buffer list. It has zero visibility into what the recompile actually
produces. Its prediction is graded after the fact, by comparing the `K` it
picked and the savings it predicted against what the hardware actually
measured.

### 4.1 Softmax

**What it runs**: `torch.softmax(x, dim=1)` on a `(R=1024, C=32768)` matrix.
Softmax normalizes across each row (the `C` axis), so `C` can't be tiled : the
tiling instead splits `R` (the row axis) into `K` row-groups, so each loop
iteration processes a smaller batch of complete rows.

**What the model picked**: `EG=256` → `K=4`, predicting the untiled
~4.2MB/core peak would shrink enough to fit comfortably in LX.

**What actually happened**: device time went from **4,639µs (untiled) to
10,804µs (K=4)** : about **2.3× slower**, not faster.

**Possible Reasons for Unfavourable Result:**:

- **Tiling evicted buffers that were already working.** Untiled, the two
  small reduction buffers (row max, row sum — 4KB/core each) were the *only*
  things placed in LX. After tiling, both get rejected: the reduction now
  produces on 1 core but the tiled op runs on 32, and that broadcast isn't
  allowed in LX. 
- **Tiling created buffers the model never saw.** New staging buffers
  (`coarse_tile_read_copy_0_arg0_1_0`, `coarse_tile_copy_buf4`) appear only in
  the tiled compile : real extra HBM traffic the model had no way to account
  for, since it only ever looked at the untiled graph. (This issue **may** not exist given the new for each tile loop design )

### 4.2 MLP Layer

**What it runs**: a two-layer MLP, `linear → silu → linear`, shape
`S=32768, Din=256, Dh=1024, Dout=256`. The two `S×Dh` activations are
~2MB/core each and co-live at the `silu` step : about 2.5× LX capacity.
Tiling splits `S` (the free/outer axis for both GEMMs).

**What the model picked**: `EG = S/4` → `K=4`, predicting the ~4MB/core peak
would drop to ~1MB/core and fit.

**What actually happened : it depends on a hidden variable the model doesn't
know about**:

- **With core division pinned** (`work_div={"Dh": 16}`, forcing cores to stay
  on `Dh` regardless of tiling): the prediction holds. Buffers land in LX as
  predicted, and a real speedup is measured.
- **Without the pin** (production has no such pin): the matmul work-division
  logic reassigns cores from `S` onto `Dh` as `S`'s tile shrinks. Result: the
  "tiled" activation buffer reports the **exact same bytes/core** as the
  untiled one : tiling changed nothing, silently.

**Why it's not working**: the cost model assumes core division is fixed while
it scales buffer sizes by `EG/CG`. But the real work-division cost model
re-decides core assignment independently, *after* seeing the smaller tile —
and it is never told that a tiling decision was even made. Two optimizers
making sequential, uncoordinated decisions about the same resource can cancel
each other out with zero warning.

---

## 5. Issue [#4233](https://github.com/torch-spyre/torch-spyre/issues/4233)

This existing issue describes a **joint decision** for choosing the most optimum
tile size, core division, and LX buffer placement together. These are three
decisions that affect each other, but today they're taken at separate phases in
the pipeline : this issue's proposal is to make that decision jointly instead.

The points relevant to this project, straight from that issue:

- **No penalty yet for running too many loops.** 
- **Where this fits into the new symbolic shapes flow.** Depending on where this
  joint decision lands relative to the symbolic-shapes work, this project's
  solution can be adapted to ensure the chosen tile size is always a divisor of
  the granularity , and we can use this solution as an alternative to the cost model 
- **Drawback: cost.** Co-optimization with just core division already eats into a
  huge chunk of compile time. Folding tile-size selection into that same joint
  decision too may make it even slower still.
