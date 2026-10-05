# Cost model 2.0 -- the scoring. Pure Python, no torch import:  python3 model.py
#
# Setup (matches the symbolic-shapes design, where tile size == granularity):
# the outermost dimension has a CONCRETE size CG, the baseline is
# for_each_tile(tile_size=CG) (one trip), and the model picks an EG dividing
# CG, giving trips = CG / EG.
#
# Budget: ONE capture of the baseline compile, taken just BEFORE LX planning.
# No compile per candidate, no solver run. Every EG is scored by arithmetic.
#
# Changes vs v3:
#   1. Only buffers that hold rows of the tiled axis are scaled by EG/CG.
#      Weights (dims=None) and anything computed only from weights keep their
#      size (the `scales` flag, set by capture.py from the graph's dataflow).
#   2. Weights are re-read from HBM on every iteration -> new term.
#   3. Spill traffic is written AND read back, at the mixed read/write
#      bandwidth, instead of counted once at datasheet peak.
#
# Still assumed: core division at EG is the same as at CG (see README).

from dataclasses import dataclass

# ── constants ──────────────────────────────────────────────────────────────
# Status of each: MEASURED (on device, source cited) / PLACEHOLDER (needs
# calibrate_for_each_tile.py) / LIVE (read from the compiler at run time).

# LIVE: capture.py replaces this with scratchpad.allocator._lx_planning_size().
LX_CAPACITY_BYTES = 1_589_376

NUM_CORES = 32  # SENCORES default

# PLACEHOLDER. Fixed cost of one iteration. 0.688 is v3's slope, measured on
# the hint-driven coarse-tiling loop with `abs` -- not on a for_each_tile
# loop. calibrate_for_each_tile.py sweep A measures the real one.
PER_ITER_OVERHEAD_US = 0.688

# MEASURED, but maybe the wrong one for matmul weights. 150 GB/s is the
# production model's read-only peak (cost_model.py CostParams.bw_peak_gbps).
# cost_model.py also documents that a matmul operand whose core split is on a
# dim it does not index (M-split -> the weight) is loaded once PER CORE at
# ~2.3 GB/s per core (mm_replicated_read_gbps_per_core) -- far slower. Which
# one applies depends on work division; calibrate_for_each_tile.py sweep B
# measures the effective number for a linear layer.
WEIGHT_READ_GBS = 150.0

# MEASURED. Spilled bytes are written then read back: mixed traffic, which
# the production model measured at ~105 GB/s effective (balanced 1R+1W,
# cost_model.py CostParams docstring). Not directly calibrated for spill.
SPILL_GBS = 105.0
SPILL_PASSES = 2  # write out + read back


@dataclass
class Params:
    lx_capacity_bytes: int = LX_CAPACITY_BYTES
    num_cores: int = NUM_CORES
    per_iter_overhead_us: float = PER_ITER_OVERHEAD_US
    weight_read_gbs: float = WEIGHT_READ_GBS
    spill_gbs: float = SPILL_GBS
    spill_passes: int = SPILL_PASSES


def _us(nbytes: float, gbs: float) -> float:
    return nbytes / (gbs * 1e3)  # GB/s == bytes/ns -> *1e3 = bytes/us


@dataclass
class Capture:
    """What the baseline compile (tile = CG) looked like before LX planning.

    buffers: v3's dict shape plus `scales`:
        {"name", "full_size_bytes", "start_tick", "end_tick", "scales"}
        full_size_bytes is PER CORE at tile = CG.
    invariant_bytes: total bytes of the dims=None operands (weights).
    """

    cg: int
    buffers: list[dict]
    invariant_bytes: int


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def candidate_tile_sizes(cg: int, min_eg: int = 1) -> list[int]:
    return [d for d in divisors(cg) if d >= min_eg]


def buffers_at(capture: Capture, eg: int) -> list[dict]:
    """The captured list re-sized for tile EG: only `scales` buffers shrink."""
    return [
        {**b, "full_size_bytes": b["full_size_bytes"] * eg // capture.cg}
        if b["scales"] else b
        for b in capture.buffers
    ]


def peak_live_bytes(buffers: list[dict]) -> int:
    """v3's function unchanged: peak bytes alive at one tick, per core."""
    ticks = {b["start_tick"] for b in buffers} | {b["end_tick"] for b in buffers}
    peak = 0
    for t in ticks:
        live = sum(
            b["full_size_bytes"] for b in buffers if b["start_tick"] <= t < b["end_tick"]
        )
        peak = max(peak, live)
    return peak


@dataclass
class Breakdown:
    trips: int
    peak_bytes: int
    loop_us: float
    weights_us: float
    spill_us: float

    @property
    def total_us(self) -> float:
        return self.loop_us + self.weights_us + self.spill_us


def predict(capture: Capture, eg: int, params: Params | None = None) -> Breakdown:
    """EG-dependent cost. (Streaming the sliced rows in and out is the same
    for every EG, so it is left out: it cannot change the ranking.)

        trips      = CG / EG
        loop_us    = trips * PER_ITER_OVERHEAD_US
        weights_us = trips * invariant_bytes / WEIGHT_READ_BW
        spill_us   = trips * overflow_per_core * cores * SPILL_PASSES / SPILL_BW
    """
    p = params or Params()
    assert capture.cg % eg == 0, f"EG={eg} does not divide CG={capture.cg}"
    trips = capture.cg // eg
    peak = peak_live_bytes(buffers_at(capture, eg))
    overflow_total = max(0, peak - p.lx_capacity_bytes) * p.num_cores
    return Breakdown(
        trips=trips,
        peak_bytes=peak,
        loop_us=trips * p.per_iter_overhead_us,
        weights_us=trips * _us(capture.invariant_bytes, p.weight_read_gbs),
        spill_us=trips * _us(overflow_total * p.spill_passes, p.spill_gbs),
    )


def pick(capture: Capture, min_eg: int = 1, params: Params | None = None) -> int:
    egs = candidate_tile_sizes(capture.cg, min_eg)
    return min(egs, key=lambda eg: predict(capture, eg, params).total_us)


def pick_largest_fitting(capture: Capture, min_eg: int = 1, params: Params | None = None) -> int:
    """#4381's rule, no timing model: largest EG whose tile fits LX."""
    p = params or Params()
    egs = candidate_tile_sizes(capture.cg, min_eg)
    fits = [eg for eg in egs
            if peak_live_bytes(buffers_at(capture, eg)) <= p.lx_capacity_bytes]
    return max(fits) if fits else min(egs)


def print_table(capture: Capture, min_eg: int = 1, params: Params | None = None):
    p = params or Params()
    print(f"  {'EG':>5} {'trips':>6} {'peak/core':>11} {'fits':>5} "
          f"{'loop':>8} {'weights':>9} {'spill':>9} {'total us':>10}")
    for eg in candidate_tile_sizes(capture.cg, min_eg):
        b = predict(capture, eg, p)
        print(f"  {eg:>5} {b.trips:>6} {b.peak_bytes:>11,} "
              f"{str(b.peak_bytes <= p.lx_capacity_bytes):>5} {b.loop_us:>8.1f} "
              f"{b.weights_us:>9.1f} {b.spill_us:>9.1f} {b.total_us:>10.1f}")


# ── toy demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # MLP-like capture at CG = 512: x (512, 256), Dh = 32768, 32 MB of weights.
    # Illustrative numbers, not captured.
    CG = 512
    act = CG * 32768 * 2 // NUM_CORES
    io = CG * 256 * 2 // NUM_CORES
    w = 32768 * 256 * 2 // NUM_CORES
    cap = Capture(
        cg=CG,
        buffers=[
            {"name": "x", "full_size_bytes": io, "start_tick": 0, "end_tick": 1, "scales": True},
            {"name": "w1", "full_size_bytes": w, "start_tick": 0, "end_tick": 1, "scales": False},
            {"name": "h", "full_size_bytes": act, "start_tick": 0, "end_tick": 2, "scales": True},
            {"name": "silu_h", "full_size_bytes": act, "start_tick": 1, "end_tick": 3, "scales": True},
            {"name": "w2", "full_size_bytes": w, "start_tick": 2, "end_tick": 3, "scales": False},
            {"name": "out", "full_size_bytes": io, "start_tick": 2, "end_tick": 3, "scales": True},
        ],
        invariant_bytes=2 * 32768 * 256 * 2,
    )
    print(f"toy MLP capture, CG={CG}, LX={LX_CAPACITY_BYTES:,} B/core\n")
    print_table(cap, min_eg=8)
    print(f"\nformula picks EG={pick(cap, 8)};  largest-fitting picks EG={pick_largest_fitting(cap, 8)}")
    for gbs in (150.0, 2.3 * NUM_CORES):
        p = Params(weight_read_gbs=gbs)
        print(f"  with weight read BW {gbs:.0f} GB/s: formula picks EG={pick(cap, 8, p)}")
