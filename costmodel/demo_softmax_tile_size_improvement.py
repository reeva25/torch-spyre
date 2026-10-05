# Same demonstration as demo_tile_size_improvement.py (MLP), but for softmax:
# does the v3 cost model recognize an oversized softmax buffer, pick a tile
# count that makes it LX-resident, and does that show up as a REAL measured
# speedup on device?
#
# Why the earlier softmax attempt failed and this one doesn't: that attempt
# used shape (DIM_A=128, DIM_B=153600) -- a small row count and a huge
# softmax-reduction axis. The reduction's own OUTPUT (one value per row) was
# then only 128 elements / 32 cores = 4 elements/core, under the 64-element
# (128-byte) stick minimum, so no valid per-core split existed at ANY tile
# size ("ownership ... cannot be represented by the physical buffer
# layout"). That's a shape problem, not a size problem -- tiling can't fix
# it, same as tiling can't fix a GEMM whose reduction axis is too thin.
#
# The team's own softmax fixture (_softmax_case in
# tests/inductor/test_solver_auto_coarse_tiling.py) avoids exactly this:
# torch.softmax(x, dim=0) over shape (R, C) -- R is the reduced axis, C is
# the free axis, and C (not R) is what gets tiled. This script uses that
# same shape convention, scaled up until the full (R, C) buffer overflows
# LX. Unlike the MLP demo, there is no work_div core-division pin here -- R
# can't be split at all for the reduction ops, so there's nothing to pin;
# see the note above _make_tiled_softmax for details.
#
# Run from anywhere inside the repo checkout: python3 demo_softmax_tile_size_improvement.py

import os
import sys
import time
from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch.profiler import ProfilerActivity, profile

import torch_spyre  # noqa: F401 -- registers the "spyre" device + Inductor backend
from torch._inductor.graph import GraphLowering
from torch_spyre.streams import synchronize


def _find_tests_inductor_dir():
    directory = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        candidate = os.path.join(directory, "tests", "inductor")
        if os.path.isdir(candidate):
            return candidate
        directory = os.path.dirname(directory)
    raise RuntimeError(
        "Could not find tests/inductor by walking up from "
        f"{os.path.dirname(os.path.abspath(__file__))} -- run this script "
        "from somewhere inside the torch-spyre repo checkout."
    )


sys.path.insert(0, _find_tests_inductor_dir())

from test_coarse_tile_e2e import LoopSpecCheck, run_coarse_tile_test, tensor  # noqa: E402
from torch_spyre._inductor import config as ts_inductor_config  # noqa: E402
from torch_spyre._inductor import passes  # noqa: E402
from torch_spyre._inductor import spyre_hint  # noqa: E402
from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator  # noqa: E402
from torch_spyre._inductor.scratchpad.lx_relayout import collect_lx_relayout_plans  # noqa: E402
from torch_spyre._inductor.scratchpad.utils import (  # noqa: E402
    calculate_liveness,
    get_ncores_for_buffers,
    mem_usage_by_buf,
)
from torch_spyre._inductor.wsr import propagate_named_dims as pnd  # noqa: E402

from sympy import divisors  # noqa: E402

# ── cost model (inlined from costmodel/lx_tile_cost_model_v3.py) ─────────────

LX_CAPACITY = 1_589_376  # bytes, PER CORE -- true allocator capacity, from _lx_planning_size()
NUM_CORES = 32
_HBM_BW_GBS = 204.8
_PER_LOOP_COST_US = 0.688  # real measured value (device time, r^2=0.932); see MLP demo



def peak_live_bytes(buffers, EG, CG):
    scaled = [{**b, "size": b["full_size_bytes"] * EG // CG} for b in buffers]
    all_ticks = {b["start_tick"] for b in scaled} | {b["end_tick"] for b in scaled}
    peak = 0
    for t in all_ticks:
        live = sum(b["size"] for b in scaled if b["start_tick"] <= t < b["end_tick"])
        peak = max(peak, live)
    return peak


def loop_overhead_us(EG, CG):
    iterations = CG / EG
    return _PER_LOOP_COST_US * iterations


def hbm_spill_us(buffers, EG, CG, num_cores=NUM_CORES, lx_capacity=LX_CAPACITY):
    overflow_per_core = max(0, peak_live_bytes(buffers, EG, CG) - lx_capacity)
    overflow_total = overflow_per_core * num_cores
    iterations = CG // EG
    return overflow_total * iterations / (_HBM_BW_GBS * 1e3)  # convert GB/s to bytes/us


def find_best_tile_size(buffers, CG, num_cores=NUM_CORES, lx_capacity=LX_CAPACITY):
    best_EG, best_score = None, float("inf")
    candidates = [EG for EG in sorted(divisors(CG))]
    #print(f"  (candidates limited to K <= {MAX_TILES}: EG >= {min(candidates)})")
    print(f"  {'EG':>6}  {'K':>5}  {'loop_us':>10}  {'spill_us':>10}  {'total_us':>10}")
    for EG in candidates:
        loop = loop_overhead_us(EG, CG)
        spill = hbm_spill_us(buffers, EG, CG, num_cores, lx_capacity)
        total = loop + spill
        print(f"  {EG:>6}  {CG // EG:>5}  {loop:>10.3f}  {spill:>10.3f}  {total:>10.3f}")
        if total < best_score:
            best_score = total
            best_EG = EG
    return best_EG


# ── timing (inlined from costmodel/calibrate_loop_overhead_v3.py) ────────────

WARMUP_ITERS = 5
TIMED_ITERS = 30
REPEAT_TRIALS = 5
NOISE_CV_WARN_THRESHOLD = 0.10


def mean_stdev(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def measure_wall_us(compiled_fn, args):
    for _ in range(WARMUP_ITERS):
        compiled_fn(*args)
    synchronize()

    samples = []
    for _ in range(TIMED_ITERS):
        start = time.perf_counter()
        compiled_fn(*args)
        synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def measure_device_us(compiled_fn, args):
    for _ in range(WARMUP_ITERS):
        compiled_fn(*args)
    synchronize()

    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "PrivateUse1"):
        activities.append(ProfilerActivity.PrivateUse1)

    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(TIMED_ITERS):
            compiled_fn(*args)
        synchronize()

    total_device_us = 0.0
    matched_any = False
    for evt in prof.events():
        if "spyre_kernel" in evt.name or "sdsc" in evt.name:
            device_us = getattr(evt, "self_device_time_total", None)
            if device_us:
                matched_any = True
                total_device_us += device_us

    if not matched_any:
        return None
    return total_device_us / TIMED_ITERS


def measure_device_us_with_noise_check(compiled_fn, args, label):
    trial_values = []
    for _ in range(REPEAT_TRIALS):
        device_us = measure_device_us(compiled_fn, args)
        if device_us is None:
            print(f"  {label}  [no device-kernel events matched -- can't measure device time]")
            return None
        trial_values.append(device_us)
    mean, stdev = mean_stdev(trial_values)
    cv = stdev / mean if mean else float("inf")
    flag = " <- NOISE WARNING: increase R/C or REPEAT_TRIALS" if cv > NOISE_CV_WARN_THRESHOLD else ""
    print(f"  {label}  device_mean={mean:.3f}us  stdev={stdev:.3f}us  cv={cv:.1%}{flag}")
    return mean


def measure_wall_us_with_noise_check(compiled_fn, args, label):
    trial_medians = [measure_wall_us(compiled_fn, args) for _ in range(REPEAT_TRIALS)]
    mean, stdev = mean_stdev(trial_medians)
    cv = stdev / mean if mean else float("inf")
    flag = " <- NOISE WARNING: increase R/C or REPEAT_TRIALS" if cv > NOISE_CV_WARN_THRESHOLD else ""
    print(f"  {label}  mean={mean:.3f}us  stdev={stdev:.3f}us  cv={cv:.1%}{flag}")
    return mean


# ── real buffer list (inlined from costmodel/buffer_list_with_residency_filter.py)


def build_buffers_with_residency_filter(graph, division_is_fixed: bool = True) -> list[dict]:
    """Buffer list (name/size/start_tick/end_tick) for op-produced buffers
    AND graph inputs, dropping anything the real allocator's residency check
    would never let into LX."""
    allocator = ScratchpadAllocator(layout_planning=None, size=0)  # solver never runs below

    lifetimes = calculate_liveness(graph)
    mem_usage = mem_usage_by_buf(graph)
    ncores, ncores_reasons, _lx_views = get_ncores_for_buffers(graph)

    plans = collect_lx_relayout_plans(graph)
    planned_lx_buffers = ScratchpadAllocator._planned_lx_buffer_names(plans)

    op_reasons = allocator._residency_reasons(
        graph,
        list(mem_usage.keys()),
        division_is_fixed=division_is_fixed,
        lifetimes=lifetimes,
        ncores=ncores,
        ncores_reasons=ncores_reasons,
        planned_lx_buffers=planned_lx_buffers,
        lx_relayout_plans=plans,
    )

    buffers = []
    for buf_name, uses in lifetimes.items():
        if not uses:  # unused graph input -- skip
            continue
        if buf_name in mem_usage:  # op-produced buffer
            if op_reasons.get(buf_name) is not None:
                continue
            size = mem_usage[buf_name].get("size_per_core", -1)
            if size < 0:
                continue
        elif buf_name in graph.graph_input_names:  # graph input
            reason = allocator._input_residency_reason(
                graph,
                buf_name,
                uses,
                ncores=ncores,
                ncores_reasons=ncores_reasons,
                division_is_fixed=division_is_fixed,
            )
            if reason is not None:
                continue
            size = ScratchpadAllocator._input_footprint(graph, buf_name, ncores)
            if size <= 0:
                continue
        else:
            continue  # no sizeable layout, no fallback -- skip

        buffers.append(
            {
                "name": buf_name,
                "full_size_bytes": size,
                "start_tick": uses[0],
                "end_tick": uses[-1] + 1,
            }
        )
    return buffers


# ── the model ────────────────────────────────────────────────────────────────
# torch.softmax(x, dim=0) over shape (R, C) -- same convention as _softmax_case
# in test_solver_auto_coarse_tiling.py. R is the reduced axis; C is the free
# axis and the ONLY one tiled (the fixture's own docstring: tiling R compiles
# but is numerically wrong -- the tiled max/sum drain through
# coarse_tile_combine/reduce_copy and land off by orders of magnitude).
#
# Sized so the full (R, C) buffer overflows LX per core:
# R*C*2 bytes / 32 cores must beat ~1.59MB, i.e. R*C > ~25M elements.
# R=1024 (mirrors the MLP demo's Dh) and C=32768 gives 2MB/core, with two
# buffers co-live (input and softmax output) -- same shape of problem as the
# MLP demo, same expected small K.

R, C = 1024, 32768
CG = C  # the tiled axis


# NOTE: unlike the MLP demo, there is no work_div pin here. The MLP could pin
# Dh because Dh was a free (splittable) axis for one of its two ops even
# though it was the reduction axis for the other. Softmax only has two axes:
# R is a reduction axis for EVERY op that touches it (nothing else to pin),
# and C is the one axis being tiled. Attempting spyre_hint(work_div={"R": N})
# fails outright, even untiled: "work_division_hint: buf0 dim d1 legal splits
# are [1]" -- the max/sum reduction over R can't be split across cores at
# all for that op, so there's no independent axis available to lock down.
# This script instead just tiles C and relies on the "Actual LX placements"
# printout to show directly whether core-division drift is even a problem
# here (it may not be -- softmax's ops are pointwise/reduction, not the
# matmul cost-model tradeoff that caused the MLP's core reassignment).


def fn(x):
    return torch.softmax(x, dim=0)


def _make_tiled_softmax(k_literal: int):
    """softmax closure with num_tiles_per_dim={'C': k_literal} as a real
    source-level literal (see module docstring)."""
    namespace = {"spyre_hint": spyre_hint, "torch": torch}
    src = (
        "def tiled(x):\n"
        f"    with spyre_hint(num_tiles_per_dim={{'C': {k_literal}}}):\n"
        "        return torch.softmax(x, dim=0)\n"
    )
    exec(src, namespace)
    return namespace["tiled"]


_ARG_DIMS = (["R", "C"],)
_ARG_SHAPES = ((R, C),)
_LOOPSPEC_INPUTS = [
    tensor(name, shape=shape, dims=dims)
    for name, shape, dims in zip(("x",), _ARG_SHAPES, _ARG_DIMS)
]


def _print_actual_lx_placements(graph: GraphLowering) -> None:
    """Print which buffers the allocator solver actually placed in LX."""
    lifetimes = calculate_liveness(graph)
    mem_usage = mem_usage_by_buf(graph)

    rows = []
    for buf_name, uses in lifetimes.items():
        if not uses:
            continue
        if buf_name not in mem_usage:
            continue
        size = mem_usage[buf_name].get("size_per_core", -1)
        if size < 0:
            continue
        buf = graph.get_buffer(buf_name)
        alloc = {}
        try:
            alloc = buf.get_layout().allocation
        except Exception:
            pass
        if "lx" in alloc:
            placement = f"LX  (addr={alloc['lx']})"
        else:
            placement = "HBM (solver rejected / ineligible)"
        rows.append((uses[0], buf_name, size, placement))

    rows.sort()
    print(f"  {'name':<28} {'bytes/core':>12}  {'placement'}")
    for _, buf_name, size, placement in rows:
        print(f"  {buf_name:<28} {size:>12}  {placement}")


class _CaptureGraphPass(passes.CustomPreSchedulingPasses):
    """Splices two passes around _maybe_scratchpad_planning:
      - BEFORE: captures the residency-filtered buffer list for the cost model.
      - AFTER:  prints the allocator's actual LX placement decisions.
    """

    captured_buffers: list[dict] | None = None

    def __init__(self):
        super().__init__()
        insert_at = self.passes.index(passes._maybe_scratchpad_planning)
        self.passes.insert(insert_at, self._capture)
        self.passes.insert(insert_at + 2, self._show_placements)

    def _capture(self, graph: GraphLowering) -> None:
        _CaptureGraphPass.captured_buffers = build_buffers_with_residency_filter(
            graph, division_is_fixed=True
        )

    def _show_placements(self, graph: GraphLowering) -> None:
        print("  Actual LX placements (post-solver):")
        _print_actual_lx_placements(graph)


def _prep_inputs():
    pnd.reset()
    for name, size in (("R", R), ("C", C)):
        pnd.declare_tensor_dim(name, size)
    torch.manual_seed(0xC0A75E)
    args = (torch.rand((R, C), dtype=torch.float16).to("spyre"),)
    for arg, dims in zip(args, _ARG_DIMS):
        pnd.name_tensor_dims(arg, dims)
    return args


def _compile_and_measure(model, args, label):
    """Fresh compile under the capture hook; returns (device_us, wall_us, result)."""
    with (
        t_inductor_config.patch("force_disable_caches", True),
        ts_inductor_config.patch("allow_all_ops_in_lx_planning", True),
        patch.object(passes, "CustomPreSchedulingPasses", _CaptureGraphPass),
    ):
        torch.compiler.reset()
        compiled = torch.compile(model, fullgraph=True)
        device_us = measure_device_us_with_noise_check(compiled, args, label=label)
        wall_us = measure_wall_us_with_noise_check(compiled, args, label=label)
        result = compiled(*args).to("cpu")
    return device_us, wall_us, result


def main():
    print(f"=== Phase 1: baseline (untiled), R={R} C={C} ===")
    args = _prep_inputs()
    baseline_device_us, baseline_wall_us, baseline_result = _compile_and_measure(
        fn, args, "baseline"
    )
    buffers = _CaptureGraphPass.captured_buffers
    if not buffers:
        print("\nNo residency-eligible buffers found at all -- nothing for "
              "tiling to improve here.")
        return
    print("\nResidency-eligible buffers (untiled):")
    for b in sorted(buffers, key=lambda b: b["start_tick"]):
        print(f"  {b['name']:<12} {b['full_size_bytes']:>12,} B/core  "
              f"ticks {b['start_tick']}-{b['end_tick']}")

    print(f"\n=== Phase 2: cost model, CG=C={CG} ===")
    full_peak = peak_live_bytes(buffers, CG, CG)
    print(f"Peak co-live bytes at full size (untiled): {full_peak:,} B")
    print(f"LX capacity (per core):                    {LX_CAPACITY:,} B")
    if full_peak <= LX_CAPACITY:
        print("\nAlready fits untiled -- no overflow to demonstrate. Raise C.")
        return

    best_EG = find_best_tile_size(buffers, CG, num_cores=NUM_CORES, lx_capacity=LX_CAPACITY)
    if best_EG is None:
        print("\nNo candidate scored -- nothing to demonstrate.")
        return
    best_K = CG // best_EG
    if best_K == 1:
        print("\nModel says untiled is already optimal -- nothing to demonstrate.")
        return
    print(f"\nModel picks EG={best_EG} -> K={best_K} tiles along C")

    print(f"\n=== Phase 3: tiled (K={best_K}) ===")
    tiled_fn = _make_tiled_softmax(best_K)
    run_coarse_tile_test(
        tiled_fn, _LOOPSPEC_INPUTS, loopspec=LoopSpecCheck(counts=[best_K]),
        correctness=False,
    )

    args = _prep_inputs()
    tiled_device_us, tiled_wall_us, tiled_result = _compile_and_measure(
        tiled_fn, args, f"K={best_K}"
    )
    tiled_buffers = _CaptureGraphPass.captured_buffers
    tiled_peak = peak_live_bytes(tiled_buffers, CG, CG) if tiled_buffers else None

    torch.testing.assert_close(
        tiled_result, baseline_result, atol=0.02, rtol=0.05,
        msg=lambda m: f"tiled result diverged from untiled result\n\n{m}\n",
    )

    print("\n=== Summary ===")
    fits = "fits" if tiled_peak is not None and tiled_peak <= LX_CAPACITY else "still over"
    tiled_peak_str = f"{tiled_peak:,}" if tiled_peak is not None else "n/a"
    print(f"Peak co-live bytes: {full_peak:,} B (untiled, overflow) "
          f"-> {tiled_peak_str} B (K={best_K}, {fits})")
    if baseline_device_us and tiled_device_us:
        delta = (baseline_device_us - tiled_device_us) / baseline_device_us * 100
        print(f"Device time: {baseline_device_us:.3f}us (untiled) -> "
              f"{tiled_device_us:.3f}us (K={best_K})  [{delta:+.1f}%]")
    if baseline_wall_us and tiled_wall_us:
        delta = (baseline_wall_us - tiled_wall_us) / baseline_wall_us * 100
        print(f"Wall time:   {baseline_wall_us:.3f}us (untiled) -> "
              f"{tiled_wall_us:.3f}us (K={best_K})  [{delta:+.1f}%]")
    print("Tiled vs untiled results match (atol=0.02, rtol=0.05).")


if __name__ == "__main__":
    main()
