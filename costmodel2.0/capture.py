# Cost model 2.0 -- capture + test plumbing (needs torch_spyre and the device).
#
# The MODEL's input is one pre-solver capture of the compile at tile = CG
# (build_capture below). Everything else in this file -- recompiling at a
# chosen EG, reading post-solver placements, timing -- belongs to the TEST
# harness, same as the demo_* scripts in costmodel/, and is not part of what
# the cost model would cost inside the compiler.
#
# A candidate is compiled as the symbolic-shapes bridge would author it,
#     for_each_tile(body, operands, dims=..., tile_size=EG, out_dim=0)
# with a concrete outermost size CG, so trips = CG / EG.

import time
from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch._inductor.dependencies import MemoryDep
from torch.profiler import ProfilerActivity, profile

import torch_spyre  # noqa: F401 -- registers the "spyre" device + Inductor backend
from torch_spyre._inductor import config as ts_config
from torch_spyre._inductor import passes
from torch_spyre._inductor.scratchpad.allocator import (
    ScratchpadAllocator,
    _lx_planning_size,
)
from torch_spyre._inductor.scratchpad.lx_relayout import collect_lx_relayout_plans
from torch_spyre._inductor.scratchpad.utils import (
    calculate_liveness,
    get_ncores_for_buffers,
    mem_usage_by_buf,
)
from torch_spyre._inductor.wsr import for_each_tile
from torch_spyre.constants import DEVICE_NAME
from torch_spyre.streams import synchronize

from model import Capture


def lx_capacity_bytes() -> int:
    return _lx_planning_size()


# ── model input: the buffer list (from costmodel/buffer_list_with_residency_filter.py)
# Unchanged except for comments. Known gap kept: restickify sources/
# destinations are not patched in (costmodel/residency-filter-restickify-gap.md).


def build_buffers_with_residency_filter(graph, division_is_fixed: bool = True) -> list[dict]:
    allocator = ScratchpadAllocator(layout_planning=None, size=0)  # solver never runs here

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
        if not uses:
            continue
        if buf_name in mem_usage:
            if op_reasons.get(buf_name) is not None:
                continue
            size = mem_usage[buf_name].get("size_per_core", -1)
            if size < 0:
                continue
        elif buf_name in graph.graph_input_names:
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
            continue

        buffers.append(
            {
                "name": buf_name,
                "full_size_bytes": size,
                "start_tick": uses[0],
                "end_tick": uses[-1] + 1,
            }
        )
    return buffers


# ── model input: which buffers shrink with EG ──────────────────────────────


def classify_scaling(graph, sliced_shapes: set, invariant_shapes: set) -> tuple[dict, list]:
    """buffer name -> True if it holds rows of the tiled (outermost) axis.

    Pure dataflow, no loop metadata (the baseline compile at tile = CG has a
    single trip, so there may be no loop to read metadata from):
      - a graph input scales if its shape is a SLICED operand's shape
        (dims=0), and not if it is an INVARIANT operand's (dims=None);
      - an op's output scales if it reads anything that scales.
    Phase 1 bodies never reduce over the outermost axis, so anything computed
    from a sliced operand keeps that axis and shrinks with the tile. Anything
    computed only from weights (e.g. a restickified weight) does not.

    Second return value: graph inputs whose shape matched both kinds or
    neither -- the classification guessed for those, so the caller prints them.
    """
    scales, unsure = {}, []
    for name in graph.graph_input_names:
        try:
            shape = tuple(int(x) for x in graph.graph_inputs[name].get_size())
        except (TypeError, AttributeError):
            continue  # symbolic / non-tensor input (e.g. the loop counter)
        s, i = shape in sliced_shapes, shape in invariant_shapes
        scales[name] = s and not i
        if s == i:
            unsure.append(name)
    for op in graph.operations:  # topological order
        reads = op.get_read_writes().reads
        out = any(scales.get(d.name, False) for d in reads if isinstance(d, MemoryDep))
        for buf in op.get_outputs():
            scales[buf.get_name()] = out
    return scales, unsure


class _CapturePasses(passes.CustomPreSchedulingPasses):
    """The real pipeline plus two read-only probes around LX planning.
    BEFORE planning: the model's input. AFTER: test-harness diagnostic only."""

    last: dict = {}
    # set by compile_and_capture before each compile
    sliced_shapes: set = set()
    invariant_shapes: set = set()

    def __init__(self):
        super().__init__()
        at = self.passes.index(passes._maybe_scratchpad_planning)
        self.passes.insert(at, self._before_planning)
        self.passes.insert(at + 2, self._after_planning)

    def _before_planning(self, graph) -> None:
        buffers = build_buffers_with_residency_filter(graph)
        scaling, unsure = classify_scaling(
            graph, _CapturePasses.sliced_shapes, _CapturePasses.invariant_shapes
        )
        for b in buffers:
            b["scales"] = scaling.get(b["name"], False)
        _CapturePasses.last = {"buffers": buffers, "unsure_inputs": unsure}

    def _after_planning(self, graph) -> None:
        placed = set()
        for b in _CapturePasses.last["buffers"]:
            try:
                if "lx" in (graph.get_buffer(b["name"]).get_layout().allocation or {}):
                    placed.add(b["name"])
            except Exception:  # noqa: BLE001 -- inputs/views without a tiled layout
                pass
        _CapturePasses.last["lx_placed"] = placed


# ── compiling a candidate (test harness) ───────────────────────────────────


def make_tiled_fn(workload, eg: int):
    """What the bridge would author for this workload, with tile_size = eg."""
    body, dims = workload.body, workload.dims

    def tiled(*operands):
        _, out = for_each_tile(
            lambda _carry, tiles: (None, body(*tiles)),
            operands,
            dims=dims,
            tile_size=eg,
            out_dim=0,
        )
        return out

    return tiled


def invariant_bytes(workload, operands) -> int:
    """Total bytes of the dims=None operands, re-read every iteration."""
    return sum(t.numel() * t.element_size()
               for t, d in zip(operands, workload.dims) if d is None)


def compile_and_capture(workload, eg: int, args, allow_all_ops_in_lx: bool = False):
    """Fresh compile of the tile_size=eg function; returns (compiled, capture dict)."""
    with (
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("allow_all_ops_in_lx_planning", allow_all_ops_in_lx),
        patch.object(passes, "CustomPreSchedulingPasses", _CapturePasses),
    ):
        torch.compiler.reset()
        _CapturePasses.last = {}
        _CapturePasses.sliced_shapes = {
            tuple(t.shape) for t, d in zip(args, workload.dims) if d is not None
        }
        _CapturePasses.invariant_shapes = {
            tuple(t.shape) for t, d in zip(args, workload.dims) if d is None
        }
        compiled = torch.compile(make_tiled_fn(workload, eg), fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        compiled(*args)
        synchronize()
        info = dict(_CapturePasses.last)
        info["compile_s"] = time.perf_counter() - t0
    return compiled, info


def to_model_capture(workload, cg, info, args) -> Capture:
    return Capture(cg=cg, buffers=info["buffers"],
                   invariant_bytes=invariant_bytes(workload, args))


def print_buffers(info, lx_capacity):
    print(f"  {'name':<34} {'bytes/core':>12} {'ticks':>9}  {'scales':<6}  after solver")
    for b in sorted(info["buffers"], key=lambda b: (b["start_tick"], b["name"])):
        where = "LX" if b["name"] in info.get("lx_placed", ()) else "HBM"
        print(f"  {b['name']:<34} {b['full_size_bytes']:>12,} "
              f"{b['start_tick']:>4}-{b['end_tick']:<4}  {str(b['scales']):<6}  {where}")
    if info.get("unsure_inputs"):
        print(f"  !! graph inputs {info['unsure_inputs']} matched both or neither a sliced "
              "and an invariant operand shape -- check their `scales` flag by hand.")


# ── timing (from costmodel/demo_mlp_swapped_dims.py) ───────────────────────

WARMUP_ITERS = 5
TIMED_ITERS = 30
REPEAT_TRIALS = 5
NOISE_CV_WARN = 0.10


def _mean_cv(values):
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return mean, (std / mean if mean else float("inf"))


def _device_us_once(fn, args):
    for _ in range(WARMUP_ITERS):
        fn(*args)
    synchronize()
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "PrivateUse1"):
        activities.append(ProfilerActivity.PrivateUse1)
    with profile(activities=activities) as prof:
        for _ in range(TIMED_ITERS):
            fn(*args)
        synchronize()
    total, matched = 0.0, False
    for evt in prof.events():
        if "spyre_kernel" in evt.name or "sdsc" in evt.name:
            us = getattr(evt, "self_device_time_total", None)
            if us:
                matched = True
                total += us
    return total / TIMED_ITERS if matched else None


def _wall_us_once(fn, args):
    for _ in range(WARMUP_ITERS):
        fn(*args)
    synchronize()
    samples = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        fn(*args)
        synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    return samples[len(samples) // 2]


def measure(fn, args, label) -> float | None:
    """Device time if the profiler matched kernel events, else wall time."""
    device = [_device_us_once(fn, args) for _ in range(REPEAT_TRIALS)]
    if all(d is not None for d in device):
        mean, cv, kind = *_mean_cv(device), "device"
    else:
        mean, cv = _mean_cv([_wall_us_once(fn, args) for _ in range(REPEAT_TRIALS)])
        kind = "wall (no device events matched)"
    flag = "  <- NOISY" if cv > NOISE_CV_WARN else ""
    print(f"  {label}: {mean:,.1f} us {kind}, cv={cv:.1%}{flag}")
    return mean
