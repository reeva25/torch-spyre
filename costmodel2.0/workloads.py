# Phase 1 workloads for cost model 2.0. The outermost dimension is the tiled
# one; its size is CG (tile size == granularity in the symbolic-shapes design).
#
# Each one is written the way the symbolic-shapes bridge (HLD sec. 8.3) would
# author it: a map-mode for_each_tile over dim 0, which is the outer, varying
# axis. Weights are INVARIANT (dims=None) and every other operand is SLICED
# on dim 0. Every op in every body reduces over a STATIC axis only, so every
# tile stands alone (HLD sec. 5.2). Nothing here is Phase 2.
#
# The whole op chain goes inside ONE for_each_tile. That keeps the
# intermediates tile-sized, which is what gives the cost model something to
# decide. (The bridge could instead wrap each op separately; see the README's
# ideation list.)
#
# Two "control" workloads (add, layernorm) fit in LX at CG by a wide margin,
# so the right answer should be EG = CG and any smaller EG is pure loop
# overhead. Two "pressure" workloads (softmax, mlp) are sized so the tile at
# CG overflows LX, which is where a smaller EG might pay off.

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class Workload:
    name: str
    description: str
    cg: int  # default outermost size == contract granularity (override: --cg)
    make_operands: Callable[[int], tuple]  # outermost size -> CPU fp16 tensors
    dims: tuple  # for_each_tile dims: 0 = SLICE on dim 0, None = INVARIANT
    body: Callable  # (*tiles) -> out_tile; also valid on the full tensors
    atol: float = 0.1  # tolerance for tiled-vs-baseline result comparison
    rtol: float = 0.1


def _rand(*shape, scale=1.0):
    return (torch.randn(*shape) * scale).half()


# ── controls ────────────────────────────────────────────────────────────────


def _add_operands(n):
    return (_rand(n, 4096), _rand(n, 4096))


ADD = Workload(
    name="add",
    description="z = x + y, x and y are (CG, 4096), both sliced (HLD sec. 9.1)",
    cg=64,
    make_operands=_add_operands,
    dims=(0, 0),
    body=lambda x, y: x + y,
)

_LN_H = 8192


def _layernorm_operands(n):
    return (_rand(n, _LN_H), _rand(_LN_H), _rand(_LN_H))


LAYERNORM = Workload(
    name="layernorm",
    description=f"layer_norm over hidden, x is (CG, {_LN_H}), weight/bias invariant",
    cg=128,
    make_operands=_layernorm_operands,
    dims=(0, None, None),
    body=lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b),
)

# ── pressure cases ──────────────────────────────────────────────────────────

_SM_C = 32768


def _softmax_operands(n):
    return (_rand(n, _SM_C),)


SOFTMAX = Workload(
    name="softmax",
    description=f"softmax over the last (static) axis, x is (CG, {_SM_C})",
    cg=512,
    make_operands=_softmax_operands,
    dims=(0,),
    body=lambda x: torch.softmax(x, dim=-1),
    atol=1e-3,  # outputs are ~1/C, so atol has to be small to mean anything
    rtol=0.1,
)

_DIN, _DH, _DOUT = 256, 32768, 256


def _mlp_operands(n):
    torch.manual_seed(0xC0A75E)
    return (
        _rand(n, _DIN),
        _rand(_DH, _DIN, scale=_DIN**-0.5),
        _rand(_DH, scale=0.1),
        _rand(_DOUT, _DH, scale=_DH**-0.5),
        _rand(_DOUT, scale=0.1),
    )


MLP = Workload(
    name="mlp",
    description=(
        f"linear -> silu -> linear, x is (CG, {_DIN}), Dh={_DH}, "
        "all four weight/bias tensors invariant (HLD sec. 9.3 / 9.5)"
    ),
    cg=512,
    make_operands=_mlp_operands,
    dims=(0, None, None, None, None),
    body=lambda x, w1, b1, w2, b2: F.linear(F.silu(F.linear(x, w1, b1)), w2, b2),
    atol=0.1,
    rtol=0.1,
)

ALL = {w.name: w for w in (ADD, LAYERNORM, SOFTMAX, MLP)}
