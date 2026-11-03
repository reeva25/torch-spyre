# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Head-major KV-cache write (indirect scatter) with a symbolic token count.

Mirrors the serving plugin's head-major KV write: the cache is
``[pages, kv_heads, block, head_size]``, viewed as ``[pages*kv_heads*block,
head_size]`` rows, and written once per KV head with ``index_copy`` at row
``(page * kv_heads + head) * block + offset``. The token count ``T`` is symbolic.

The flat view is built outside the compiled function, as the plugin's
``kv_slot_views`` does: a reshape of the carry inside the loop body, or a carry that
starts as a different-rank view of a graph input, is not supported today.

The caller groups the tokens and their row indices into rows of ``G`` with the
same trim-then-split ``for_each_tile`` uses internally, and loops one group per
trip (``tile_size=1``). Row indices travel as ``[T, 1]`` columns, so each trip's
indices sit in adjacent sticks in the default layout.

Data is whole numbers so the device round trip is exact and results are compared
exactly. Sizes include ones below and above the first call's, and only the user
function's compiles are counted.
"""

import unittest
from functools import wraps

import torch
from torch._inductor.compile_fx import compile_fx

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._C import (
    SpyreTensorLayout,
    as_strided_with_layout,
    get_device_dtype,
    get_elem_in_stick,
)
from torch_spyre._inductor.wsr.for_each_tile import for_each_tile
from torch_spyre.constants import DEVICE_NAME

P, H, B, D = 16, 8, 128, 128  # pages, KV heads, tokens per page, head size
ROWS = P * H * B  # rows of the flattened cache view
G = 64  # tokens per trip
T_MIN, T_MAX = 128, 512
T_SPEC = {0: {"min": T_MIN, "max": T_MAX, "granularity": G}}
# First call in the middle, then below and ABOVE it.
SIZES = [256, 128, 448, 192, 512]


def head_major_rows(slots, head):
    """Row of the flattened cache that token slot ``s`` writes for ``head``."""
    page, offset = slots // B, slots % B
    return (page * H + head) * B + offset


def _check_tokens(x):
    rows = x.shape[0]
    torch._check(rows >= T_MIN)
    torch._check(rows <= T_MAX)
    torch._check(rows % G == 0)


def _group(t):
    """[T, ...] -> [T // G, G, ...], spelled like for_each_tile's own split."""
    n = t.shape[0] // G
    return torch.unflatten(t.narrow(0, 0, n * G), 0, (n, G))


def _body(flat, tiles):
    # The carry is the flat [ROWS, D] view the caller passes in.
    x_tile, *row_tiles = tiles  # [1, G, H, D] and H x [1, G, 1]
    for head in range(H):
        flat = flat.index_copy(
            0, row_tiles[head][0].reshape(-1), x_tile[0][:, head, :]
        )
    return flat, None


def kv_write(cache, x, *rows):
    _check_tokens(x)
    final, _ = for_each_tile(
        _body,
        (_group(x), *(_group(r) for r in rows)),
        dims=(0,) * (1 + H),
        tile_size=1,
        init=cache,
    )
    cache.copy_(final)  # write the result back into the caller's cache


def _reference(cache, x, slots):
    flat = cache.reshape(ROWS, D).clone()
    for head in range(H):
        flat.index_copy_(0, head_major_rows(slots, head), x[:, head, :])
    return flat.reshape(P, H, B, D)


def _cache():
    g = torch.Generator().manual_seed(0)
    return torch.randint(-100, 100, (P, H, B, D), generator=g).half()


def _cache_on_device(cache):
    """The cache in the plugin's head_major_kv_layout, viewed flat as [ROWS, D].

    A plain .view() keeps the 4-D layout on the flat view, which compiled code
    misreads. The flat view is given the flat layout of the same memory instead.
    """
    eps = get_elem_in_stick(torch.float16)
    dt = get_device_dtype(torch.float16)
    head_major = SpyreTensorLayout(
        device_size=[P * H, B, D // eps, eps],
        stride_map=[B * D, D, eps, 1],
        device_dtype=dt,
    )
    flat = SpyreTensorLayout(
        device_size=[ROWS, D // eps, eps], stride_map=[D, eps, 1], device_dtype=dt
    )
    pages = cache.to(DEVICE_NAME, device_layout=head_major)
    return as_strided_with_layout(pages, (ROWS, D), (D, 1), pages.storage_offset(), flat)


def _step(T, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-100, 100, (T, H, D), generator=g).half()
    slots = torch.randperm(P * B, generator=g)[:T]  # unique token slots
    return x, slots


def _rows_on_device(slots, dtype, declare):
    out = []
    for head in range(H):
        r = head_major_rows(slots, head).reshape(-1, 1).to(dtype)
        out.append(r.to(DEVICE_NAME, dynamic=T_SPEC) if declare else r.to(DEVICE_NAME))
    return out


class _Counted:
    """torch.compile(fn) that counts only fn's own compiles."""

    def __init__(self, fn, **kwargs):
        self.compiles = 0

        def backend(gm, example_inputs):
            self.compiles += 1
            return compile_fx(gm, example_inputs)

        self.fn = torch.compile(fn, backend=backend, fullgraph=True, **kwargs)


def _with_dynamo_reset(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        torch._dynamo.reset()
        return fn(*args, **kwargs)

    return wrapper


class _ResetEachTest(unittest.TestCase):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if name.startswith("test") and callable(value):
                setattr(cls, name, _with_dynamo_reset(value))

    def _assert_written(self, out, cache, x, slots, msg):
        expected = _reference(cache, x, slots)
        self.assertTrue(torch.equal(out, expected), f"wrong cache: {msg}")
        written = torch.zeros(ROWS, dtype=torch.bool)
        for head in range(H):
            written[head_major_rows(slots, head)] = True
        flat_out, flat_in = out.reshape(ROWS, D), cache.reshape(ROWS, D)
        self.assertTrue(
            torch.equal(flat_out[~written], flat_in[~written]), f"row clobbered: {msg}"
        )


class TestSymbolicHeadMajorKvWrite(_ResetEachTest):
    """One binary for every token count in range, exact at each size."""

    def _run_sizes(self, index_dtype):
        c = _Counted(kv_write)
        cache = _cache()
        for i, T in enumerate(SIZES):
            x, slots = _step(T, seed=i)
            dev = _cache_on_device(cache)
            c.fn(
                dev,
                x.to(DEVICE_NAME, dynamic=T_SPEC),
                *_rows_on_device(slots, index_dtype, declare=True),
            )
            out = dev.cpu().reshape(P, H, B, D)
            self._assert_written(out, cache, x, slots, f"T={T}")
        self.assertEqual(c.compiles, 1, "recompiled within the declared range")

    def test_int64_index(self):
        # The plugin sends int64 slot indices; they are stored as int32 on device.
        self._run_sizes(torch.int64)

    def test_int32_index(self):
        self._run_sizes(torch.int32)


class TestStaticHeadMajorKvWrite(_ResetEachTest):
    """The same grouped loop with a fixed token count, no declaration."""

    def test_fixed_count(self):
        cache = _cache()
        x, slots = _step(T_MIN, seed=0)
        dev = _cache_on_device(cache)
        torch.compile(kv_write, fullgraph=True, dynamic=False)(
            dev,
            x.to(DEVICE_NAME),
            *_rows_on_device(slots, torch.int64, declare=False),
        )
        out = dev.cpu().reshape(P, H, B, D)
        self._assert_written(out, cache, x, slots, f"T={T_MIN}")


class TestScatterKnownLimits(_ResetEachTest):
    """Pinned gaps, so a fix shows up as a test to update."""

    def test_ungrouped_tile_size_g_is_refused(self):
        # The loop tags the G-token axis; resolving it asks for the scatter's
        # output coordinates, which hold the data-dependent slot index.
        def flat_write(cache, x, slots):
            final, _ = for_each_tile(
                lambda c, t: (c.index_copy(0, t[1].reshape(-1), t[0]), None),
                (x, slots),
                dims=(0, 0),
                tile_size=G,
                init=cache,
            )
            return final

        cache = torch.randint(-100, 100, (576, 64)).half()
        x = torch.randint(-100, 100, (T_MIN, 64)).half()
        slots = torch.randperm(576)[:T_MIN].reshape(T_MIN, 1).to(torch.int32)
        with self.assertRaisesRegex(Exception, "indirect symbol"):
            torch.compile(flat_write, fullgraph=True, dynamic=False)(
                cache.to(DEVICE_NAME), x.to(DEVICE_NAME), slots.to(DEVICE_NAME)
            )


if __name__ == "__main__":
    unittest.main()
