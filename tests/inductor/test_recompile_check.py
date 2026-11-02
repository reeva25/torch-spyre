# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""The recompile check: user recompiles are reported, torch-spyre's are not.

The CPU tests compile with ``backend="eager"`` and run anywhere. The device
tests need a Spyre device and the ``dynamic=`` reservation of #5179.
"""

import os
import unittest
import warnings
from unittest import mock

import torch

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import recompile_check
from torch_spyre._inductor.recompile_check import (
    MODE_ENV,
    RecompileError,
    RecompileWarning,
    compile_ledger,
    install_recompile_check,
    reset_compile_ledger,
)
from torch_spyre.constants import DEVICE_NAME

COLS = 8


def _reports(caught):
    return [str(w.message) for w in caught if issubclass(w.category, RecompileWarning)]


def _run(fn, *inputs):
    """Call ``fn`` on each input in turn, recording recompile warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for x in inputs:
            fn(x)
    return _reports(caught)


class _Base(unittest.TestCase):
    def setUp(self):
        install_recompile_check()
        torch._dynamo.reset()
        reset_compile_ledger()


class TestRecompileCheckCPU(_Base):
    @staticmethod
    def _block():
        @torch.compile(backend="eager")
        def block(x):
            return x.relu() + 1

        return block

    def test_recompile_warns_by_default_with_the_reason(self):
        reports = _run(self._block(), torch.randn(128, COLS), torch.randn(192, COLS))
        self.assertEqual(len(reports), 1)
        self.assertIn("block", reports[0])
        self.assertIn("recompiled (compile 2)", reports[0])
        self.assertIn("expected 128, actual 192", reports[0])

    def test_cached_call_is_not_reported(self):
        reports = _run(self._block(), torch.randn(128, COLS), torch.randn(128, COLS))
        self.assertEqual(reports, [])

    def test_error_mode_stops_the_recompile(self):
        block = self._block()
        block(torch.randn(128, COLS))
        with mock.patch.dict(os.environ, {MODE_ENV: "error"}):
            with self.assertRaisesRegex(RecompileError, "recompiled"):
                block(torch.randn(192, COLS))

    def test_off_mode_reports_nothing(self):
        with mock.patch.dict(os.environ, {MODE_ENV: "off"}):
            reports = _run(
                self._block(), torch.randn(128, COLS), torch.randn(192, COLS)
            )
        self.assertEqual(reports, [])

    def test_ledger_counts_and_times_user_compiles(self):
        _run(self._block(), torch.randn(128, COLS), torch.randn(192, COLS))
        rows = [row for name, row in compile_ledger().items() if "block" in name]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["compiles"], 2)
        self.assertGreater(rows[0]["seconds"], 0.0)

    def test_torch_and_torch_spyre_files_are_internal(self):
        eager_py = os.path.join(recompile_check._SPYRE_DIR, "ops", "eager.py")
        torch_py = os.path.join(recompile_check._TORCH_DIR, "_compile.py")
        self.assertTrue(recompile_check._is_internal(eager_py))
        self.assertTrue(recompile_check._is_internal(torch_py))
        self.assertFalse(recompile_check._is_internal(__file__))

    def test_install_is_idempotent(self):
        import torch._dynamo.eval_frame as ef

        hook = ef._callback_from_stance
        install_recompile_check()
        self.assertIs(ef._callback_from_stance, hook)

    def test_graph_break_reports_both_pieces(self):
        @torch.compile(backend="eager")
        def broken(x):
            y = x.relu()
            torch._dynamo.graph_break()
            return y * 3

        reports = _run(broken, torch.randn(128, COLS), torch.randn(192, COLS))
        self.assertEqual(len(reports), 2)
        self.assertTrue(any("resume_in_broken" in r for r in reports))

    def test_graph_break_inside_torch_spyre_code_still_reports_user_pieces(self):
        # A helper whose code lives in the torch-spyre package, as
        # for_each_tile does. Dynamo resumes in the user's frame after the
        # break, so the user's pieces are still checked; only the helper's own
        # pieces are skipped.
        src = (
            "import torch\n"
            "def spyre_helper(x):\n"
            "    y = x * 2\n"
            "    torch._dynamo.graph_break()\n"
            "    return y + 1\n"
        )
        helper_file = os.path.join(recompile_check._SPYRE_DIR, "_test_helper.py")
        namespace = {}
        exec(compile(src, helper_file, "exec"), namespace)  # noqa: S102
        spyre_helper = namespace["spyre_helper"]

        @torch.compile(backend="eager")
        def calls_helper(x):
            return spyre_helper(x.relu()) * 3

        reports = _run(calls_helper, torch.randn(128, COLS), torch.randn(192, COLS))
        self.assertEqual(len(reports), 2)
        self.assertFalse(any("_test_helper.py" in r for r in reports))
        self.assertTrue(any("resume_in_calls_helper" in r for r in reports))


class TestRecompileCheckOnDevice(_Base):
    def test_eager_ops_at_new_sizes_are_not_reported(self):
        # Each eager op is a small per-op compile that recompiles per size by
        # design. Even in error mode they must keep working.
        with mock.patch.dict(os.environ, {MODE_ENV: "error"}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for n in (128, 192):
                    x = torch.randn(n, 64, dtype=torch.float16).to(DEVICE_NAME)
                    x.clone()
                    x + x
        self.assertEqual(_reports(caught), [])

    def test_lost_declaration_recompile_is_reported(self):
        # A view drops the declaration, so the next block sees an ordinary
        # tensor. Under dynamic=None, the supported setting, its second size
        # is an automatic-dynamic recompile and must be reported. Spyre may
        # then refuse that compile (an unbounded size outside a loop); the
        # report is made before the compile either way.
        @torch.compile(dynamic=None)
        def block(h):
            return h * 2.0

        spec = {0: {"min": 128, "max": 512, "granularity": 64}}
        first, second = (
            torch.randn(n, 64, dtype=torch.float16)
            .to(DEVICE_NAME, dynamic=spec)
            .view(-1, 64)
            for n in (128, 192)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            block(first)
            try:
                block(second)
            except Exception as e:  # noqa: BLE001
                if "kDimension" not in str(e):
                    raise
        reports = _reports(caught)
        self.assertEqual(len(reports), 1)
        self.assertIn("block", reports[0])

    def test_torch_spyre_internal_compiles_are_not_reported(self):
        # Custom ops and decompositions that torch-spyre compiles behind the
        # user's back, each at two sizes. They recompile per size by design,
        # so even error mode must let them through.
        w = torch.ones(64, dtype=torch.float16).to(DEVICE_NAME)
        paths = {
            "copy_from_d2d": lambda x: torch.empty_like(x).copy_(x),
            "to_dtype_d2d": lambda x: x.to(torch.float32),
            "rms_norm": lambda x: torch.ops.spyre.rms_norm(x, [64], w, 1e-5),
            "decomposition (sign)": lambda x: x.sign(),
        }
        for name, op in paths.items():
            with self.subTest(name):
                xs = [
                    torch.randn(n, 64, dtype=torch.float16).to(DEVICE_NAME)
                    for n in (128, 192)
                ]
                torch._dynamo.reset()
                try:
                    with mock.patch.dict(os.environ, {MODE_ENV: "off"}):
                        for x in xs:
                            op(x)
                except Exception as e:  # noqa: BLE001
                    self.skipTest(f"{name} does not run here: {type(e).__name__}")
                torch._dynamo.reset()
                with mock.patch.dict(os.environ, {MODE_ENV: "error"}):
                    reports = _run(op, *xs)
                self.assertEqual(reports, [])


if __name__ == "__main__":
    unittest.main()
