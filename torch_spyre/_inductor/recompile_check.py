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

"""Report recompiles of user code, and keep a compile ledger.

After warm-up, every call to a compiled function should reuse a program. A
recompile means something split it: a size branch, a declaration lost through
``clone``/``view``/a graph break, or a guard the narrowing check could not see.
This hook wraps the callback Dynamo builds for every cache miss, so it sees each
compile, and reports the ones that are recompiles of user code.

torch-spyre's own per-op eager programs (``torch_spyre/ops/eager.py``) and
PyTorch's internals recompile per size by design. Their code lives inside the
torch or torch-spyre package, so they are recognised by file and ignored.

``TORCH_SPYRE_RECOMPILE_CHECK`` sets what a recompile does: ``warn`` (default),
``error``, or ``off``. The ledger records every user compile whatever the mode.
"""

import os
import time
import warnings

import torch

MODE_ENV = "TORCH_SPYRE_RECOMPILE_CHECK"
_MODES = ("warn", "error", "off")
_DEFAULT_MODE = "warn"

_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__)) + os.sep
_SPYRE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

# {"name (file:line)": {"compiles": int, "seconds": float}}
_ledger = {}


class RecompileError(RuntimeError):
    """User code recompiled, in ``error`` mode."""


class RecompileWarning(UserWarning):
    """User code recompiled."""


def _mode():
    mode = os.environ.get(MODE_ENV, _DEFAULT_MODE).lower()
    if mode not in _MODES:
        raise ValueError(f"{MODE_ENV}={mode!r}: expected one of {', '.join(_MODES)}")
    return mode


def _is_internal(filename):
    path = os.path.abspath(filename)
    return path.startswith(_TORCH_DIR) or path.startswith(_SPYRE_DIR)


def _reasons(entries, frame, callback):
    """The guards that failed, as PyTorch words them; [] if unavailable."""
    from torch._dynamo.eval_frame import innermost_fn
    from torch._dynamo.guards import get_and_maybe_log_recompilation_reasons

    backend = innermost_fn(callback)
    for arg in (entries, entries[0]):  # a list on torch 2.13, one entry before
        try:
            return get_and_maybe_log_recompilation_reasons(
                arg, frame, backend, skip_logging=True
            )
        except Exception:  # noqa: BLE001  the reason is a nicety, never fatal
            continue
    return []


def compile_ledger():
    """A copy of the ledger: compiles and seconds per user function."""
    return {name: dict(row) for name, row in _ledger.items()}


def reset_compile_ledger():
    _ledger.clear()


def _wrap(callback):
    from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list

    def wrapped(frame, *args, **kwargs):
        code = frame.f_code
        if _is_internal(code.co_filename):
            return callback(frame, *args, **kwargs)

        name = f"{code.co_name} ({code.co_filename}:{code.co_firstlineno})"
        report = None
        mode = _mode()
        if mode != "off":
            entries = _debug_get_cache_entry_list(code)
            if entries:
                reasons = "; ".join(_reasons(entries, frame, callback)) or "unknown"
                report = (
                    f"Warning! {name} recompiled (compile {len(entries) + 1}). "
                    f"Reason: {reasons}"
                )
        if report and mode == "error":
            raise RecompileError(report)
        if report:
            warnings.warn(report, RecompileWarning, stacklevel=2)

        start = time.perf_counter()
        try:
            return callback(frame, *args, **kwargs)
        finally:
            row = _ledger.setdefault(name, {"compiles": 0, "seconds": 0.0})
            row["compiles"] += 1
            row["seconds"] += time.perf_counter() - start

    # Dynamo compares backends through this attribute; keep the original's.
    wrapped._torchdynamo_orig_backend = getattr(
        callback, "_torchdynamo_orig_backend", callback
    )
    return wrapped


def install_recompile_check():
    """Wrap Dynamo's per-miss callback once. Safe to call more than once."""
    import torch._dynamo.eval_frame as ef

    original = getattr(ef, "_callback_from_stance", None)
    if original is None or getattr(original, "_spyre_recompile_check", False):
        return

    def callback_from_stance(callback):
        cb = original(callback)
        if cb in (False, None):
            return cb
        return _wrap(cb)

    callback_from_stance._spyre_recompile_check = True
    ef._callback_from_stance = callback_from_stance
