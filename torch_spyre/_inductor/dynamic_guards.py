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

"""Compile-time checks on dynamic dimensions of Spyre inputs (#4384)."""

from .constants import DEVICE_NAME


class SpyreDynamicShapeError(ValueError):
    """A dynamic dimension that the compiled code cannot safely serve."""


def _traced_inputs(gm, example_inputs):
    """Pair each graph input's traced value with the real tensor passed in."""
    import torch

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for node, real in zip(placeholders, example_inputs):
        traced = node.meta.get("example_value")
        if (
            isinstance(real, torch.Tensor)
            and isinstance(traced, torch.Tensor)
            and real.device.type == DEVICE_NAME
        ):
            yield _input_name(node), real, traced


def _input_name(node):
    """The name PyTorch's own errors use for an input, such as L['x']."""
    source = getattr(node.meta.get("grapharg"), "source", None)
    if source is None:
        return node.name
    name = source.name
    return name() if callable(name) else name


def check_undeclared_dynamic_dims(gm, example_inputs):
    """Refuse a dynamic dimension of a Spyre input that has no declaration.

    Dynamo makes a dimension dynamic on its own once it sees a second size, so
    a tensor moved with a plain .to("spyre") can reach here with a symbolic
    dimension and no min, max or granularity. Nothing then tiles it, and the
    kernel faults on the device.
    """
    import torch
    from torch_spyre._C import get_reserved_dims

    for name, real, traced in _traced_inputs(gm, example_inputs):
        declared = get_reserved_dims(real) or {}
        for dim, size in enumerate(traced.size()):
            if not isinstance(size, torch.SymInt) or dim in declared:
                continue
            raise SpyreDynamicShapeError(
                f"Input {name} dim {dim} is dynamic but has no declaration, so "
                f"no compiled loop can serve it. Declare it when moving the "
                f'tensor: .to("spyre", dynamic={{{dim}: {{"min": ..., '
                f'"max": ..., "granularity": ...}}}})'
            )
