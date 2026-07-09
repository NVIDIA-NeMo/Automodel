# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for async-TP linear graph shaping (_tp_linear helpers and TPLinear)."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components._tp_linear import (
    _async_tp_linear,
    _is_async_tp_linear_enabled,
)
from nemo_automodel.components.distributed.parallel_styles import TPLinear


@pytest.fixture
def micro_pipeline_tp_enabled():
    """Enable torch._inductor.config._micro_pipeline_tp and restore it afterwards."""
    original = torch._inductor.config._micro_pipeline_tp
    torch._inductor.config._micro_pipeline_tp = True
    yield
    torch._inductor.config._micro_pipeline_tp = original


def _capture_compiled_graphs(module: nn.Module, x: torch.Tensor) -> tuple[list[torch.fx.GraphModule], torch.Tensor]:
    """Compile ``module`` with a graph-capturing backend and run it on ``x``.

    Args:
        module: Module to compile; its forward must accept a single tensor.
        x: Input activations of shape ``[..., in_features]``.

    Returns:
        A tuple of (captured Dynamo FX graphs, output tensor of shape
        ``[..., out_features]``).
    """
    graphs: list[torch.fx.GraphModule] = []

    def _backend(gm: torch.fx.GraphModule, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(module, backend=_backend, fullgraph=True)
    out = compiled(x)
    return graphs, out


def _call_function_targets(graphs: list[torch.fx.GraphModule]) -> set:
    """Collect call_function targets from a list of Dynamo FX graphs."""
    return {node.target for gm in graphs for node in gm.graph.nodes if node.op == "call_function"}


class TestIsAsyncTpLinearEnabled:
    """Tests for the _is_async_tp_linear_enabled gate."""

    def test_false_in_eager_even_with_flag_set(self, micro_pipeline_tp_enabled):
        """The gate must stay closed outside torch.compile tracing."""
        assert not torch.compiler.is_compiling()
        assert _is_async_tp_linear_enabled() is False

    def test_false_when_compiling_without_flag(self):
        """The gate must stay closed when _micro_pipeline_tp is not set."""
        assert torch._inductor.config._micro_pipeline_tp is False
        with patch("torch.compiler.is_compiling", return_value=True):
            assert _is_async_tp_linear_enabled() is False

    def test_true_when_compiling_with_flag(self, micro_pipeline_tp_enabled):
        """The gate opens only under compile with _micro_pipeline_tp set."""
        with patch("torch.compiler.is_compiling", return_value=True):
            assert _is_async_tp_linear_enabled() is True


class TestAsyncTpLinearNumerics:
    """Numerical equivalence of _async_tp_linear against F.linear."""

    @pytest.mark.parametrize("shape", [(8, 16), (2, 5, 16)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_matches_f_linear_forward(self, shape, use_bias):
        """_async_tp_linear must match F.linear for 2-D and 3-D inputs."""
        torch.manual_seed(0)
        x = torch.randn(*shape)
        weight = torch.randn(12, 16)
        bias = torch.randn(12) if use_bias else None

        out = _async_tp_linear(x, weight, bias)
        ref = F.linear(x, weight, bias)

        assert out.shape == ref.shape
        assert torch.allclose(out, ref)

    def test_matches_f_linear_backward(self):
        """Gradients through _async_tp_linear must match F.linear."""
        torch.manual_seed(0)
        x = torch.randn(2, 5, 16, requires_grad=True)
        weight = torch.randn(12, 16, requires_grad=True)
        bias = torch.randn(12, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)

        grad = torch.randn(2, 5, 12)
        _async_tp_linear(x, weight, bias).backward(grad)
        F.linear(x_ref, weight_ref, bias_ref).backward(grad)

        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(weight.grad, weight_ref.grad)
        assert torch.allclose(bias.grad, bias_ref.grad)


class TestTPLinearGraphShaping:
    """TPLinear.forward must emit the graph async-TP fusion can pattern-match."""

    def _make_tp_linear(self) -> nn.Linear:
        torch.manual_seed(0)
        linear = nn.Linear(16, 12)
        linear.__class__ = TPLinear
        return linear

    def test_async_tp_mode_emits_native_linear(self, micro_pipeline_tp_enabled):
        """With _micro_pipeline_tp set, compile must trace F.linear, not bmm."""
        linear = self._make_tp_linear()
        x = torch.randn(2, 5, 16)

        graphs, out = _capture_compiled_graphs(linear, x)
        targets = _call_function_targets(graphs)

        assert torch.bmm not in targets
        assert F.linear in targets
        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))

    def test_default_compile_path_keeps_bmm(self):
        """Without the flag, the DTensor-safe bmm path must be preserved."""
        assert torch._inductor.config._micro_pipeline_tp is False
        linear = self._make_tp_linear()
        x = torch.randn(2, 5, 16)

        graphs, out = _capture_compiled_graphs(linear, x)
        targets = _call_function_targets(graphs)

        assert torch.bmm in targets
        assert F.linear not in targets
        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))

    def test_eager_path_unaffected_by_flag(self, micro_pipeline_tp_enabled):
        """Eager TPLinear.forward must not change when only the flag is set."""
        linear = self._make_tp_linear()
        x = torch.randn(2, 5, 16)

        out = linear(x)

        assert torch.allclose(out, F.linear(x, linear.weight, linear.bias))
