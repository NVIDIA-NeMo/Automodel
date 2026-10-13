# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import math
from unittest.mock import patch

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.flop_counter import FlopCounterMode as TorchFlopCounterMode

from nemo_automodel.components.utils import flop_counter
from nemo_automodel.components.utils.flop_counter import FlopCounterMode


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Map x [tokens, hidden] and weight [hidden, output] to [tokens, output]."""
    return torch.nn.functional.silu(x @ weight)


@pytest.mark.parametrize("warm", [False, True])
def test_compiled_forward_backward_cache_and_restoration(warm: bool) -> None:
    torch.compiler.reset()
    normal_backend = CompileCounterWithBackend("inductor")
    compiled = torch.compile(_linear, backend=normal_backend, fullgraph=True)
    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    upstream = torch.randn(2, 4)
    reference = _linear(x, weight)
    reference_grads = torch.autograd.grad(reference, (x, weight), upstream)
    if warm:
        compiled(x, weight).backward(upstream)

    counter = FlopCounterMode(display=False)
    real_compiler = flop_counter.compile_fx

    def compile_with_calibration(graph, inputs):
        # Inductor can benchmark real GEMMs during lazy backward compilation.
        # Such compiler work must be excluded even when it uses real tensors.
        torch.ones(7, 11) @ torch.ones(11, 13)
        return real_compiler(graph, inputs)

    with patch.object(flop_counter, "compile_fx", side_effect=compile_with_calibration) as compiler:
        for _ in range(2):
            x.grad = weight.grad = None
            with counter:
                actual = compiled(x, weight)
                assert counter.get_total_flops() == 2 * 2 * 3 * 4
                actual.backward(upstream)
            assert counter.get_total_flops() == 6 * 2 * 3 * 4
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(x.grad, reference_grads[0])
            torch.testing.assert_close(weight.grad, reference_grads[1])
        # Separate real Inductor forward and backward graphs, reused on call 2.
        assert compiler.call_count == 2
    assert normal_backend.frame_count == int(warm)
    compiled(x, weight).backward(upstream)
    assert normal_backend.frame_count == 1
    assert counter.get_total_flops() == 144


def test_fused_custom_formula_is_counted_on_every_execution() -> None:
    torch.compiler.reset()

    @torch.compile(fullgraph=True)
    def fused(x: torch.Tensor) -> torch.Tensor:
        """Apply elementwise math to x [elements], preserving its shape."""
        return torch.sin(x) * torch.cos(x)

    def mul_flops(a_shape, b_shape, **kwargs):
        return math.prod(a_shape)

    x = torch.randn(17)
    with FlopCounterMode(display=False, custom_mapping={torch.ops.aten.mul: mul_flops}) as counter:
        results = [fused(x) for _ in range(3)]
    assert counter.get_total_flops() == 3 * 17
    for result in results:
        torch.testing.assert_close(result, x.sin() * x.cos())


def test_dynamic_shapes_are_not_counted_using_the_first_shape() -> None:
    torch.compiler.reset()
    compiled = torch.compile(_linear, fullgraph=True, dynamic=True)
    weight = torch.randn(3, 4, requires_grad=True)
    counter = FlopCounterMode(display=False)
    for tokens in (2, 5, 2):
        x = torch.randn(tokens, 3, requires_grad=True)
        with counter:
            compiled(x, weight).sum().backward()
        assert counter.get_total_flops() == 6 * tokens * 3 * 4


def test_graph_break_counts_eager_and_compiled_regions_once() -> None:
    torch.compiler.reset()

    @torch.compiler.disable
    def eager(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Multiply x [tokens, hidden] and weight [hidden, output]."""
        return x @ weight

    @torch.compile
    def mixed(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Sum eager and compiled products of x [tokens, hidden] and weight [hidden, output]."""
        return x @ weight + eager(x, weight)

    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    with FlopCounterMode(display=False) as counter:
        mixed(x, weight).sum().backward()
    assert counter.get_total_flops() == 2 * 6 * 2 * 3 * 4


def test_eager_count_and_exception_cleanup() -> None:
    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    with TorchFlopCounterMode(display=False) as reference:
        _linear(x, weight).sum().backward()
    counter = FlopCounterMode(display=False)
    with pytest.raises(ValueError, match="intentional"):
        with counter:
            _linear(x, weight).sum().backward()
            raise ValueError("intentional")
    assert counter.get_total_flops() == reference.get_total_flops()
    _linear(x, weight).sum().backward()
    assert counter.get_total_flops() == reference.get_total_flops()
    with counter:
        _linear(x, weight).sum().backward()
    assert counter.get_total_flops() == reference.get_total_flops()


@pytest.mark.parametrize("input_grad,weight_grad", [(False, True), (True, False), (False, False)])
def test_counts_only_backward_graphs_that_execute(input_grad: bool, weight_grad: bool) -> None:
    torch.compiler.reset()
    compiled = torch.compile(_linear, fullgraph=True)
    x = torch.randn(2, 3, requires_grad=input_grad)
    weight = torch.randn(3, 4, requires_grad=weight_grad)
    with FlopCounterMode(display=False) as counter:
        result = compiled(x, weight)
        if input_grad or weight_grad:
            result.sum().backward()
    assert counter.get_total_flops() == (1 + input_grad + weight_grad) * 2 * 2 * 3 * 4
