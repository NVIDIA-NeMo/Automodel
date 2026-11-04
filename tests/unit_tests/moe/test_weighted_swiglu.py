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

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.moe.megatron import moe_utils


@pytest.fixture(autouse=True)
def eager_kernels(monkeypatch):
    """Keep small CPU semantic checks independent of compiler startup."""
    for activation in ("swiglu", "geglu", "quick_geglu"):
        for name in (activation, f"weighted_{activation}", f"{activation}_back", f"weighted_{activation}_back"):
            monkeypatch.setattr(moe_utils, name, getattr(moe_utils, name)._torchdynamo_orig_callable)
    monkeypatch.setattr(moe_utils, "quick_gelu", moe_utils.quick_gelu._torchdynamo_orig_callable)


@pytest.mark.parametrize("train_router", [False, True])
@pytest.mark.parametrize("constant_weights", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "geglu", "quick_geglu"])
@pytest.mark.parametrize("input_dtype,weight_dtype", [(torch.float64, torch.float64), (torch.bfloat16, torch.float32)])
def test_weighted_activation_matches_autograd_and_skips_unused_router_gradient(
    monkeypatch, train_router, constant_weights, activation, input_dtype, weight_dtype
):
    torch.manual_seed(12)
    values = torch.randn(7, 16, dtype=input_dtype, requires_grad=True)
    weights = torch.ones(7, 1, dtype=weight_dtype) if constant_weights else torch.rand(7, 1, dtype=weight_dtype)
    weights.requires_grad_(train_router)
    grad = torch.randn(7, 8, dtype=input_dtype)
    expected_values = values.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_(train_router)
    gate, up = expected_values.chunk(2, -1)
    if activation == "swiglu":
        activated = F.silu(gate) * up
        function = moe_utils.WeightedSwiGLUFunction
        extra_args = ()
    elif activation == "geglu":
        activated = F.gelu(gate, approximate="tanh") * up
        function = moe_utils.WeightedGEGLUFunction
        extra_args = ()
    else:
        activated = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        function = moe_utils.WeightedQuickGeGLUFunction
        extra_args = (torch.tensor(1.0),)
    expected = (activated * expected_weights).to(input_dtype)
    expected.backward(grad)

    backward_name = f"weighted_{activation}_back"
    backward_kernel = getattr(moe_utils, backward_name)
    requested = []

    def checked_backward(
        g: torch.Tensor, y: torch.Tensor, w: torch.Tensor, *args, need_weight_grad: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Check whether the weighted activation requests routing gradients.

        Args:
            g: Upstream gradient shaped [tokens, intermediate].
            y: Gate/up activations shaped [tokens, 2 * intermediate].
            w: Routing weights shaped [tokens, 1].
            need_weight_grad: Whether routing weights require a gradient.

        Returns:
            Activation gradient shaped like ``y`` and the optional gradient of ``w``.
        """
        requested.append(need_weight_grad)
        input_grad, weight_grad = backward_kernel(g, y, w, *args, need_weight_grad=need_weight_grad)
        assert input_grad.dtype == values.dtype
        assert (weight_grad is not None) == train_router
        return input_grad, weight_grad

    monkeypatch.setattr(moe_utils, backward_name, checked_backward)
    actual = function.apply(values, weights, False, *extra_args)
    actual.backward(grad)
    assert requested == [train_router]
    torch.testing.assert_close(actual, expected)
    if input_dtype == torch.bfloat16:
        # Manual and autograd backward round BF16 intermediates at different
        # points. Bound the norm error without amplifying near-zero entries.
        error = (values.grad.float() - expected_values.grad.float()).norm()
        assert error <= torch.finfo(input_dtype).eps * expected_values.grad.float().norm()
    else:
        torch.testing.assert_close(values.grad, expected_values.grad)
    assert values.grad.dtype == values.dtype
    if train_router:
        if input_dtype == torch.bfloat16:
            error = (weights.grad - expected_weights.grad).norm()
            assert error <= torch.finfo(input_dtype).eps * expected_weights.grad.norm()
        else:
            torch.testing.assert_close(weights.grad, expected_weights.grad)
    else:
        assert weights.grad is None
