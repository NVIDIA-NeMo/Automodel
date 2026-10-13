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
    for name in ("swiglu", "weighted_swiglu", "swiglu_back", "weighted_swiglu_back"):
        monkeypatch.setattr(moe_utils, name, getattr(moe_utils, name)._torchdynamo_orig_callable)


@pytest.mark.parametrize("train_router", [False, True])
@pytest.mark.parametrize("constant_weights", [False, True])
def test_weighted_swiglu_matches_autograd_and_skips_unused_router_gradient(monkeypatch, train_router, constant_weights):
    torch.manual_seed(12)
    values = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    weights = torch.ones(7, 1, dtype=torch.float64) if constant_weights else torch.rand(7, 1, dtype=torch.float64)
    weights.requires_grad_(train_router)
    grad = torch.randn(7, 8, dtype=torch.float64)
    expected_values = values.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_(train_router)
    gate, up = expected_values.chunk(2, -1)
    expected = F.silu(gate) * up * expected_weights
    expected.backward(grad)

    backward_kernel = moe_utils.weighted_swiglu_back
    requested = []

    def checked_backward(
        g: torch.Tensor, y: torch.Tensor, w: torch.Tensor, *, need_weight_grad: bool = True
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
        input_grad, weight_grad = backward_kernel(g, y, w, need_weight_grad=need_weight_grad)
        assert (weight_grad is not None) == train_router
        return input_grad, weight_grad

    monkeypatch.setattr(moe_utils, "weighted_swiglu_back", checked_backward)
    actual = moe_utils.WeightedSwiGLUFunction.apply(values, weights, False)
    actual.backward(grad)
    assert requested == [train_router]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(values.grad, expected_values.grad)
    if train_router:
        torch.testing.assert_close(weights.grad, expected_weights.grad)
    else:
        assert weights.grad is None
