# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Tests for Gemma4-specific layer components:

- GEGLU activation functions (geglu, weighted_geglu, geglu_back, weighted_geglu_back)
- WeightedGEGLUFunction autograd custom op
- weighted_bias_geglu_impl entry point
- Gemma4Gate router logic
"""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.moe.megatron.moe_utils import (
    WeightedGEGLUFunction,
    geglu,
    geglu_back,
    weighted_bias_geglu_impl,
    weighted_geglu,
    weighted_geglu_back,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# geglu forward
# ---------------------------------------------------------------------------
class TestGeglu:
    def test_output_shape_is_half_input(self, device):
        y = torch.randn(4, 8, 128, device=device)
        out = geglu(y)
        assert out.shape == (4, 8, 64)

    def test_matches_manual_gelu_tanh_times_up(self, device):
        y = torch.randn(2, 6, 64, device=device)
        y_gate, y_up = torch.chunk(y, 2, -1)
        expected = F.gelu(y_gate, approximate="tanh") * y_up

        out = geglu(y)
        torch.testing.assert_close(out, expected)

    def test_zero_input_produces_zero(self, device):
        y = torch.zeros(1, 4, 32, device=device)
        out = geglu(y)
        torch.testing.assert_close(out, torch.zeros(1, 4, 16, device=device))

    def test_supports_bfloat16(self, device):
        y = torch.randn(2, 4, 32, device=device, dtype=torch.bfloat16)
        out = geglu(y)
        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# weighted_geglu forward
# ---------------------------------------------------------------------------
class TestWeightedGeglu:
    def test_output_shape(self, device):
        y = torch.randn(4, 8, 64, device=device)
        weights = torch.ones(4, 8, 1, device=device)
        out = weighted_geglu(y, weights)
        assert out.shape == (4, 8, 32)

    def test_unit_weights_match_plain_geglu(self, device):
        y = torch.randn(2, 6, 64, device=device)
        weights = torch.ones(2, 6, 1, device=device)
        plain = geglu(y)

        out = weighted_geglu(y, weights)
        torch.testing.assert_close(out, plain)

    def test_zero_weights_produce_zero(self, device):
        y = torch.randn(2, 4, 32, device=device)
        weights = torch.zeros(2, 4, 1, device=device)
        out = weighted_geglu(y, weights)
        torch.testing.assert_close(out, torch.zeros(2, 4, 16, device=device))

    def test_preserves_dtype(self, device):
        y = torch.randn(2, 4, 32, device=device, dtype=torch.bfloat16)
        weights = torch.ones(2, 4, 1, device=device, dtype=torch.bfloat16)
        out = weighted_geglu(y, weights)
        assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# geglu_back (manual backward)
# ---------------------------------------------------------------------------
class TestGegluBack:
    def test_output_shape_matches_input(self, device):
        y = torch.randn(2, 4, 64, device=device)
        g = torch.randn(2, 4, 32, device=device)
        grad = geglu_back(g, y)
        assert grad.shape == y.shape

    def test_grad_matches_autograd(self, device):
        """Verify geglu_back matches PyTorch autograd for the same forward."""
        y = torch.randn(2, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
        out = geglu(y)
        g = torch.randn_like(out)
        out.backward(g)
        autograd_grad = y.grad.clone()

        manual_grad = geglu_back(g, y.detach())
        torch.testing.assert_close(manual_grad, autograd_grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# weighted_geglu_back
# ---------------------------------------------------------------------------
class TestWeightedGegluBack:
    def test_returns_input_and_weights_grad(self, device):
        y = torch.randn(2, 4, 64, device=device)
        weights = torch.ones(2, 4, 1, device=device)
        g = torch.randn(2, 4, 32, device=device)

        input_grad, weights_grad = weighted_geglu_back(g, y, weights)
        assert input_grad.shape == y.shape
        assert weights_grad.shape == weights.shape

    def test_grad_matches_autograd(self, device):
        y = torch.randn(2, 4, 64, device=device, dtype=torch.float32, requires_grad=True)
        w = torch.randn(2, 4, 1, device=device, dtype=torch.float32, requires_grad=True)
        out = weighted_geglu(y, w)
        g = torch.randn_like(out)
        out.backward(g)

        manual_input_grad, manual_w_grad = weighted_geglu_back(g, y.detach(), w.detach())
        torch.testing.assert_close(manual_input_grad, y.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(manual_w_grad, w.grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# WeightedGEGLUFunction autograd custom op
# ---------------------------------------------------------------------------
class TestWeightedGEGLUFunction:
    def test_forward_matches_weighted_geglu(self, device):
        y = torch.randn(8, 64, device=device)
        w = torch.ones(8, 1, device=device)
        expected = weighted_geglu(y, w)
        out = WeightedGEGLUFunction.apply(y, w, False)
        torch.testing.assert_close(out, expected)

    def test_backward_produces_grads(self, device):
        y = torch.randn(8, 64, device=device, requires_grad=True)
        w = torch.randn(8, 1, device=device, requires_grad=True)
        out = WeightedGEGLUFunction.apply(y, w, False)
        loss = out.sum()
        loss.backward()
        assert y.grad is not None
        assert w.grad is not None
        assert y.grad.shape == y.shape
        assert w.grad.shape == w.shape

    def test_gradcheck(self, device):
        y = torch.randn(4, 16, device=device, dtype=torch.float64, requires_grad=True)
        w = torch.randn(4, 1, device=device, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda inp, wt: WeightedGEGLUFunction.apply(inp, wt, False),
            (y, w),
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# weighted_bias_geglu_impl (entry point)
# ---------------------------------------------------------------------------
class TestWeightedBiasGegluImpl:
    def test_2d_input(self, device):
        x = torch.randn(8, 64, device=device, requires_grad=True)
        w = torch.ones(8, 1, device=device, requires_grad=True)
        out = weighted_bias_geglu_impl(x, w)
        assert out.shape == (8, 32)
        out.sum().backward()
        assert x.grad is not None

    def test_3d_input_reshapes_correctly(self, device):
        x = torch.randn(2, 4, 64, device=device, requires_grad=True)
        w = torch.ones(8, 1, device=device, requires_grad=True)
        out = weighted_bias_geglu_impl(x, w)
        assert out.shape == (2, 4, 32)
        out.sum().backward()
        assert x.grad is not None

    def test_1d_input_no_reshape(self, device):
        x = torch.randn(64, device=device, requires_grad=True)
        w = torch.ones(1, 1, device=device, requires_grad=True)
        out = weighted_bias_geglu_impl(x, w)
        assert out.shape == (1, 32)


# ---------------------------------------------------------------------------
# is_gated_activation recognizes "geglu"
# ---------------------------------------------------------------------------
class TestIsGatedActivation:
    def test_geglu_is_gated(self):
        from nemo_automodel.components.moe.experts import is_gated_activation
        assert is_gated_activation("geglu") is True

    def test_swiglu_is_gated(self):
        from nemo_automodel.components.moe.experts import is_gated_activation
        assert is_gated_activation("swiglu") is True

    def test_relu2_is_not_gated(self):
        from nemo_automodel.components.moe.experts import is_gated_activation
        assert is_gated_activation("relu2") is False


# ---------------------------------------------------------------------------
# get_expert_activation_for_deepep selects correct function for "geglu"
# ---------------------------------------------------------------------------
class TestGetExpertActivation:
    def test_geglu_returns_weighted_bias_geglu_impl(self):
        from nemo_automodel.components.moe.config import MoEConfig
        from nemo_automodel.components.moe.experts import get_expert_activation_for_deepep

        cfg = MoEConfig(
            dim=64,
            inter_dim=128,
            moe_inter_dim=32,
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=True,
            expert_activation="geglu",
            softmax_before_topk=False,
        )
        fn = get_expert_activation_for_deepep(cfg)
        assert fn is weighted_bias_geglu_impl

    def test_invalid_activation_raises(self):
        from nemo_automodel.components.moe.config import MoEConfig
        from nemo_automodel.components.moe.experts import get_expert_activation_for_deepep

        cfg = MoEConfig(
            dim=64, inter_dim=128, moe_inter_dim=32,
            n_routed_experts=4, n_shared_experts=0, n_activated_experts=2,
            n_expert_groups=0, n_limited_groups=0,
            train_gate=True, gate_bias_update_factor=0.0,
            score_func="softmax", route_scale=1.0, aux_loss_coeff=0.0,
            norm_topk_prob=True, expert_activation="invalid_act",
            softmax_before_topk=False,
        )
        with pytest.raises(ValueError, match="Invalid expert activation"):
            get_expert_activation_for_deepep(cfg)
