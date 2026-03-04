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

"""Tests for torch._grouped_mm expert backend (ported from main branch).

Tests cover:
- BackendConfig.experts field
- _apply_bias helper function
- _torch_mm_experts_fwd helper function
- GroupedExpertsDeepEP init with torch_mm backend
"""

import pytest
import torch

from nemo_automodel.components.moe.layers import (
    GroupedExpertsDeepEP,
    MoEConfig,
    _apply_bias,
    _torch_mm_experts_fwd,
)
from nemo_automodel.components.models.common import BackendConfig


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=128,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.bfloat16,
    )


# ──────────────────────────────────────────────────────────────────────
# BackendConfig.experts field
# ──────────────────────────────────────────────────────────────────────


class TestBackendConfigExperts:
    """Test BackendConfig experts field."""

    def test_default_experts_is_gmm(self):
        config = BackendConfig()
        assert config.experts == "gmm"

    def test_set_experts_torch_mm(self):
        config = BackendConfig(experts="torch_mm")
        assert config.experts == "torch_mm"

    def test_set_experts_gmm_explicit(self):
        config = BackendConfig(experts="gmm")
        assert config.experts == "gmm"


# ──────────────────────────────────────────────────────────────────────
# _apply_bias helper
# ──────────────────────────────────────────────────────────────────────


class TestApplyBias:
    """Test the _apply_bias helper function."""

    def test_none_bias_returns_input(self):
        value = torch.randn(10, 32)
        tokens_per_expert = torch.tensor([5, 5])
        result = _apply_bias(value, None, tokens_per_expert)
        assert torch.equal(result, value)

    def test_bias_without_probs(self, device):
        n_experts = 2
        dim = 4
        tokens_per_expert = torch.tensor([3, 2], device=device)
        total_tokens = tokens_per_expert.sum().item()

        value = torch.ones(total_tokens, dim, device=device, dtype=torch.float32)
        bias = torch.ones(n_experts, dim, device=device, dtype=torch.float32) * 0.5

        result = _apply_bias(value, bias, tokens_per_expert)
        assert result.shape == value.shape
        torch.testing.assert_close(result, torch.full_like(result, 1.5))

    def test_bias_with_probs(self, device):
        n_experts = 2
        dim = 4
        tokens_per_expert = torch.tensor([2, 2], device=device)
        total_tokens = 4

        value = torch.zeros(total_tokens, dim, device=device, dtype=torch.float32)
        bias = torch.ones(n_experts, dim, device=device, dtype=torch.float32)
        probs = torch.full((total_tokens, 1), 0.3, device=device, dtype=torch.float32)

        result = _apply_bias(value, bias, tokens_per_expert, probs)
        assert result.shape == value.shape
        torch.testing.assert_close(result, torch.full_like(result, 0.3))


# ──────────────────────────────────────────────────────────────────────
# _torch_mm_experts_fwd helper
# ──────────────────────────────────────────────────────────────────────


class TestTorchMMExpertsFwd:
    """Test _torch_mm_experts_fwd helper function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_basic_forward(self, device):
        n_experts = 2
        dim = 32
        inter_dim = 64
        total_tokens = 6

        hidden = torch.randn(total_tokens, dim, dtype=torch.bfloat16, device=device)
        gate_up = torch.randn(n_experts, dim, inter_dim * 2, dtype=torch.bfloat16, device=device) * 0.02
        down = torch.randn(n_experts, inter_dim, dim, dtype=torch.bfloat16, device=device) * 0.02
        tpe = torch.tensor([3, 3], device=device)
        probs = torch.rand(total_tokens, 1, dtype=torch.float32, device=device)

        from nemo_automodel.components.moe.megatron.moe_utils import weighted_bias_swiglu_impl

        output = _torch_mm_experts_fwd(hidden, gate_up, down, tpe, probs, weighted_bias_swiglu_impl)

        assert output.shape == (total_tokens, dim)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_single_expert(self, device):
        """Test with all tokens routed to a single expert."""
        dim = 16
        inter_dim = 32
        total_tokens = 4

        hidden = torch.randn(total_tokens, dim, dtype=torch.bfloat16, device=device)
        gate_up = torch.randn(1, dim, inter_dim * 2, dtype=torch.bfloat16, device=device) * 0.02
        down = torch.randn(1, inter_dim, dim, dtype=torch.bfloat16, device=device) * 0.02
        tpe = torch.tensor([4], device=device)
        probs = torch.rand(total_tokens, 1, dtype=torch.float32, device=device)

        from nemo_automodel.components.moe.megatron.moe_utils import weighted_bias_swiglu_impl

        output = _torch_mm_experts_fwd(hidden, gate_up, down, tpe, probs, weighted_bias_swiglu_impl)

        assert output.shape == (total_tokens, dim)
        assert not torch.isnan(output).any()


# ──────────────────────────────────────────────────────────────────────
# GroupedExpertsDeepEP with torch_mm backend
# ──────────────────────────────────────────────────────────────────────


class TestGroupedExpertsDeepEPTorchMM:
    """Test GroupedExpertsDeepEP init with torch_mm backend."""

    def test_init_with_torch_mm(self, moe_config):
        backend = BackendConfig(experts="torch_mm")
        experts = GroupedExpertsDeepEP(moe_config, backend=backend)
        assert experts.use_torch_mm is True

    def test_init_without_backend(self, moe_config):
        experts = GroupedExpertsDeepEP(moe_config)
        assert experts.use_torch_mm is False

    def test_init_with_default_backend(self, moe_config):
        experts = GroupedExpertsDeepEP(moe_config, backend=BackendConfig())
        assert experts.use_torch_mm is False

    def test_init_with_gmm_backend(self, moe_config):
        backend = BackendConfig(experts="gmm")
        experts = GroupedExpertsDeepEP(moe_config, backend=backend)
        assert experts.use_torch_mm is False

    def test_init_with_bias(self, moe_config):
        moe_config.expert_bias = True
        backend = BackendConfig(experts="torch_mm")
        experts = GroupedExpertsDeepEP(moe_config, backend=backend)
        assert experts.use_torch_mm is True
        assert experts.gate_up_proj_bias is not None
        assert experts.down_proj_bias is not None
