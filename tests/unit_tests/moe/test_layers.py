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

import importlib.util
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    GroupedExpertsTE,
)
from nemo_automodel.components.moe.layers import (
    MLP,
    FakeBalancedGate,
    Gate,
    MoE,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_CUDA = torch.cuda.is_available()
SKIP_TE_TESTS = not (HAVE_TE and HAVE_CUDA)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0.01,
        score_func="softmax",
        route_scale=1.0,
        dim=128,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestMLP:
    """Test MLP layer."""

    def test_mlp_init(self, device):
        """Test MLP initialization."""
        dim, inter_dim = 64, 128
        mlp = MLP(dim, inter_dim, backend="torch")

        assert mlp.gate_proj.in_features == dim
        assert mlp.gate_proj.out_features == inter_dim
        assert mlp.down_proj.in_features == inter_dim
        assert mlp.down_proj.out_features == dim
        assert mlp.up_proj.in_features == dim
        assert mlp.up_proj.out_features == inter_dim

    def test_mlp_forward_shape(self, device):
        """Test MLP forward pass shape preservation."""
        dim, inter_dim = 64, 128
        mlp = MLP(dim, inter_dim, backend="torch")
        mlp = mlp.to(device)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)

        output = mlp(x)

        assert output.shape == (batch_size, seq_len, dim)
        assert output.device == device

    def test_mlp_forward_computation(self, device):
        """Test MLP forward computation correctness."""
        dim, inter_dim = 4, 8
        mlp = MLP(dim, inter_dim, backend="torch")
        mlp = mlp.to(device)

        x = torch.randn(1, 1, dim, dtype=torch.bfloat16, device=device)

        # Manual computation for verification
        gate_out = mlp.gate_proj(x)
        up_out = mlp.up_proj(x)
        expected = mlp.down_proj(F.silu(gate_out) * up_out)

        output = mlp(x)

        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    def test_mlp_init_weights(self, device):
        """Test MLP weight initialization."""
        mlp = MLP(64, 128, backend="torch")

        original_gate_weight = mlp.gate_proj.weight.clone().detach()

        with torch.no_grad():
            mlp.init_weights(device, init_std=0.02)

        # Weights should have changed
        assert not torch.equal(mlp.gate_proj.weight.detach(), original_gate_weight)


class TestFakeBalancedGate:
    """Test FakeBalancedGate for uniform expert routing."""

    def test_fake_balanced_gate_init(self, moe_config):
        """Test FakeBalancedGate initialization."""
        gate = FakeBalancedGate(moe_config)

        assert gate.n_routed_experts == moe_config.n_routed_experts
        assert gate.n_activated_experts == moe_config.n_activated_experts

    def test_fake_balanced_gate_forward_shape(self, moe_config, device):
        """Test FakeBalancedGate forward output shapes."""
        gate = FakeBalancedGate(moe_config)
        gate = gate.to(device)

        batch_size, seq_len = 4, 8
        x = torch.randn(batch_size * seq_len, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        expected_shape = (batch_size * seq_len, moe_config.n_activated_experts)
        assert weights.shape == expected_shape
        assert indices.shape == expected_shape
        assert aux_loss is None

    def test_fake_balanced_gate_uniform_weights(self, moe_config, device):
        """Test that FakeBalancedGate produces uniform weights."""
        gate = FakeBalancedGate(moe_config)
        gate = gate.to(device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # All weights should be 1/n_activated_experts
        expected_weight = 1.0 / moe_config.n_activated_experts
        torch.testing.assert_close(weights, torch.full_like(weights, expected_weight))

    def test_fake_balanced_gate_cycling_indices(self, moe_config, device):
        """Test that FakeBalancedGate cycles through experts."""
        gate = FakeBalancedGate(moe_config)
        gate = gate.to(device)

        num_tokens = moe_config.n_routed_experts * 2  # Two full cycles
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Check that we cycle through experts
        flat_indices = indices.flatten()
        for i in range(moe_config.n_routed_experts):
            assert i in flat_indices

    def test_routing_with_skip_first_expert(self, moe_config, device):
        """Test routing when skipping the first expert."""
        skip_n = 1
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        batch_size = 16
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Check indices skip the first expert (expert 0)
        assert indices.min() >= skip_n
        assert indices.max() < moe_config.n_routed_experts

        # Check that expert 0 is never selected
        assert (indices == 0).sum() == 0

    def test_routing_with_skip_multiple_experts(self, moe_config, device):
        """Test routing when skipping multiple experts."""
        skip_n = 3
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        batch_size = 32
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Check indices skip the first 3 experts (experts 0, 1, 2)
        assert indices.min() >= skip_n
        assert indices.max() < moe_config.n_routed_experts

        # Check that experts 0, 1, 2 are never selected
        for i in range(skip_n):
            assert (indices == i).sum() == 0

    def test_load_balancing_with_skip(self, moe_config, device):
        """Test that load is balanced across available experts when skipping."""
        skip_n = 2
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        # Use a large batch to ensure good distribution
        batch_size = 1000
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # Count how many times each expert is selected
        available_experts = moe_config.n_routed_experts - skip_n
        expert_counts = torch.zeros(moe_config.n_routed_experts, dtype=torch.int64, device=device)
        for i in range(moe_config.n_routed_experts):
            expert_counts[i] = (indices == i).sum()

        # First skip_n experts should have 0 assignments
        assert expert_counts[:skip_n].sum() == 0

        # Remaining experts should have roughly equal assignments
        remaining_counts = expert_counts[skip_n:]
        expected_count = (batch_size * moe_config.n_activated_experts) // available_experts

        # Allow some tolerance for distribution
        assert torch.all(remaining_counts > 0), "All available experts should be used"
        assert torch.all(
            torch.abs(remaining_counts - expected_count) < expected_count * 0.2
        ), "Load should be roughly balanced"

    def test_weights_are_uniform_with_skip(self, moe_config, device):
        """Test that weights are always uniform regardless of skip parameter."""
        for skip_n in [0, 1, 3]:
            gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
            gate = gate.to(device)

            batch_size = 8
            x = torch.randn(batch_size, moe_config.dim, device=device)
            token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            weights, _, _ = gate(x, token_mask, cp_mesh=None)

            # All weights should be 1 / n_activated_experts
            expected_weight = 1.0 / moe_config.n_activated_experts
            assert torch.allclose(weights, torch.ones_like(weights) * expected_weight)

    def test_skip_almost_all_experts(self, moe_config, device):
        """Test edge case where we skip all but one expert."""
        skip_n = moe_config.n_routed_experts - 1
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n)
        gate = gate.to(device)

        batch_size = 8
        x = torch.randn(batch_size, moe_config.dim, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        # All tokens should route to the last expert
        assert torch.all(indices == moe_config.n_routed_experts - 1)

    def test_output_dtype_matches_input(self, moe_config, device):
        """Test that output weights match input dtype."""
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=0)
        gate = gate.to(device)

        batch_size = 8

        # Test with float32
        x_fp32 = torch.randn(batch_size, moe_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        weights_fp32, _, _ = gate(x_fp32, token_mask, cp_mesh=None)
        assert weights_fp32.dtype == torch.float32

        # Test with float16
        x_fp16 = torch.randn(batch_size, moe_config.dim, dtype=torch.float16, device=device)
        weights_fp16, _, _ = gate(x_fp16, token_mask, cp_mesh=None)
        assert weights_fp16.dtype == torch.float16


class TestFakeBalancedGateNoise:
    """Test FakeBalancedGate with configurable noise."""

    def test_noise_zero_matches_balanced(self, moe_config, device):
        """Test that noise=0 gives identical results to the default balanced gate."""
        gate_default = FakeBalancedGate(moe_config)
        gate_zero = FakeBalancedGate(moe_config, noise=0.0)
        gate_default.to(device)
        gate_zero.to(device)

        x = torch.randn(16, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(16, dtype=torch.bool, device=device)

        w1, i1, _ = gate_default(x, mask, cp_mesh=None)
        w2, i2, _ = gate_zero(x, mask, cp_mesh=None)

        torch.testing.assert_close(w1, w2)
        assert torch.equal(i1, i2)

    def test_noise_one_random_indices(self, moe_config, device):
        """Test that noise=1.0 produces fully random indices (not perfectly balanced)."""
        gate = FakeBalancedGate(moe_config, noise=1.0)
        gate.to(device)

        # Large batch to ensure statistical detection of randomness
        n_tokens = 1000
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        weights, indices, _ = gate(x, mask, cp_mesh=None)

        # With noise=1.0, weights should NOT be uniform
        expected_uniform = 1.0 / moe_config.n_activated_experts
        assert not torch.allclose(weights, torch.full_like(weights, expected_uniform))

        # Weights should still sum to ~1 per token
        weight_sums = weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-3, rtol=1e-3)

    def test_noise_creates_load_imbalance(self, moe_config, device):
        """Test that noise creates measurable load imbalance across experts."""
        gate = FakeBalancedGate(moe_config, noise=0.5)
        gate.to(device)

        n_tokens = 2000
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        # Count tokens per expert
        counts = torch.zeros(moe_config.n_routed_experts, device=device)
        for i in range(moe_config.n_routed_experts):
            counts[i] = (indices == i).sum()

        # With noise, std dev should be nonzero (not perfectly balanced)
        assert counts.float().std() > 0

    def test_noise_output_shapes(self, moe_config, device):
        """Test that noisy gate outputs have correct shapes."""
        gate = FakeBalancedGate(moe_config, noise=0.3)
        gate.to(device)

        n_tokens = 32
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, mask, cp_mesh=None)

        assert weights.shape == (n_tokens, moe_config.n_activated_experts)
        assert indices.shape == (n_tokens, moe_config.n_activated_experts)
        assert aux_loss is None

    def test_noise_indices_in_valid_range(self, moe_config, device):
        """Test that noisy indices stay within valid expert range."""
        gate = FakeBalancedGate(moe_config, noise=1.0)
        gate.to(device)

        n_tokens = 200
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        assert indices.min() >= 0
        assert indices.max() < moe_config.n_routed_experts

    def test_noise_with_skip_experts(self, moe_config, device):
        """Test that noise respects skip_first_n_experts."""
        skip_n = 2
        gate = FakeBalancedGate(moe_config, skip_first_n_experts=skip_n, noise=0.8)
        gate.to(device)

        n_tokens = 200
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        # Skipped experts should never appear
        assert indices.min() >= skip_n
        assert indices.max() < moe_config.n_routed_experts

    def test_noise_indices_unique_per_token(self, moe_config, device):
        """Test that each token's expert indices are unique (no duplicates).

        Duplicate expert indices per token cause undefined behavior in the
        scatter-back step (y[token_ids] += ...) during the MoE forward pass.
        """
        gate = FakeBalancedGate(moe_config, noise=1.0)
        gate.to(device)

        n_tokens = 500
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        _, indices, _ = gate(x, mask, cp_mesh=None)

        # Each row should have unique expert indices
        for i in range(n_tokens):
            row = indices[i]
            assert row.unique().numel() == row.numel(), f"Token {i} has duplicate experts: {row}"

    def test_noise_weights_positive(self, moe_config, device):
        """Test that noisy weights are all positive."""
        gate = FakeBalancedGate(moe_config, noise=0.9)
        gate.to(device)

        n_tokens = 64
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        weights, _, _ = gate(x, mask, cp_mesh=None)

        assert (weights > 0).all()

    def test_noise_deterministic_for_same_input(self, moe_config, device):
        """Test that same input produces same routing (activation checkpointing safe).

        The seed is derived from the input content, so forward and recompute
        (which receive the same x) produce identical routing.
        """
        gate = FakeBalancedGate(moe_config, noise=0.7)
        gate.to(device)

        n_tokens = 64
        x = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        w1, i1, _ = gate(x, mask, cp_mesh=None)
        w2, i2, _ = gate(x, mask, cp_mesh=None)

        torch.testing.assert_close(w1, w2)
        assert torch.equal(i1, i2)

    def test_noise_dynamic_for_different_input(self, moe_config, device):
        """Test that different inputs produce different routing.

        This mimics real training where each step has different hidden states,
        reproducing the dynamic tokens_per_expert pattern of real Gate.
        """
        gate = FakeBalancedGate(moe_config, noise=0.7)
        gate.to(device)

        n_tokens = 64
        mask = torch.ones(n_tokens, dtype=torch.bool, device=device)

        x1 = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        x2 = torch.randn(n_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)

        _, i1, _ = gate(x1, mask, cp_mesh=None)
        _, i2, _ = gate(x2, mask, cp_mesh=None)

        assert not torch.equal(i1, i2)

    def test_noise_stored_as_attribute(self, moe_config):
        """Test that noise is stored as an attribute."""
        gate = FakeBalancedGate(moe_config, noise=0.42)
        assert gate.noise == 0.42

        gate_default = FakeBalancedGate(moe_config)
        assert gate_default.noise == 0.0


class TestGate:
    """Test Gate (router) module."""

    def test_gate_init_basic(self, moe_config):
        """Test Gate initialization with basic config."""
        gate = Gate(moe_config)

        assert gate.dim == moe_config.dim
        assert gate.n_experts == moe_config.n_routed_experts
        assert gate.topk == moe_config.n_activated_experts
        assert gate.weight.shape == (moe_config.n_routed_experts, moe_config.dim)
        assert gate.bias is None  # router_bias is False in fixture

    def test_gate_init_with_bias(self, moe_config):
        """Test Gate initialization with bias enabled."""
        moe_config.router_bias = True
        gate = Gate(moe_config)

        assert gate.bias is not None
        assert gate.bias.shape == (moe_config.n_routed_experts,)

    def test_gate_init_with_correction_bias(self, moe_config):
        """Test Gate initialization with bias update factor."""
        moe_config.gate_bias_update_factor = 0.1
        gate = Gate(moe_config)

        assert gate.e_score_correction_bias is not None
        assert gate.e_score_correction_bias.shape == (moe_config.n_routed_experts,)

    def test_gate_forward_softmax_mode(self, moe_config, device):
        """Test Gate forward pass in softmax mode."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config)
        gate = gate.to(device)

        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            if gate.bias is not None:
                gate.bias.zero_()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # In softmax mode, weights should sum to 1 along last dim
        # Use detach() to avoid gradient warnings
        weights_detached = weights.detach()
        expected = torch.ones(num_tokens, dtype=torch.bfloat16, device=device)
        torch.testing.assert_close(weights_detached.sum(dim=-1), expected, rtol=1e-4, atol=1e-4)

    def test_gate_forward_sigmoid_mode(self, moe_config, device):
        """Test Gate forward pass in sigmoid mode."""
        moe_config.score_func = "sigmoid"
        gate = Gate(moe_config)
        gate = gate.to(device)

        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            gate.weight.normal_(0, 0.02)
            if gate.bias is not None:
                gate.bias.zero_()

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        # In sigmoid mode, all weights should be between 0 and 1
        weights_detached = weights.detach()
        assert (weights_detached >= 0).all() and (weights_detached <= 1).all()

    def test_gate_forward_with_aux_loss(self, moe_config, device):
        """Test Gate forward pass with auxiliary loss computation."""
        moe_config.aux_loss_coeff = 0.01
        gate = Gate(moe_config)
        gate = gate.to(device)
        gate.train()  # Enable training mode for aux loss

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert aux_loss is not None
        assert aux_loss.numel() == 1  # Scalar loss
        assert aux_loss.requires_grad

    def test_gate_update_bias(self, moe_config, device):
        """Test gate bias update mechanism."""
        moe_config.gate_bias_update_factor = 0.1
        gate = Gate(moe_config)
        gate = gate.to(device)
        gate.train()

        # Simulate some expert load
        expert_load = torch.rand(moe_config.n_routed_experts, dtype=torch.bfloat16, device=device) * 10
        gate._cumulative_expert_load = expert_load

        original_bias = gate.e_score_correction_bias.clone()

        gate.update_bias()

        # Bias should have been updated
        assert not torch.equal(gate.e_score_correction_bias, original_bias)
        # Cumulative load should be reset
        assert gate._cumulative_expert_load is None

    def test_gate_init_weights(self, moe_config, device):
        """Test Gate weight initialization."""
        gate = Gate(moe_config)

        original_weight = gate.weight.clone().detach()

        with torch.no_grad():
            gate.init_weights(device, init_std=0.02)

        # Weight should have changed
        assert not torch.equal(gate.weight.detach(), original_weight)

    def test_gate_init_with_precision(self, moe_config):
        """Test Gate initialization with gate_precision set."""
        gate = Gate(moe_config, gate_precision=torch.float32)
        assert gate.gate_precision == torch.float32

        gate = Gate(moe_config, gate_precision=torch.float64)
        assert gate.gate_precision == torch.float64

    def test_gate_init_default_precision(self, moe_config):
        """Test Gate initialization with default precision (None)."""
        gate = Gate(moe_config, gate_precision=None)
        assert gate.gate_precision is None

        gate = Gate(moe_config)
        assert gate.gate_precision is None

    def test_gate_forward_with_fp32_precision(self, moe_config, device):
        """Test Gate forward pass with fp32 precision."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_forward_with_fp64_precision(self, moe_config, device):
        """Test Gate forward pass with fp64 precision."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config, gate_precision=torch.float64)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert indices.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_output_dtype_matches_input(self, moe_config, device):
        """Test that output dtype matches input dtype regardless of gate_precision."""
        moe_config.score_func = "softmax"

        for input_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for gate_precision in [None, torch.float32, torch.float64]:
                gate = Gate(moe_config, gate_precision=gate_precision)
                gate = gate.to(device)

                with torch.no_grad():
                    gate.weight.normal_(0, 0.02)

                num_tokens = 8
                x = torch.randn(num_tokens, moe_config.dim, dtype=input_dtype, device=device)
                token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

                weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

                assert weights.dtype == input_dtype, (
                    f"Expected output dtype {input_dtype} but got {weights.dtype} "
                    f"with gate_precision={gate_precision}"
                )

    def test_gate_precision_with_sigmoid(self, moe_config, device):
        """Test Gate precision with sigmoid score function."""
        moe_config.score_func = "sigmoid"
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16
        weights_detached = weights.detach()
        assert (weights_detached >= 0).all() and (weights_detached <= 1).all()

    def test_gate_precision_with_correction_bias(self, moe_config, device):
        """Test Gate precision with correction bias enabled."""
        moe_config.score_func = "sigmoid"
        moe_config.gate_bias_update_factor = 0.1
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_with_norm_topk_prob(self, moe_config, device):
        """Test Gate precision with norm_topk_prob enabled."""
        moe_config.score_func = "softmax"
        moe_config.norm_topk_prob = True
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_with_softmax_before_topk(self, moe_config, device):
        """Test Gate precision with softmax_before_topk enabled."""
        moe_config.score_func = "softmax"
        moe_config.softmax_before_topk = True
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, moe_config.n_activated_experts)
        assert weights.dtype == torch.bfloat16

    def test_gate_precision_consistency_across_calls(self, moe_config, device):
        """Test that Gate with precision produces consistent results across calls."""
        moe_config.score_func = "softmax"
        gate = Gate(moe_config, gate_precision=torch.float32)
        gate = gate.to(device)
        gate.eval()

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        with torch.no_grad():
            weights1, indices1, _ = gate(x, token_mask, cp_mesh=None)
            weights2, indices2, _ = gate(x, token_mask, cp_mesh=None)

        torch.testing.assert_close(weights1, weights2)
        torch.testing.assert_close(indices1, indices2)

    def test_dtype_string_input(self):
        """Test that dtype field accepts string input and converts to torch.dtype."""
        config = MoEConfig(
            n_routed_experts=8,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=128,
            inter_dim=256,
            moe_inter_dim=256,
            norm_topk_prob=False,
            dtype="torch.float16",
        )

        assert config.dtype == torch.float16

    def test_gate_forward_with_string_precision_via_backend(self, device):
        """Test Gate forward pass with string precision input via BackendConfig."""
        config = MoEConfig(
            n_routed_experts=8,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=128,
            inter_dim=256,
            moe_inter_dim=256,
            norm_topk_prob=False,
            dtype="bfloat16",
        )

        backend_config = BackendConfig(gate_precision="float32")
        assert backend_config.gate_precision == torch.float32

        gate = Gate(config, gate_precision=backend_config.gate_precision)
        gate = gate.to(device)

        with torch.no_grad():
            gate.weight.normal_(0, 0.02)

        num_tokens = 16
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)

        assert weights.shape == (num_tokens, config.n_activated_experts)
        assert weights.dtype == torch.bfloat16
        assert gate.gate_precision == torch.float32


class TestMoE:
    """Test MoE (Mixture of Experts) module."""

    def test_moe_init_with_fake_balanced_gate(self, moe_config, backend_config):
        """Test MoE initialization with fake balanced gate."""
        backend_config.fake_balanced_gate = True
        moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, FakeBalancedGate)
        assert isinstance(moe.experts, GroupedExperts)

    def test_moe_init_fake_gate_noise_passed_through(self, moe_config, backend_config):
        """Test that fake_gate_noise from BackendConfig is passed to FakeBalancedGate."""
        backend_config.fake_balanced_gate = True
        backend_config.fake_gate_noise = 0.5
        moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, FakeBalancedGate)
        assert moe.gate.noise == 0.5

    def test_moe_init_with_deepep_single_device(self, moe_config, backend_config):
        """DeepEP dispatcher enabled but world size == 1 should fall back to GroupedExperts."""
        backend_config.experts = "te"
        backend_config.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExperts)

    @pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
    def test_moe_init_with_deepep_multi_device(self, moe_config, backend_config):
        """DeepEP dispatcher enabled and world size > 1 should use GroupedExpertsTE."""
        backend_config.experts = "te"
        backend_config.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsTE)

    def test_moe_init_with_gmm_experts_with_deepep(self, moe_config, backend_config):
        """GMM experts with deepep dispatcher should use GroupedExpertsDeepEP."""
        backend_config.experts = "gmm"
        backend_config.dispatcher = "deepep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsDeepEP)

    def test_moe_init_with_hybridep_single_device(self, moe_config, backend_config):
        """HybridEP dispatcher enabled but world size == 1 should fall back to GroupedExperts."""
        backend_config.experts = "torch_mm"
        backend_config.dispatcher = "hybridep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExperts)

    def test_moe_init_with_hybridep_multi_device(self, moe_config, backend_config):
        """HybridEP dispatcher enabled and world size > 1 should use GroupedExpertsDeepEP."""
        backend_config.experts = "torch_mm"
        backend_config.dispatcher = "hybridep"
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
            moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsDeepEP)
        assert moe.experts.dispatcher_backend == "hybridep"
        assert moe.experts.dispatcher_num_sms == backend_config.dispatcher_num_sms

    def test_moe_init_with_shared_experts(self, moe_config, backend_config):
        """Test MoE initialization with shared experts."""
        moe_config.n_shared_experts = 2
        moe = MoE(moe_config, backend_config)

        assert moe.shared_experts is not None
        assert isinstance(moe.shared_experts, MLP)

    def test_moe_init_without_shared_experts(self, moe_config, backend_config):
        """Test MoE initialization without shared experts."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)

        assert moe.shared_experts is None

    def test_moe_forward_without_shared_experts(self, moe_config, backend_config, device):
        """Test MoE forward pass without shared experts."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)

        with patch.object(moe.gate, "forward") as mock_gate, patch.object(moe.experts, "forward") as mock_experts:
            # Mock gate outputs
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )

            # Mock expert outputs
            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            output = moe(x)

            assert output.shape == x.shape
            assert output.device == device

    def test_moe_forward_with_shared_experts(self, moe_config, backend_config, device):
        """Test MoE forward pass with shared experts."""
        moe_config.n_shared_experts = 2
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)

        with (
            patch.object(moe.gate, "forward") as mock_gate,
            patch.object(moe.experts, "forward") as mock_experts,
            patch.object(moe.shared_experts, "forward") as mock_shared,
        ):
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)
            mock_shared.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            # Patch at the module level to avoid CUDA stream issues on CPU
            with (
                patch("torch.cuda.Stream") as mock_stream_class,
                patch("torch.cuda.current_stream") as mock_current_stream,
                patch("torch.cuda.stream") as mock_stream_context,
            ):
                mock_stream = Mock()
                mock_stream.wait_stream = Mock()
                mock_stream_class.return_value = mock_stream
                mock_current_stream.return_value = Mock()

                # Create a context manager that just yields
                mock_context = Mock()
                mock_context.__enter__ = Mock(return_value=None)
                mock_context.__exit__ = Mock(return_value=None)
                mock_stream_context.return_value = mock_context

                output = moe(x)

                assert output.shape == x.shape
                assert output.device == device

    def test_moe_forward_with_padding_mask(self, moe_config, backend_config, device):
        """Test MoE forward pass with padding mask."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        padding_mask[:, -2:] = True  # Mask last 2 tokens

        with patch.object(moe.gate, "forward") as mock_gate, patch.object(moe.experts, "forward") as mock_experts:
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                None,
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            output = moe(x, padding_mask=padding_mask)

            assert output.shape == x.shape
            # Verify gate was called with correct token mask
            mock_gate.assert_called_once()
            gate_args = mock_gate.call_args[0]
            token_mask = gate_args[1]
            expected_mask = (~padding_mask).flatten()
            torch.testing.assert_close(token_mask.float(), expected_mask.float())

    def test_moe_forward_return_tuple_with_aux_loss(self, moe_config, backend_config, device):
        """Test MoE forward returns tuple when there's auxiliary loss."""
        moe_config.n_shared_experts = 0
        moe = MoE(moe_config, backend_config)
        moe = moe.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, moe_config.dim, device=device)

        with patch.object(moe.gate, "forward") as mock_gate, patch.object(moe.experts, "forward") as mock_experts:
            aux_loss = torch.tensor(0.01, device=device)
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(
                    0,
                    moe_config.n_routed_experts,
                    (batch_size * seq_len, moe_config.n_activated_experts),
                    device=device,
                ),
                aux_loss,
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            result = moe(x)

            # Should return the reshaped output since aux_loss handling is done in gate
            assert result.shape == x.shape
