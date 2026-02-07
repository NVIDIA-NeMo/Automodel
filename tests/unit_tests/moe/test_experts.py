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

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_CUDA = torch.cuda.is_available()
SKIP_TE_TESTS = not (HAVE_TE and HAVE_CUDA)

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    get_expert_activation,
    get_expert_activation_for_deepep,
    is_gated_activation,
    relu2,
    swiglu,
)
from nemo_automodel.components.moe.config import MoEConfig


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


class TestActivationFunctions:
    """Test activation functions used in MoE layers."""

    def test_swiglu_shape_preservation(self, device):
        """Test that swiglu preserves expected output shape."""
        batch_size, seq_len, dim = 4, 8, 64
        inter_dim = 128

        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)
        gate_and_up_proj = torch.randn(dim, inter_dim * 2, dtype=torch.bfloat16, device=device)
        down_proj = torch.randn(inter_dim, dim, dtype=torch.bfloat16, device=device)

        result = swiglu(x, gate_and_up_proj=gate_and_up_proj, down_proj=down_proj)

        assert result.shape == (batch_size, seq_len, dim)
        assert result.device == device

    def test_swiglu_with_bias(self, device):
        """Test swiglu with bias terms."""
        batch_size, seq_len, dim = 2, 4, 32
        inter_dim = 64

        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)
        gate_and_up_proj = torch.randn(dim, inter_dim * 2, dtype=torch.bfloat16, device=device)
        down_proj = torch.randn(inter_dim, dim, dtype=torch.bfloat16, device=device)
        gate_up_proj_bias = torch.randn(inter_dim * 2, dtype=torch.bfloat16, device=device)
        down_proj_bias = torch.randn(dim, dtype=torch.bfloat16, device=device)

        result = swiglu(
            x,
            gate_and_up_proj=gate_and_up_proj,
            down_proj=down_proj,
            gate_up_proj_bias=gate_up_proj_bias,
            down_proj_bias=down_proj_bias,
        )

        assert result.shape == (batch_size, seq_len, dim)

    def test_get_expert_activation_swiglu(self, moe_config):
        """Test getting swiglu activation function."""
        moe_config.expert_activation = "swiglu"
        activation_fn = get_expert_activation(moe_config)

        assert activation_fn == swiglu

    def test_get_expert_activation_quick_geglu(self, moe_config):
        """Test getting quick_geglu activation function."""
        moe_config.expert_activation = "quick_geglu"
        activation_fn = get_expert_activation(moe_config)

        # Should be a partial function
        assert callable(activation_fn)

    def test_get_expert_activation_invalid(self, moe_config):
        """Test error handling for invalid activation."""
        moe_config.expert_activation = "invalid"

        with pytest.raises(ValueError, match="Invalid expert activation"):
            get_expert_activation(moe_config)

    def test_get_expert_activation_for_deepep_swiglu(self, moe_config):
        """Test getting swiglu activation for DeepEP."""
        moe_config.expert_activation = "swiglu"

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            activation_fn = get_expert_activation_for_deepep(moe_config)
            assert activation_fn == mock_swiglu


class TestGroupedExpertsZeroActiveExperts:
    """Test GroupedExperts handling of zero active local experts.

    When using expert parallelism, it's possible for no tokens to be routed
    to the local experts on a particular rank. This test class verifies that
    the GroupedExperts module correctly handles this edge case by:
    1. Returning correct output shape (all zeros for the local contribution)
    2. Maintaining gradient flow through expert parameters
    """

    @pytest.fixture
    def initialized_experts(self, moe_config, device):
        """Create GroupedExperts with properly initialized weights."""
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
        return experts

    @pytest.fixture
    def initialized_experts_with_bias(self, moe_config, device):
        """Create GroupedExperts with bias and properly initialized weights."""
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights to avoid NaN issues
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.zero_()
        return experts

    def test_zero_active_experts_forward_shape(self, initialized_experts, moe_config, device):
        """Test forward pass returns correct shape when no tokens select any expert."""
        experts = initialized_experts

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)

        # Set indices to an expert ID that doesn't exist (out of range)
        # This simulates the case where all tokens select experts on other ranks
        # In EP scenario, experts_start_idx to experts_end_idx defines local experts
        # Setting indices outside this range means no local experts are selected
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,  # Non-existent expert
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device
        # Check that output doesn't contain NaN
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_zero_active_experts_backward_no_error(self, moe_config, device):
        """Test backward pass completes without error when no tokens select any expert.

        When combined with other model outputs (like residual connections), the backward
        pass should complete without errors even when no local experts are active.
        """
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        # Set indices to non-existent expert (simulates all tokens routed elsewhere)
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Verify forward pass produces correct output
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

        # Simulate real training: MoE output combined with other model components
        # (e.g., residual connection). This ensures backward can run without error.
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # Input should have gradients from the residual path
        assert x.grad is not None, "Input should have gradients from residual path"

    def test_zero_active_experts_with_bias_backward_no_error(self, moe_config, device):
        """Test backward pass completes without error with bias when no tokens select any expert.

        When combined with other model outputs (like residual connections), the backward
        pass should complete without errors even when no local experts are active.
        """
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights and biases
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.zero_()

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        # Set indices to non-existent expert
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Verify forward pass produces correct output
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

        # Simulate real training: MoE output combined with other model components
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # Input should have gradients from the residual path
        assert x.grad is not None, "Input should have gradients from residual path"

    def test_zero_active_experts_partial_token_mask(self, initialized_experts, moe_config, device):
        """Test zero active experts case with partial token mask (some masked tokens)."""
        experts = initialized_experts

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        # Mask half the tokens
        token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        token_mask[: num_tokens // 2] = True
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)

        # Non-existent expert indices
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        # Check that output doesn't contain NaN
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_zero_active_experts_quick_geglu_activation(self, moe_config, device):
        """Test zero active experts case with quick_geglu activation function."""
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        moe_config.expert_activation = "quick_geglu"
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Verify forward pass produces correct output
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

        # Simulate real training: MoE output combined with other model components
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # Input should have gradients from the residual path
        assert x.grad is not None, "Input should have gradients from residual path"

    def test_mixed_active_and_inactive_experts(self, initialized_experts, moe_config, device):
        """Test when some tokens select local experts and others don't."""
        experts = initialized_experts

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)

        # Half tokens go to valid experts, half to non-existent
        indices = torch.zeros((num_tokens, moe_config.n_activated_experts), dtype=torch.long, device=device)
        indices[: num_tokens // 2] = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens // 2, moe_config.n_activated_experts), device=device
        )
        indices[num_tokens // 2 :] = moe_config.n_routed_experts + 100  # Non-existent

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        # Check that output doesn't contain NaN
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_zero_active_experts_output_is_minimal(self, initialized_experts, moe_config, device):
        """Test that output contribution from zero-active-experts path is minimal.

        When no tokens select any expert, the dummy computation should contribute
        minimally to the output (the contribution is multiplied by weights which
        could be small, and uses zeros as input).
        """
        experts = initialized_experts

        num_tokens = 8
        # Use bfloat16 to match the initialized_experts dtype
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        # Use small weights to ensure minimal contribution
        weights = torch.full(
            (num_tokens, moe_config.n_activated_experts), 0.01, dtype=torch.bfloat16, device=device
        )

        # Non-existent expert indices
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # The output should be very small since we're using zeros as input
        # and multiplying by small weights
        assert output.abs().max() < 1.0, "Output magnitude should be small for zero active experts"

    def test_zero_active_experts_grad_norm_no_hang(self, moe_config, device):
        """Test that computing gradient norm doesn't hang when no tokens select any expert.

        This test verifies that torch.nn.utils.clip_grad_norm_ completes without hanging,
        which is important for distributed training where all ranks must participate in
        gradient synchronization.
        """
        # Use float32 dtype for gradient computation
        moe_config.dtype = torch.float32
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

        # Set indices to non-existent expert (simulates all tokens routed elsewhere)
        indices = torch.full(
            (num_tokens, moe_config.n_activated_experts),
            fill_value=moe_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        # Simulate real training: MoE output combined with residual connection
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        combined = output + residual
        loss = combined.sum()
        loss.backward()

        # This is the critical test: clip_grad_norm_ should complete without hanging
        # In distributed training, if gradients don't exist, this could cause a hang
        grad_norm = torch.nn.utils.clip_grad_norm_(experts.parameters(), max_norm=1.0)

        # Verify grad_norm is a valid finite number (not NaN or Inf)
        assert torch.isfinite(grad_norm), f"Gradient norm should be finite, got {grad_norm}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_zero_active_experts_has_expert_gradients(self, moe_config, device):
        """Test that expert parameters have gradients when no tokens select any expert.

        Note: This test runs in a subprocess to avoid caching issues
        when run alongside other tests. The test code is in run_zero_active_experts_gradient_test.py.
        """
        import subprocess
        import sys

        # Run test as a module to avoid path resolution issues with torch.compile caching
        result = subprocess.run(
            [sys.executable, "-m", "tests.unit_tests.moe.run_zero_active_experts_gradient_test", str(device)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Subprocess test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "SUCCESS" in result.stdout, (
            f"Test did not complete successfully:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


class TestGroupedExperts:
    """Test GroupedExperts module."""

    def test_grouped_experts_init(self, moe_config):
        """Test GroupedExperts initialization."""
        experts = GroupedExperts(moe_config)

        assert experts.n_routed_experts == moe_config.n_routed_experts
        assert experts.expert_bias == moe_config.expert_bias
        expected_shape = (moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim * 2)
        assert experts.gate_and_up_projs.shape == expected_shape

        down_shape = (moe_config.n_routed_experts, moe_config.moe_inter_dim, moe_config.dim)
        assert experts.down_projs.shape == down_shape

    def test_grouped_experts_init_with_bias(self, moe_config):
        """Test GroupedExperts initialization with bias."""
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)

        assert experts.gate_up_proj_bias is not None
        assert experts.down_proj_bias is not None
        assert experts.gate_up_proj_bias.shape == (moe_config.n_routed_experts, moe_config.moe_inter_dim * 2)
        assert experts.down_proj_bias.shape == (moe_config.n_routed_experts, moe_config.dim)

    def test_grouped_experts_forward_shape(self, moe_config, device):
        """Test GroupedExperts forward pass shape preservation."""
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_grouped_experts_gpu_execution(self, moe_config):
        """Test GroupedExperts execution on GPU."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)

        num_tokens = 8
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        try:
            output = experts(x, token_mask, weights, indices)
            assert output.shape == x.shape
            assert output.device == device
            # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"GPU execution failed: {e}")


class TestGroupedExpertsDeepEP:
    """Test GroupedExpertsDeepEP module."""

    def test_grouped_experts_deepep_init(self, moe_config):
        """Test GroupedExpertsDeepEP initialization."""
        experts = GroupedExpertsDeepEP(moe_config)

        assert experts.config == moe_config
        assert experts.expert_bias == moe_config.expert_bias
        expected_shape = (moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim * 2)
        assert experts.gate_and_up_projs.shape == expected_shape

    def test_grouped_experts_deepep_token_dispatcher_init(self, moe_config):
        """Test token dispatcher initialization."""
        experts = GroupedExpertsDeepEP(moe_config)

        # Mock device mesh with proper integer returns
        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()

        # Patch the MoEFlexTokenDispatcher to avoid the TPxEP assertion
        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            assert hasattr(experts, "token_dispatcher")
            assert experts.ep_size == 2
            assert experts.ep_rank == 0

    def test_grouped_experts_deepep_apply_bias_no_bias(self, moe_config):
        """Test _apply_bias method with no bias."""
        experts = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8)
        tokens_per_expert = torch.tensor([2, 2])

        result = experts._apply_bias(value, bias=None, tokens_per_expert=tokens_per_expert)

        torch.testing.assert_close(result, value)

    def test_grouped_experts_deepep_apply_bias_with_bias(self, moe_config):
        """Test _apply_bias method with bias."""
        experts = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8)
        bias = [torch.randn(8), torch.randn(8)]
        tokens_per_expert = torch.tensor([2, 2])

        result = experts._apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert)

        assert result.shape == value.shape
        assert result.dtype == value.dtype

    def test_grouped_experts_deepep_apply_bias_with_probs(self, moe_config):
        """Test _apply_bias method with permuted probabilities."""
        experts = GroupedExpertsDeepEP(moe_config)

        # The bias application works on flattened tokens (4 tokens total)
        # Split by tokens_per_expert: [2, 2] means first 2 tokens go to expert 0, next 2 to expert 1
        value = torch.randn(4, 8)  # 4 tokens, 8 features each
        bias = [torch.randn(8), torch.randn(8)]  # One bias per expert (8 features each)
        tokens_per_expert = torch.tensor([2, 2])  # 2 tokens per expert
        # Permuted probs need to match the shape after broadcasting with bias
        # Each expert gets 2 tokens, and bias has shape (8,), so probs should have shape (2, 8) total
        # But looking at the code, it seems like permuted_probs should be per-token, not per-feature
        permuted_probs = torch.randn(4, 8)  # 4 tokens, 8 features each to match bias shape

        result = experts._apply_bias(
            value, bias=bias, tokens_per_expert=tokens_per_expert, permuted_probs=permuted_probs
        )

        assert result.shape == value.shape

class TestNonGatedActivations:
    """Test non-gated activation support (ReLU²) for memory-efficient MoE.

    Non-gated activations like ReLU² only need up_projs with shape [n_experts, dim, inter_dim]
    instead of gate_and_up_projs with shape [n_experts, dim, 2*inter_dim], saving 50% memory.
    """

    @pytest.fixture
    def relu2_config(self):
        """Create MoEConfig with ReLU² activation (non-gated)."""
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
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def swiglu_config(self):
        """Create MoEConfig with SwiGLU activation (gated)."""
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
            dtype=torch.bfloat16,
        )

    def test_is_gated_activation_swiglu(self):
        """Test is_gated_activation returns True for swiglu."""
        assert is_gated_activation("swiglu") is True

    def test_is_gated_activation_quick_geglu(self):
        """Test is_gated_activation returns True for quick_geglu."""
        assert is_gated_activation("quick_geglu") is True

    def test_is_gated_activation_relu2(self):
        """Test is_gated_activation returns False for relu2."""
        assert is_gated_activation("relu2") is False

    def test_grouped_experts_relu2_uses_smaller_projections(self, relu2_config):
        """Test that GroupedExperts with ReLU² uses smaller gate_and_up_projs (inter_dim, not 2*inter_dim)."""
        experts = GroupedExperts(relu2_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, inter_dim] (not 2*inter_dim)
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            relu2_config.n_routed_experts,
            relu2_config.dim,
            relu2_config.moe_inter_dim,  # inter_dim, not 2*inter_dim
        )

        # Should have down_projs (same for both gated and non-gated)
        assert experts.down_projs is not None
        assert experts.down_projs.shape == (
            relu2_config.n_routed_experts,
            relu2_config.moe_inter_dim,
            relu2_config.dim,
        )

    def test_grouped_experts_swiglu_uses_gate_and_up_projs(self, swiglu_config):
        """Test that GroupedExperts with SwiGLU creates gate_and_up_projs with 2*inter_dim."""
        experts = GroupedExperts(swiglu_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, 2*inter_dim]
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            swiglu_config.n_routed_experts,
            swiglu_config.dim,
            swiglu_config.moe_inter_dim * 2,
        )

    def test_grouped_experts_relu2_with_bias(self, relu2_config):
        """Test GroupedExperts with ReLU² and bias uses smaller gate_up_proj_bias (inter_dim)."""
        relu2_config.expert_bias = True
        experts = GroupedExperts(relu2_config)

        # Should have gate_up_proj_bias with shape [n_experts, inter_dim] (not 2*inter_dim)
        assert experts.gate_up_proj_bias is not None
        assert experts.gate_up_proj_bias.shape == (
            relu2_config.n_routed_experts,
            relu2_config.moe_inter_dim,  # inter_dim, not 2*inter_dim
        )

        # Should have down_proj_bias
        assert experts.down_proj_bias is not None

    def test_grouped_experts_swiglu_with_bias(self, swiglu_config):
        """Test GroupedExperts with SwiGLU and bias uses gate_up_proj_bias with 2*inter_dim."""
        swiglu_config.expert_bias = True
        experts = GroupedExperts(swiglu_config)

        # Should have gate_up_proj_bias with shape [n_experts, 2*inter_dim]
        assert experts.gate_up_proj_bias is not None
        assert experts.gate_up_proj_bias.shape == (
            swiglu_config.n_routed_experts,
            swiglu_config.moe_inter_dim * 2,
        )

    def test_grouped_experts_relu2_forward(self, relu2_config, device):
        """Test GroupedExperts with ReLU² forward pass works correctly."""
        experts = GroupedExperts(relu2_config)
        experts = experts.to(device)

        # Initialize weights
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)

        num_tokens = 8
        x = torch.randn(num_tokens, relu2_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, relu2_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, relu2_config.n_routed_experts, (num_tokens, relu2_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device
        assert not torch.isnan(output).any(), "Output should not contain NaN values"

    def test_relu2_function_shape_preservation(self, device):
        """Test that relu2 function preserves expected output shape."""
        batch_size, seq_len, dim = 4, 8, 64
        inter_dim = 128

        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)
        gate_and_up_proj = torch.randn(dim, inter_dim, dtype=torch.bfloat16, device=device)
        down_proj = torch.randn(inter_dim, dim, dtype=torch.bfloat16, device=device)

        result = relu2(x, gate_and_up_proj=gate_and_up_proj, down_proj=down_proj)

        assert result.shape == (batch_size, seq_len, dim)
        assert result.device == device

    def test_relu2_function_with_bias(self, device):
        """Test relu2 with bias terms."""
        batch_size, seq_len, dim = 2, 4, 32
        inter_dim = 64

        x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16, device=device)
        gate_and_up_proj = torch.randn(dim, inter_dim, dtype=torch.bfloat16, device=device)
        down_proj = torch.randn(inter_dim, dim, dtype=torch.bfloat16, device=device)
        gate_up_proj_bias = torch.randn(inter_dim, dtype=torch.bfloat16, device=device)
        down_proj_bias = torch.randn(dim, dtype=torch.bfloat16, device=device)

        result = relu2(
            x,
            gate_and_up_proj=gate_and_up_proj,
            down_proj=down_proj,
            gate_up_proj_bias=gate_up_proj_bias,
            down_proj_bias=down_proj_bias,
        )

        assert result.shape == (batch_size, seq_len, dim)

    def test_relu2_memory_efficiency(self, relu2_config, swiglu_config):
        """Test that ReLU² uses ~50% less memory for up projection weights than SwiGLU."""
        relu2_experts = GroupedExperts(relu2_config)
        swiglu_experts = GroupedExperts(swiglu_config)

        # Calculate parameter sizes
        relu2_up_params = relu2_experts.gate_and_up_projs.numel()
        swiglu_up_params = swiglu_experts.gate_and_up_projs.numel()

        # ReLU² should have exactly half the up projection parameters
        assert relu2_up_params * 2 == swiglu_up_params

    def test_get_expert_activation_relu2(self, relu2_config):
        """Test getting relu2 activation function."""
        activation_fn = get_expert_activation(relu2_config)
        assert activation_fn == relu2

    def test_grouped_experts_deepep_relu2_uses_smaller_projections(self, relu2_config):
        """Test that GroupedExpertsDeepEP with ReLU² uses smaller gate_and_up_projs."""
        experts = GroupedExpertsDeepEP(relu2_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, inter_dim] (not 2*inter_dim)
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            relu2_config.n_routed_experts,
            relu2_config.dim,
            relu2_config.moe_inter_dim,  # inter_dim, not 2*inter_dim
        )

    def test_grouped_experts_deepep_swiglu_uses_gate_and_up_projs(self, swiglu_config):
        """Test that GroupedExpertsDeepEP with SwiGLU creates gate_and_up_projs with 2*inter_dim."""
        experts = GroupedExpertsDeepEP(swiglu_config)

        # Should have gate_and_up_projs with shape [n_experts, dim, 2*inter_dim]
        assert experts.gate_and_up_projs is not None
        assert experts.gate_and_up_projs.shape == (
            swiglu_config.n_routed_experts,
            swiglu_config.dim,
            swiglu_config.moe_inter_dim * 2,
        )

@pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
class TestGroupedExpertsTE:
    """Test GroupedExpertsTE module using Transformer Engine's GroupedLinear."""

    @pytest.fixture
    def te_moe_config(self):
        """Create MoE config for TE tests."""
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

    @pytest.fixture
    def te_moe_config_with_bias(self):
        """Create MoE config with bias for TE tests."""
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
            expert_bias=True,
            expert_activation="swiglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            dtype=torch.bfloat16,
        )

    def _materialize_weights(self, experts, device):
        """Materialize meta device weights to actual device."""
        from transformer_engine.pytorch import GroupedLinear

        config = experts.config
        gate_up_out_features = config.moe_inter_dim * 2 if experts.is_gated else config.moe_inter_dim
        # Re-create on actual device
        experts.gate_up_linear = GroupedLinear(
            num_gemms=experts.num_local_experts,
            in_features=config.dim,
            out_features=gate_up_out_features,
            bias=experts.expert_bias,
            params_dtype=config.dtype,
            device=device,
        )
        experts.down_linear = GroupedLinear(
            num_gemms=experts.num_local_experts,
            in_features=config.moe_inter_dim,
            out_features=config.dim,
            bias=experts.expert_bias,
            params_dtype=config.dtype,
            device=device,
        )

    def test_grouped_experts_te_init(self, te_moe_config):
        """Test GroupedExpertsTE initialization."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config)

        assert experts.config == te_moe_config
        assert experts.expert_bias == te_moe_config.expert_bias
        assert experts.num_local_experts == te_moe_config.n_routed_experts
        assert experts.dim == te_moe_config.dim
        assert experts.moe_inter_dim == te_moe_config.moe_inter_dim
        assert experts.token_dispatcher is None
        assert experts.ep_mesh is None
        assert experts.ep_rank == 0

    def test_grouped_experts_te_init_with_bias(self, te_moe_config_with_bias):
        """Test GroupedExpertsTE initialization with bias."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config_with_bias)

        assert experts.expert_bias is True
        assert experts.gate_up_linear.use_bias is True
        assert experts.down_linear.use_bias is True

    def test_grouped_experts_te_weight_properties(self, te_moe_config):
        """Test weight property getters return correct shapes."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Test gate_and_up_projs shape: [n_experts, dim, moe_inter_dim * 2]
        gate_up = experts.gate_and_up_projs
        expected_shape = (te_moe_config.n_routed_experts, te_moe_config.dim, te_moe_config.moe_inter_dim * 2)
        assert gate_up.shape == expected_shape

        # Test down_projs shape: [n_experts, moe_inter_dim, dim]
        down = experts.down_projs
        expected_shape = (te_moe_config.n_routed_experts, te_moe_config.moe_inter_dim, te_moe_config.dim)
        assert down.shape == expected_shape

    def test_grouped_experts_te_weight_setters(self, te_moe_config):
        """Test weight property setters."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Create new weights
        new_gate_up = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.dim,
            te_moe_config.moe_inter_dim * 2,
            dtype=te_moe_config.dtype,
            device=device,
        )
        new_down = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.moe_inter_dim,
            te_moe_config.dim,
            dtype=te_moe_config.dtype,
            device=device,
        )

        # Set weights
        experts.gate_and_up_projs = new_gate_up
        experts.down_projs = new_down

        # Verify weights were set (check internal flag)
        assert hasattr(experts, "_weights_loaded_from_checkpoint")
        assert experts._weights_loaded_from_checkpoint is True

        # Verify shapes match
        assert experts.gate_and_up_projs.shape == new_gate_up.shape
        assert experts.down_projs.shape == new_down.shape

    def test_grouped_experts_te_bias_properties(self, te_moe_config_with_bias):
        """Test bias property getters and setters."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        # Test gate_up_proj_bias shape: [n_experts, moe_inter_dim * 2]
        gate_up_bias = experts.gate_up_proj_bias
        expected_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.moe_inter_dim * 2)
        assert gate_up_bias.shape == expected_shape

        # Test down_proj_bias shape: [n_experts, dim]
        down_bias = experts.down_proj_bias
        expected_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.dim)
        assert down_bias.shape == expected_shape

    def test_grouped_experts_te_bias_setter(self, te_moe_config_with_bias):
        """Test bias property setters."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        # Create new biases
        new_gate_up_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim * 2,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        new_down_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.dim,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )

        # Set biases
        experts.gate_up_proj_bias = new_gate_up_bias
        experts.down_proj_bias = new_down_bias

        # Verify shapes match
        assert experts.gate_up_proj_bias.shape == new_gate_up_bias.shape
        assert experts.down_proj_bias.shape == new_down_bias.shape

    def test_grouped_experts_te_no_bias_returns_none(self, te_moe_config):
        """Test bias properties return None when expert_bias is False."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        assert experts.gate_up_proj_bias is None
        assert experts.down_proj_bias is None

    def test_grouped_experts_te_state_dict(self, te_moe_config):
        """Test state_dict returns correct keys and shapes."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        state = experts.state_dict()

        # Check keys
        assert "gate_and_up_projs" in state
        assert "down_projs" in state

        # Check shapes
        expected_gate_up_shape = (te_moe_config.n_routed_experts, te_moe_config.dim, te_moe_config.moe_inter_dim * 2)
        expected_down_shape = (te_moe_config.n_routed_experts, te_moe_config.moe_inter_dim, te_moe_config.dim)

        assert state["gate_and_up_projs"].shape == expected_gate_up_shape
        assert state["down_projs"].shape == expected_down_shape

        # No bias keys since expert_bias is False
        assert "gate_up_proj_bias" not in state
        assert "down_proj_bias" not in state

    def test_grouped_experts_te_state_dict_with_bias(self, te_moe_config_with_bias):
        """Test state_dict includes bias when enabled."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        state = experts.state_dict()

        # Check bias keys exist
        assert "gate_up_proj_bias" in state
        assert "down_proj_bias" in state

        # Check bias shapes
        expected_gate_up_bias_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.moe_inter_dim * 2)
        expected_down_bias_shape = (te_moe_config_with_bias.n_routed_experts, te_moe_config_with_bias.dim)

        assert state["gate_up_proj_bias"].shape == expected_gate_up_bias_shape
        assert state["down_proj_bias"].shape == expected_down_bias_shape

    def test_grouped_experts_te_state_dict_with_prefix(self, te_moe_config):
        """Test state_dict with prefix."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        prefix = "layer.experts."
        state = experts.state_dict(prefix=prefix)

        assert f"{prefix}gate_and_up_projs" in state
        assert f"{prefix}down_projs" in state

    def test_grouped_experts_te_load_state_dict(self, te_moe_config):
        """Test _load_from_state_dict loads weights correctly."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Create a state dict with known values
        gate_up_weights = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.dim,
            te_moe_config.moe_inter_dim * 2,
            dtype=te_moe_config.dtype,
            device=device,
        )
        down_weights = torch.randn(
            te_moe_config.n_routed_experts,
            te_moe_config.moe_inter_dim,
            te_moe_config.dim,
            dtype=te_moe_config.dtype,
            device=device,
        )

        state_dict = {
            "gate_and_up_projs": gate_up_weights.clone(),
            "down_projs": down_weights.clone(),
        }

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        experts._load_from_state_dict(
            state_dict, "", None, True, missing_keys, unexpected_keys, error_msgs
        )

        assert len(missing_keys) == 0
        assert len(error_msgs) == 0

        # Verify weights were loaded
        loaded_gate_up = experts.gate_and_up_projs
        loaded_down = experts.down_projs

        torch.testing.assert_close(loaded_gate_up, gate_up_weights, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(loaded_down, down_weights, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_load_state_dict_with_bias(self, te_moe_config_with_bias):
        """Test _load_from_state_dict loads biases correctly."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config_with_bias)
        self._materialize_weights(experts, device)

        # Create state dict with known values
        gate_up_weights = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.dim,
            te_moe_config_with_bias.moe_inter_dim * 2,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        down_weights = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim,
            te_moe_config_with_bias.dim,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        gate_up_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim * 2,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )
        down_bias = torch.randn(
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.dim,
            dtype=te_moe_config_with_bias.dtype,
            device=device,
        )

        state_dict = {
            "gate_and_up_projs": gate_up_weights.clone(),
            "down_projs": down_weights.clone(),
            "gate_up_proj_bias": gate_up_bias.clone(),
            "down_proj_bias": down_bias.clone(),
        }

        missing_keys = []
        experts._load_from_state_dict(state_dict, "", None, True, missing_keys, [], [])

        assert len(missing_keys) == 0

        # Verify biases were loaded
        torch.testing.assert_close(experts.gate_up_proj_bias, gate_up_bias, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(experts.down_proj_bias, down_bias, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_load_state_dict_missing_keys(self, te_moe_config):
        """Test _load_from_state_dict reports missing keys."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Empty state dict
        state_dict = {}
        missing_keys = []

        experts._load_from_state_dict(state_dict, "", None, True, missing_keys, [], [])

        assert "gate_and_up_projs" in missing_keys
        assert "down_projs" in missing_keys

    def test_grouped_experts_te_init_token_dispatcher(self, te_moe_config):
        """Test init_token_dispatcher initializes correctly."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config)

        # Mock device mesh
        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()
        mock_mesh.mesh_dim_names = ("ep",)

        # Patch MoEFlexTokenDispatcher
        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            assert experts.ep_mesh == mock_mesh
            assert experts.ep_size == 2
            assert experts.ep_rank == 0
            assert experts.num_local_experts == te_moe_config.n_routed_experts // 2
            assert experts.token_dispatcher is not None

    def test_grouped_experts_te_init_token_dispatcher_updates_linear_layers(self, te_moe_config):
        """Test init_token_dispatcher recreates linear layers with correct num_gemms."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        experts = GroupedExpertsTE(te_moe_config)

        # Initial num_gemms should be full expert count
        initial_gate_up_num_gemms = experts.gate_up_linear.num_gemms
        assert initial_gate_up_num_gemms == te_moe_config.n_routed_experts

        # Mock device mesh with ep_size=2
        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()
        mock_mesh.mesh_dim_names = ("ep",)

        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            # After init, num_gemms should be n_routed_experts / ep_size
            expected_local_experts = te_moe_config.n_routed_experts // 2
            assert experts.gate_up_linear.num_gemms == expected_local_experts
            assert experts.down_linear.num_gemms == expected_local_experts

    def test_grouped_experts_te_state_dict_roundtrip(self, te_moe_config):
        """Test state_dict -> load_state_dict roundtrip preserves weights."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # Create first instance and set specific weights
        experts1 = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts1, device)

        # Initialize weights with specific values
        with torch.no_grad():
            for i in range(experts1.gate_up_linear.num_gemms):
                getattr(experts1.gate_up_linear, f"weight{i}").normal_(0, 0.02)
                getattr(experts1.down_linear, f"weight{i}").normal_(0, 0.02)

        # Get state dict
        state = experts1.state_dict()

        # Create second instance and load state
        experts2 = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts2, device)

        missing_keys = []
        experts2._load_from_state_dict(state, "", None, True, missing_keys, [], [])

        assert len(missing_keys) == 0

        # Compare weights
        torch.testing.assert_close(
            experts1.gate_and_up_projs, experts2.gate_and_up_projs, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            experts1.down_projs, experts2.down_projs, rtol=1e-4, atol=1e-4
        )

    def test_grouped_experts_te_weight_setter_with_none(self, te_moe_config):
        """Test weight setters handle None gracefully."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Get original weights
        original_gate_up = experts.gate_and_up_projs.clone()

        # Setting None should be a no-op
        experts.gate_and_up_projs = None
        experts.down_projs = None

        # Weights should be unchanged
        torch.testing.assert_close(experts.gate_and_up_projs, original_gate_up, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_init_weights(self, te_moe_config):
        """Test init_weights method."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        experts = GroupedExpertsTE(te_moe_config)
        self._materialize_weights(experts, device)

        # Get weights before init
        old_gate_up = experts.gate_and_up_projs.clone()

        # Initialize weights
        experts.init_weights(device, init_std=0.02)

        # Weights should have changed
        new_gate_up = experts.gate_and_up_projs
        assert not torch.equal(old_gate_up, new_gate_up)

    def test_grouped_experts_te_relu2_init(self):
        """Test GroupedExpertsTE initialization with ReLU² (non-gated)."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        config = MoEConfig(
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
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)

        assert experts.is_gated is False
        # gate_up_linear out_features should be moe_inter_dim (not 2*moe_inter_dim)
        assert experts.gate_up_linear.out_features == config.moe_inter_dim

    def test_grouped_experts_te_relu2_weight_shapes(self):
        """Test GroupedExpertsTE weight shapes with ReLU² (non-gated)."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
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
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)
        self._materialize_weights(experts, device)

        # gate_and_up_projs: [n_experts, dim, moe_inter_dim] (not 2*moe_inter_dim)
        gate_up = experts.gate_and_up_projs
        assert gate_up.shape == (config.n_routed_experts, config.dim, config.moe_inter_dim)

        # down_projs: [n_experts, moe_inter_dim, dim] (same for gated and non-gated)
        down = experts.down_projs
        assert down.shape == (config.n_routed_experts, config.moe_inter_dim, config.dim)

    def test_grouped_experts_te_relu2_with_bias(self):
        """Test GroupedExpertsTE with ReLU² and bias uses smaller gate_up_proj_bias."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
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
            expert_bias=True,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)
        self._materialize_weights(experts, device)

        # gate_up_proj_bias: [n_experts, moe_inter_dim] (not 2*moe_inter_dim)
        gate_up_bias = experts.gate_up_proj_bias
        assert gate_up_bias.shape == (config.n_routed_experts, config.moe_inter_dim)

        # down_proj_bias: [n_experts, dim]
        down_bias = experts.down_proj_bias
        assert down_bias.shape == (config.n_routed_experts, config.dim)

    def test_grouped_experts_te_relu2_state_dict(self):
        """Test state_dict with ReLU² returns correct shapes."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
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
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)
        self._materialize_weights(experts, device)

        state = experts.state_dict()

        assert "gate_and_up_projs" in state
        assert "down_projs" in state

        # ReLU² uses moe_inter_dim, not 2*moe_inter_dim
        assert state["gate_and_up_projs"].shape == (config.n_routed_experts, config.dim, config.moe_inter_dim)
        assert state["down_projs"].shape == (config.n_routed_experts, config.moe_inter_dim, config.dim)

    def test_grouped_experts_te_relu2_load_state_dict_roundtrip(self):
        """Test state_dict -> load_state_dict roundtrip with ReLU²."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        config = MoEConfig(
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
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

        # Create first instance with known weights
        experts1 = GroupedExpertsTE(config)
        self._materialize_weights(experts1, device)
        with torch.no_grad():
            for i in range(experts1.gate_up_linear.num_gemms):
                getattr(experts1.gate_up_linear, f"weight{i}").normal_(0, 0.02)
                getattr(experts1.down_linear, f"weight{i}").normal_(0, 0.02)

        state = experts1.state_dict()

        # Load into second instance
        experts2 = GroupedExpertsTE(config)
        self._materialize_weights(experts2, device)
        missing_keys = []
        experts2._load_from_state_dict(state, "", None, True, missing_keys, [], [])

        assert len(missing_keys) == 0
        torch.testing.assert_close(experts1.gate_and_up_projs, experts2.gate_and_up_projs, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(experts1.down_projs, experts2.down_projs, rtol=1e-4, atol=1e-4)

    def test_grouped_experts_te_relu2_memory_efficiency(self):
        """Test that TE ReLU² uses ~50% less memory for up projection than SwiGLU."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        base_kwargs = dict(
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
            dtype=torch.bfloat16,
        )

        relu2_experts = GroupedExpertsTE(MoEConfig(**base_kwargs, expert_activation="relu2"))
        swiglu_experts = GroupedExpertsTE(MoEConfig(**base_kwargs, expert_activation="swiglu"))

        # ReLU² gate_up_linear out_features should be half of SwiGLU's
        assert relu2_experts.gate_up_linear.out_features * 2 == swiglu_experts.gate_up_linear.out_features

    def test_grouped_experts_te_relu2_init_token_dispatcher(self):
        """Test init_token_dispatcher with ReLU² creates correctly sized linear layers."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTE

        config = MoEConfig(
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
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )
        experts = GroupedExpertsTE(config)

        # Before init_token_dispatcher, out_features should be moe_inter_dim
        assert experts.gate_up_linear.out_features == config.moe_inter_dim

        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()
        mock_mesh.mesh_dim_names = ("ep",)

        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()
            experts.init_token_dispatcher(mock_mesh)

            # After init_token_dispatcher, out_features should still be moe_inter_dim (not 2*)
            assert experts.gate_up_linear.out_features == config.moe_inter_dim
            assert experts.gate_up_linear.num_gemms == config.n_routed_experts // 2
