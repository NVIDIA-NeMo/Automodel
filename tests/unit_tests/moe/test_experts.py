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

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _apply_bias,
    _permute_tokens_for_grouped_mm,
    _select_te_ops_activation,
    _torch_mm_experts_fwd,
    get_expert_activation_for_deepep,
    is_gated_activation,
    swiglu_clamped_deepep,
    swiglu_step_deepep,
)


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

    def test_get_expert_activation_for_deepep_swiglu(self, moe_config):
        """Test getting swiglu activation for DeepEP."""
        moe_config.expert_activation = "swiglu"

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            activation_fn = get_expert_activation_for_deepep(moe_config)
            assert activation_fn == mock_swiglu

    def test_get_expert_activation_for_deepep_swiglu_default_uses_fused(self, moe_config):
        """``swiglu_limit == 0`` (default) keeps the fast fused ``weighted_bias_swiglu_impl`` path."""
        moe_config.expert_activation = "swiglu"
        moe_config.swiglu_limit = 0.0

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            assert get_expert_activation_for_deepep(moe_config) is mock_swiglu

    def test_get_expert_activation_for_deepep_swiglu_with_limit_uses_clamped(self, moe_config):
        """``swiglu_limit > 0`` dispatches to the clamped FP32 variant for DSV4."""
        from functools import partial

        moe_config.expert_activation = "swiglu"
        moe_config.swiglu_limit = 7.0

        activation_fn = get_expert_activation_for_deepep(moe_config)

        # Should be a functools.partial wrapping swiglu_clamped_deepep with limit=7.0
        assert isinstance(activation_fn, partial)
        assert activation_fn.func is swiglu_clamped_deepep
        assert activation_fn.keywords == {"limit": 7.0}

    def test_get_expert_activation_for_deepep_swiglu_negative_limit_uses_fused(self, moe_config):
        """A non-positive ``swiglu_limit`` (e.g. 0.0 or negative) falls back to the fused path."""
        moe_config.expert_activation = "swiglu"
        moe_config.swiglu_limit = -1.0

        with patch("nemo_automodel.components.moe.experts.weighted_bias_swiglu_impl") as mock_swiglu:
            assert get_expert_activation_for_deepep(moe_config) is mock_swiglu

    def test_get_expert_activation_for_deepep_step_swiglu_uses_post_silu_clamp(self, moe_config):
        """Step's routed experts select their distinct post-SiLU clamp."""
        from functools import partial

        moe_config.expert_activation = "swiglu_step"
        moe_config.swiglu_limit = 7.0

        activation_fn = get_expert_activation_for_deepep(moe_config)

        assert isinstance(activation_fn, partial)
        assert activation_fn.func is swiglu_step_deepep
        assert activation_fn.keywords == {"limit": 7.0}


class TestSwigluClampedDeepEP:
    """Tests for the DSV4-style clamped FP32 SwiGLU activation."""

    def _eager_reference(self, x, permuted_probs, limit):
        """Reference implementation matching the DSV4 official Expert.forward."""
        gate, up = torch.chunk(x, 2, dim=-1)
        gate = gate.float().clamp(max=limit)
        up = up.float().clamp(min=-limit, max=limit)
        inter = torch.nn.functional.silu(gate) * up
        return (inter * permuted_probs).to(x.dtype)

    def test_output_shape_and_dtype(self):
        """Output should have shape ``[..., inter_dim]`` and the input's dtype."""
        torch.manual_seed(0)
        n_tokens = 4
        inter_dim = 8
        # x: [n_tokens, 2*inter_dim]
        x = torch.randn(n_tokens, 2 * inter_dim, dtype=torch.float32)
        probs = torch.rand(n_tokens, 1, dtype=torch.float32)

        out = swiglu_clamped_deepep(x, probs, limit=2.0)

        assert out.shape == (n_tokens, inter_dim)
        assert out.dtype == x.dtype

    @pytest.mark.parametrize("limit", [0.5, 2.0, 7.0])
    def test_matches_eager_reference_fp32(self, limit):
        """Compiled output must match an eager FP32 reference within tight tolerance."""
        torch.manual_seed(42)
        n_tokens = 8
        inter_dim = 16
        x = torch.randn(n_tokens, 2 * inter_dim, dtype=torch.float32) * 5.0  # exercise clamping
        probs = torch.rand(n_tokens, 1, dtype=torch.float32)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=limit)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        ref = self._eager_reference(x, probs, limit)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_clamping_caps_gate_above_limit(self):
        """When gate >> limit, silu(gate) saturates near gate≈limit (gate.clamp(max=limit))."""
        n_tokens = 2
        inter_dim = 4
        limit = 1.0
        # gate: very large positive => clamped at limit; up: modest (within range).
        gate = torch.full((n_tokens, inter_dim), 50.0, dtype=torch.float32)
        up = torch.full((n_tokens, inter_dim), 0.5, dtype=torch.float32)
        x = torch.cat([gate, up], dim=-1)
        probs = torch.ones(n_tokens, 1, dtype=torch.float32)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=limit)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        ref = self._eager_reference(x, probs, limit)
        # silu(1.0) * 0.5 ≈ 0.7311 * 0.5 ≈ 0.3655
        expected = torch.full_like(out, torch.nn.functional.silu(torch.tensor(limit)).item() * 0.5)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)

    def test_clamping_caps_up_outside_range(self):
        """``up`` is clamped symmetrically to [-limit, limit]."""
        n_tokens = 2
        inter_dim = 4
        limit = 1.0
        gate = torch.zeros((n_tokens, inter_dim), dtype=torch.float32)  # silu(0) = 0
        up = torch.full((n_tokens, inter_dim), -100.0, dtype=torch.float32)  # clamps to -1.0
        x = torch.cat([gate, up], dim=-1)
        probs = torch.ones(n_tokens, 1, dtype=torch.float32)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=limit)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        # silu(0) * (-1.0) = 0
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)

    def test_dtype_roundtrip_bf16(self):
        """Output dtype must equal input dtype (bf16 in, bf16 out)."""
        torch.manual_seed(0)
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        probs = torch.rand(2, 1, dtype=torch.bfloat16)

        try:
            out = swiglu_clamped_deepep(x, probs, limit=4.0)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"torch.compile path unavailable on this host: {exc}")

        assert out.dtype == torch.bfloat16
        assert out.shape == (2, 4)


class TestStepSwigluDeepEP:
    """Tests for Step3.5/3.7's post-SiLU clamped routed activation."""

    @staticmethod
    def _eager_reference(x, permuted_probs, limit):
        gate, up = torch.chunk(x, 2, dim=-1)
        gate = torch.nn.functional.silu(gate).clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return (gate * up * permuted_probs).to(x.dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_step_reference_and_differs_from_dsv4(self, dtype):
        limit = 1.5
        gate = torch.tensor([[4.0, -2.0]], dtype=dtype)
        up = torch.tensor([[3.0, -3.0]], dtype=dtype)
        x = torch.cat((gate, up), dim=-1)
        probs = torch.tensor([[0.75]], dtype=dtype)

        output = swiglu_step_deepep(x, probs, limit=limit)
        reference = self._eager_reference(x, probs, limit)
        dsv4_output = swiglu_clamped_deepep(x, probs, limit=limit)

        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        assert not torch.equal(output, dsv4_output)


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
        weights = torch.full((num_tokens, moe_config.n_activated_experts), 0.01, dtype=torch.bfloat16, device=device)

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
        assert result.returncode == 0, f"Subprocess test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
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


class TestGroupedExpertsForwardLoopDTensorBias:
    """Test that _forward_loop correctly handles DTensor biases.

    When expert parallelism shards bias parameters as DTensors, the
    _forward_loop path must convert them to local tensors before arithmetic
    with the plain-tensor matmul outputs.  A missing conversion causes:
        RuntimeError: aten.add.Tensor got mixed torch.Tensor and DTensor
    """

    @staticmethod
    def _init_experts(moe_config, device):
        moe_config.expert_bias = True
        experts = GroupedExperts(moe_config)
        experts = experts.to(device)
        with torch.no_grad():
            for p in experts.parameters():
                p.normal_(0, 0.02)
        return experts

    def test_forward_loop_with_bias_produces_correct_shape(self, moe_config, device):
        """Forward pass with expert_bias=True through _forward_loop should work."""
        experts = self._init_experts(moe_config, device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)
        assert output.shape == x.shape

    def test_forward_loop_bias_affects_output(self, moe_config, device):
        """Verify that biases actually influence the output (not silently ignored)."""
        experts = self._init_experts(moe_config, device)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        # Output with zero biases
        with torch.no_grad():
            experts.gate_up_proj_bias.zero_()
            experts.down_proj_bias.zero_()
        output_zero_bias = experts(x, token_mask, weights, indices)

        # Output with non-zero biases
        with torch.no_grad():
            experts.gate_up_proj_bias.fill_(1.0)
            experts.down_proj_bias.fill_(1.0)
        output_nonzero_bias = experts(x, token_mask, weights, indices)

        assert not torch.allclose(output_zero_bias, output_nonzero_bias), "Bias should change the output"

    def test_forward_loop_dtensor_bias_converted_to_local(self, moe_config, device, monkeypatch):
        """Verify that isinstance(bias, DTensor) triggers .to_local() in forward.

        We monkeypatch the isinstance check in experts.py so that the plain
        bias tensors are treated as DTensors.  A .to_local() method is attached
        to confirm the conversion path is exercised.
        """
        import builtins

        from torch.distributed.tensor import DTensor

        experts = self._init_experts(moe_config, device)

        to_local_calls = []
        original_isinstance = builtins.isinstance

        def patched_isinstance(obj, classinfo):
            """Make bias parameters appear as DTensor instances."""
            if original_isinstance(classinfo, type) and classinfo is DTensor:
                if hasattr(obj, "_fake_dtensor"):
                    return True
            if original_isinstance(classinfo, tuple) and DTensor in classinfo:
                if hasattr(obj, "_fake_dtensor"):
                    return True
            return original_isinstance(obj, classinfo)

        def fake_to_local(self_tensor):
            to_local_calls.append(self_tensor)
            return self_tensor.data

        # Mark biases as fake DTensors and add .to_local()
        experts.gate_up_proj_bias._fake_dtensor = True
        experts.gate_up_proj_bias.to_local = lambda: fake_to_local(experts.gate_up_proj_bias)
        experts.down_proj_bias._fake_dtensor = True
        experts.down_proj_bias.to_local = lambda: fake_to_local(experts.down_proj_bias)

        num_tokens = 16
        x = torch.randn(num_tokens, moe_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device
        )

        monkeypatch.setattr(builtins, "isinstance", patched_isinstance)
        try:
            output = experts(x, token_mask, weights, indices)
        finally:
            monkeypatch.undo()

        assert output.shape == x.shape
        assert len(to_local_calls) >= 2, f"Expected .to_local() called for both biases, got {len(to_local_calls)} calls"


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

        # Patch the MoEFlexTokenDispatcher to avoid the TPxEP assertion and the
        # DeepEP buffer allocation, which requires the optional runtime.
        with (
            patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher,
            patch.object(experts, "_init_deepep_buffer") as mock_init_buffer,
        ):
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            assert hasattr(experts, "token_dispatcher")
            assert experts.ep_size == 2
            assert experts.ep_rank == 0
            mock_init_buffer.assert_called_once_with(mock_mesh.get_group.return_value)

    def test_grouped_experts_deepep_apply_bias_no_bias(self, moe_config):
        """Test _apply_bias method with no bias."""
        _ = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8)
        tokens_per_expert = torch.tensor([2, 2])

        result = _apply_bias(value, bias=None, tokens_per_expert=tokens_per_expert)

        torch.testing.assert_close(result, value)

    def test_grouped_experts_deepep_apply_bias_with_bias(self, moe_config):
        """Test _apply_bias method with bias."""
        _ = GroupedExpertsDeepEP(moe_config)

        value = torch.randn(4, 8)
        bias = [torch.randn(8), torch.randn(8)]
        tokens_per_expert = torch.tensor([2, 2])

        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert)

        assert result.shape == value.shape
        assert result.dtype == value.dtype

    def test_grouped_experts_deepep_apply_bias_with_probs(self, moe_config):
        """Test _apply_bias method with permuted probabilities."""
        _ = GroupedExpertsDeepEP(moe_config)

        # The bias application works on flattened tokens (4 tokens total)
        # Split by tokens_per_expert: [2, 2] means first 2 tokens go to expert 0, next 2 to expert 1
        value = torch.randn(4, 8)  # 4 tokens, 8 features each
        bias = [torch.randn(8), torch.randn(8)]  # One bias per expert (8 features each)
        tokens_per_expert = torch.tensor([2, 2])  # 2 tokens per expert
        # Permuted probs need to match the shape after broadcasting with bias
        # Each expert gets 2 tokens, and bias has shape (8,), so probs should have shape (2, 8) total
        # But looking at the code, it seems like permuted_probs should be per-token, not per-feature
        permuted_probs = torch.randn(4, 8)  # 4 tokens, 8 features each to match bias shape

        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert, permuted_probs=permuted_probs)

        assert result.shape == value.shape

    def test_grouped_experts_deepep_init_with_hybridep_backend(self, moe_config):
        """Test GroupedExpertsDeepEP initialization with hybridep backend."""
        experts = GroupedExpertsDeepEP(
            moe_config,
            dispatcher_backend="hybridep",
            dispatcher_num_sms=24,
            dispatcher_num_sms_preprocessing=32,
            dispatcher_hybridep_enable_custom_allgather=False,
            dispatcher_share_token_dispatcher=False,
            dispatcher_async_dispatch=True,
        )

        assert experts.dispatcher_backend == "hybridep"
        assert experts.dispatcher_num_sms == 24
        assert experts.dispatcher_num_sms_preprocessing == 32
        assert experts.dispatcher_hybridep_enable_custom_allgather is False
        assert experts.dispatcher_share_token_dispatcher is False
        assert experts.dispatcher_async_dispatch is True
        assert experts.config == moe_config

    def test_grouped_experts_deepep_token_dispatcher_init_hybridep(self, moe_config):
        """Test init_token_dispatcher passes hybridep config to TokenDispatcherConfig."""
        experts = GroupedExpertsDeepEP(
            moe_config,
            dispatcher_backend="hybridep",
            dispatcher_num_sms=24,
            dispatcher_num_sms_preprocessing=32,
            dispatcher_hybridep_enable_custom_allgather=False,
            dispatcher_share_token_dispatcher=False,
            dispatcher_async_dispatch=True,
        )

        mock_mesh = Mock()
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = Mock()

        with patch("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher") as mock_dispatcher:
            mock_dispatcher.return_value = Mock()

            experts.init_token_dispatcher(mock_mesh)

            # Verify the TokenDispatcherConfig was created with hybridep settings
            call_kwargs = mock_dispatcher.call_args
            config_arg = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            if config_arg is None:
                config_arg = call_kwargs[0][2]  # positional arg
            assert config_arg.moe_flex_dispatcher_backend == "hybridep"
            assert config_arg.moe_hybridep_num_sms == 24
            assert config_arg.moe_hybridep_num_sms_preprocessing == 32
            assert config_arg.moe_hybridep_enable_custom_allgather is False
            assert config_arg.moe_deepep_num_sms == 24
            assert config_arg.moe_share_token_dispatcher is False
            assert config_arg.moe_deepep_async_dispatch is True


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

    def test_relu2_memory_efficiency(self, relu2_config, swiglu_config):
        """Test that ReLU² uses ~50% less memory for up projection weights than SwiGLU."""
        relu2_experts = GroupedExperts(relu2_config)
        swiglu_experts = GroupedExperts(swiglu_config)

        # Calculate parameter sizes
        relu2_up_params = relu2_experts.gate_and_up_projs.numel()
        swiglu_up_params = swiglu_experts.gate_and_up_projs.numel()

        # ReLU² should have exactly half the up projection parameters
        assert relu2_up_params * 2 == swiglu_up_params

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


@pytest.mark.parametrize("tail_shape", [(), (3,)])
def test_te_ops_glu_block_interleave_roundtrip(tail_shape):
    """TE's 32-wide GLU layout round-trips weights and biases without reordering values."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    canonical = torch.arange(2 * 128 * max(1, int(torch.tensor(tail_shape).prod())), dtype=torch.float32)
    canonical = canonical.reshape(2, 128, *tail_shape)

    interleaved = GroupedExpertsTeOps._interleave_glu_blocks(canonical)
    expected_prefix = torch.cat((canonical[:, :64:2], canonical[:, 1:64:2]), dim=1)

    torch.testing.assert_close(interleaved[:, :64], expected_prefix)
    torch.testing.assert_close(GroupedExpertsTeOps._deinterleave_glu_blocks(interleaved), canonical)


@pytest.mark.parametrize(
    ("activation", "expected_op", "route_scaled", "full_mxfp8_fusion"),
    [
        ("swiglu", "scaled_swiglu", True, True),
        ("swiglu_step", "scaled_swiglu", True, True),
        ("swigluoai", "scaled_clamped_qgeglu", True, True),
        ("quick_geglu", "scaled_clamped_qgeglu", True, True),
        ("geglu", "geglu", False, False),
        ("relu2", "scaled_srelu", True, True),
    ],
)
def test_te_ops_activation_selection(
    moe_config,
    activation,
    expected_op,
    route_scaled,
    full_mxfp8_fusion,
):
    """Every supported MoE activation selects an exact TE-ops pipeline."""
    moe_config.expert_activation = activation

    op_name, kwargs, selected_route_scaled, selected_full_mxfp8_fusion = _select_te_ops_activation(moe_config)

    assert op_name == expected_op
    assert selected_route_scaled is route_scaled
    assert selected_full_mxfp8_fusion is full_mxfp8_fusion
    assert "glu_interleave_size" not in kwargs


def test_te_ops_activation_selection_preserves_exact_unfused_variants(moe_config):
    """Numerically distinct clamped/unclamped variants stay available in BF16."""
    moe_config.expert_activation = "swiglu"
    moe_config.swiglu_limit = 10.0
    op_name, kwargs, route_scaled, full_mxfp8_fusion = _select_te_ops_activation(moe_config)
    assert (op_name, route_scaled, full_mxfp8_fusion) == ("exact_gated", False, False)
    assert kwargs == {"alpha": 1.0, "linear_offset": 0.0, "limit": 10.0}

    moe_config.expert_activation = "swiglu_step"
    moe_config.swiglu_limit = 10.0
    op_name, kwargs, route_scaled, full_mxfp8_fusion = _select_te_ops_activation(moe_config)
    assert (op_name, route_scaled, full_mxfp8_fusion) == ("exact_gated", False, False)
    assert kwargs == {
        "alpha": 1.0,
        "linear_offset": 0.0,
        "limit": 10.0,
        "clamp_after_gate_activation": True,
        "use_input_dtype": True,
    }

    moe_config.expert_activation = "swigluoai"
    moe_config.activation_limit = 0.0
    op_name, kwargs, route_scaled, full_mxfp8_fusion = _select_te_ops_activation(moe_config)
    assert (op_name, route_scaled, full_mxfp8_fusion) == ("exact_gated", False, False)
    assert kwargs == {"alpha": 1.702, "linear_offset": 1.0, "limit": None}

    moe_config.expert_activation = "quick_geglu"
    moe_config.moe_inter_dim = 30
    op_name, kwargs, route_scaled, full_mxfp8_fusion = _select_te_ops_activation(moe_config)
    assert (op_name, route_scaled, full_mxfp8_fusion) == ("scaled_clamped_qgeglu", True, False)
    assert "glu_interleave_size" not in kwargs


@pytest.mark.parametrize("activation", ["swiglu", "swiglu_step", "swigluoai", "quick_geglu"])
def test_te_ops_fused_glu_selection_uses_block32_only_when_effective(moe_config, activation):
    """Unfused GLUs consume concat layout; effective CuTe GLUs consume blocks."""
    moe_config.expert_activation = activation
    _, unfused_kwargs, _, unfused_eligible = _select_te_ops_activation(
        moe_config,
        full_mxfp8_fusion_requested=False,
    )
    _, fused_kwargs, _, fused_eligible = _select_te_ops_activation(
        moe_config,
        full_mxfp8_fusion_requested=True,
    )

    assert unfused_eligible and fused_eligible
    assert "glu_interleave_size" not in unfused_kwargs
    assert fused_kwargs["glu_interleave_size"] == 32


@pytest.mark.parametrize("activation", ["swiglu", "swiglu_step", "swigluoai", "quick_geglu", "relu2"])
def test_te_ops_full_fusion_rejects_unsupported_grouped_mlp_dims(moe_config, activation):
    """Odd expert dimensions fall back before TE's fused-dimension validator."""
    moe_config.expert_activation = activation
    moe_config.dim = 96
    _, kwargs, _, full_mxfp8_fusion = _select_te_ops_activation(
        moe_config,
        full_mxfp8_fusion_requested=True,
    )

    assert not full_mxfp8_fusion
    assert "glu_interleave_size" not in kwargs


@pytest.mark.parametrize(
    ("activation", "fusion_eligible"),
    [
        ("swiglu", True),
        ("swiglu_step", True),
        ("swigluoai", True),
        ("quick_geglu", True),
        ("geglu", False),
        ("relu2", True),
    ],
)
def test_te_ops_padding_and_fusion_expectation_are_activation_aware(moe_config, activation, fusion_eligible):
    """Only activation patterns supported by the full CuTe MLP request padding."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    moe_config.expert_activation = activation
    _, _, _, selected_fusion_eligible = _select_te_ops_activation(
        moe_config,
        full_mxfp8_fusion_requested=True,
    )
    assert selected_fusion_eligible is fusion_eligible

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    experts._te_ops_mxfp8_fusion_requested = True
    experts._te_ops_full_mxfp8_fusion_eligible = selected_fusion_eligible
    experts._te_ops_uses_padded_capacity = selected_fusion_eligible
    experts._te_ops_fusion_checked = False
    assert experts._router_expert_pad_multiple() == (256 if fusion_eligible else None)
    if not fusion_eligible:
        experts._check_te_ops_fusion(fp8_active=True)


def test_te_ops_bf16_graph_paged_capacity_does_not_enable_dispatcher_padding(monkeypatch):
    """TE's graph-safe BF16 GroupedTensor path may overallocate only the graph bucket."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    checker = Mock(return_value=True)
    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    experts._te_ops_fp8_configured = False
    experts._te_ops_configured_mxfp8 = False
    experts._te_ops_graph_uses_paged_capacity = False
    experts._te_ops_uses_padded_capacity = False
    experts.config = Mock(dtype=torch.bfloat16)
    experts.gate_up_linear = Mock(num_groups=4, _is_graph_safe_path_supported=checker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))

    supported, reason = experts._te_ops_dynamic_splits_graph_capability()

    assert supported
    assert reason == ""
    assert experts._te_ops_graph_uses_paged_capacity
    assert experts._router_expert_pad_multiple() is None
    checker.assert_called_once_with(
        with_quantized_compute=False,
        input_quantizers=[None] * 4,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(("capability", "supported"), [((10, 0), True), ((11, 0), True), ((12, 0), False)])
def test_te_ops_mxfp8_graph_paged_capacity_is_independent_of_full_fusion(monkeypatch, capability, supported):
    """Unfused MXFP8 GroupedLinear can page graph input without enabling dispatcher padding."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    experts._te_ops_fp8_configured = True
    experts._te_ops_configured_mxfp8 = True
    experts._te_ops_graph_uses_paged_capacity = False
    experts._te_ops_uses_padded_capacity = False
    experts.gate_up_linear = Mock(_is_graph_safe_path_supported=Mock())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    graph_safe, reason = experts._te_ops_dynamic_splits_graph_capability()

    assert graph_safe is supported
    assert experts._te_ops_graph_uses_paged_capacity is supported
    assert experts._router_expert_pad_multiple() is None
    assert (reason == "") is supported


@pytest.mark.parametrize("full_mxfp8_fusion_requested", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "swiglu_step", "swigluoai", "quick_geglu", "geglu", "relu2"])
def test_te_ops_canonical_layout_roundtrip_by_activation(moe_config, activation, full_mxfp8_fusion_requested):
    """Canonical and physical layouts round-trip in fused and unfused modes."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    moe_config.expert_activation = activation
    _, kwargs, _, full_mxfp8_fusion = _select_te_ops_activation(
        moe_config,
        full_mxfp8_fusion_requested=full_mxfp8_fusion_requested,
    )
    block_size = kwargs.get("glu_interleave_size")
    canonical = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128)
    physical = GroupedExpertsTeOps._to_te_gate_up_layout(canonical, activation, block_size)
    restored = GroupedExpertsTeOps._from_te_gate_up_layout(physical, activation, block_size)

    expect_layout_conversion = activation == "quick_geglu" or (
        full_mxfp8_fusion_requested and full_mxfp8_fusion and is_gated_activation(activation)
    )
    if expect_layout_conversion:
        assert not torch.equal(physical, canonical)
    else:
        assert physical.data_ptr() == canonical.data_ptr()
    if activation == "quick_geglu" and not full_mxfp8_fusion_requested:
        expected_physical = GroupedExpertsTeOps._pair_interleaved_to_concatenated(canonical)
        torch.testing.assert_close(physical, expected_physical)
    torch.testing.assert_close(restored, canonical)


@pytest.mark.parametrize("tail_shape", [(), (3,)])
def test_te_ops_concatenated_glu_block_interleave_roundtrip(tail_shape):
    """Canonical ``[gate | up]`` rows round-trip through TE's fused layout."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    canonical = torch.arange(2 * 128 * max(1, int(torch.tensor(tail_shape).prod())), dtype=torch.float32)
    canonical = canonical.reshape(2, 128, *tail_shape)
    physical = GroupedExpertsTeOps._interleave_concatenated_glu_blocks(canonical)
    expected_prefix = torch.cat((canonical[:, :32], canonical[:, 64:96]), dim=1)

    torch.testing.assert_close(physical[:, :64], expected_prefix)
    torch.testing.assert_close(GroupedExpertsTeOps._deinterleave_concatenated_glu_blocks(physical), canonical)


def test_te_ops_init_weights_preserves_parameter_objects_and_honors_std():
    """TE-ops initialization must preserve owners and use the canonical normal std."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_groups = 2
            self.in_features = 8
            self.use_bias = True
            self._stacked_weight = torch.nn.Parameter(torch.zeros(2, 8, 8))
            self._stacked_bias = torch.nn.Parameter(torch.ones(2, 8))

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts.gate_up_linear = FakeGroupedLinear()
    experts.down_linear = FakeGroupedLinear()
    parameter_ids = {name: id(param) for name, param in experts.named_parameters()}

    experts.init_weights(torch.device("cpu"), init_std=0.0)

    assert {name: id(param) for name, param in experts.named_parameters()} == parameter_ids
    for param in experts.parameters():
        torch.testing.assert_close(param, torch.zeros_like(param))


def test_te_ops_fuser_owner_identity_guard_rejects_stale_parameters():
    """The lazy TE fuser must hold the same plain Parameters as its linears."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    class FakeGroupedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._stacked_weight = torch.nn.Parameter(torch.randn(2, 4, 4))
            self._stacked_bias = torch.nn.Parameter(torch.randn(2, 4))

    class FakeFuser:
        def __init__(self, linears):
            self._basic_ops = list(linears)
            self._basic_op_params = [list(linear.parameters()) for linear in linears]

    class FakeSequential:
        def __init__(self, fuser):
            self._module_groups = [fuser]

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts.gate_up_linear = FakeGroupedLinear()
    experts.down_linear = FakeGroupedLinear()
    fuser = FakeFuser((experts.gate_up_linear, experts.down_linear))
    experts.__dict__["_te_grouped_mlp"] = FakeSequential(fuser)
    experts._te_ops_fuser_owner_signature = None

    experts._assert_te_fuser_owner_identity()

    experts.gate_up_linear._stacked_weight = torch.nn.Parameter(experts.gate_up_linear._stacked_weight.detach().clone())
    with pytest.raises(RuntimeError, match="captured stale expert parameters"):
        experts._assert_te_fuser_owner_identity()


def test_te_ops_empty_rank_preserves_dispatch_backward_and_explicit_zero_grads():
    """An empty destination rank must retain EP autograd edges and real expert parameters."""
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    class FakeExpertLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

    class EmptyDispatch(torch.autograd.Function):
        backward_called = False

        @staticmethod
        def forward(ctx, hidden_states, token_probs):
            ctx.hidden_shape = hidden_states.shape
            ctx.prob_shape = token_probs.shape
            return hidden_states[:0], token_probs.reshape(-1)[:0]

        @staticmethod
        def backward(ctx, grad_hidden_states, grad_token_probs):
            EmptyDispatch.backward_called = True
            assert grad_hidden_states.shape[0] == 0
            assert grad_token_probs.shape[0] == 0
            return (
                grad_hidden_states.new_zeros(ctx.hidden_shape),
                grad_token_probs.new_zeros(ctx.prob_shape),
            )

    class FakeTokenDispatcher:
        def __init__(self):
            self.unpermutation_input = None

        def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
            del num_local_tokens, token_indices
            dispatched_hidden, dispatched_probs = EmptyDispatch.apply(hidden_states, token_probs)
            return dispatched_hidden, torch.zeros(2, dtype=torch.int64), dispatched_probs

        def token_unpermutation(self, hidden_states):
            self.unpermutation_input = hidden_states
            return hidden_states

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts.config = Mock(n_routed_experts=2)
    experts.ep_size = 1
    experts.expert_bias = True
    experts.gate_up_linear = FakeExpertLinear()
    experts.down_linear = FakeExpertLinear()
    experts.__dict__["_te_grouped_mlp"] = Mock(side_effect=AssertionError("empty ranks must bypass TE"))
    experts.token_dispatcher = FakeTokenDispatcher()

    parameter_ids = {name: id(parameter) for name, parameter in experts.named_parameters()}
    optimizer = torch.optim.AdamW(experts.parameters(), lr=0.1, weight_decay=0.2)
    parameters_before_step = {name: parameter.detach().clone() for name, parameter in experts.named_parameters()}
    hidden_states = torch.randn(3, 4, requires_grad=True)
    token_probs = torch.randn(3, 2, requires_grad=True)
    token_indices = torch.zeros(3, 2, dtype=torch.int64)

    output = experts(hidden_states, torch.ones(3, dtype=torch.bool), token_probs, token_indices)
    assert output.shape == (0, 4)
    assert experts.token_dispatcher.unpermutation_input is not None
    experts._te_grouped_mlp.assert_not_called()
    output.sum().backward()

    assert EmptyDispatch.backward_called
    assert hidden_states.grad is not None
    assert token_probs.grad is not None
    assert torch.count_nonzero(hidden_states.grad) == 0
    assert torch.count_nonzero(token_probs.grad) == 0
    assert {name: id(parameter) for name, parameter in experts.named_parameters()} == parameter_ids
    for parameter in experts.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0

    optimizer.step()
    for name, parameter in experts.named_parameters():
        torch.testing.assert_close(parameter, parameters_before_step[name] * (1 - 0.1 * 0.2))


@pytest.mark.parametrize(
    ("partial_expert_graphs", "expected_mode"),
    [(False, "gemm_ready_fixed"), (True, "fixed_address")],
)
def test_te_ops_mxfp8_weight_cache_mode_follows_partial_expert_graphs(
    monkeypatch,
    partial_expert_graphs,
    expected_mode,
):
    from types import SimpleNamespace

    from nemo_automodel.components.moe.experts import GroupedExpertsTE, GroupedExpertsTeOps

    monkeypatch.setattr(GroupedExpertsTE, "__init__", lambda self, *args, **kwargs: None)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=SimpleNamespace(recipe="mxfp8"),
        te_ops_mxfp8_weight_cache=True,
        partial_cuda_graph_experts=partial_expert_graphs,
    )

    experts = GroupedExpertsTeOps(Mock(), backend=backend)

    assert experts._te_ops_mxfp8_weight_cache_requested is True
    assert experts._te_ops_mxfp8_weight_cache_mode == expected_mode
    assert experts._te_ops_mxfp8_weight_cache_enabled is False


def test_te_ops_reads_static_rank_budget_from_backend_without_requiring_backend_schema(monkeypatch):
    """The provisional HybridEP budget is threaded from the TE-ops backend by attribute."""
    from types import SimpleNamespace

    from nemo_automodel.components.moe.experts import GroupedExpertsTE, GroupedExpertsTeOps

    monkeypatch.setattr(GroupedExpertsTE, "__init__", lambda self, *args, **kwargs: None)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=None,
        moe_expert_rank_capacity_factor=1.25,
    )

    experts = GroupedExpertsTeOps(Mock(), backend=backend)

    assert experts.moe_expert_rank_capacity_factor == 1.25


def test_te_ops_mxfp8_internal_group_quantize_ab_mode_survives_graph_selection():
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps, _TeOpsMXFP8WeightCache

    class FakeLinear:
        def __init__(self):
            self.calls = []

        def set_mxfp8_weight_cache_enabled(self, enabled, *, fallback_reason, mode):
            self.calls.append((enabled, fallback_reason, mode))

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    experts._te_ops_mxfp8_weight_cache_graph_off_mode = _TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE
    experts._te_ops_mxfp8_weight_cache_mode = _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts._te_ops_mxfp8_weight_cache_fallback_reason = ""
    experts.gate_up_linear = FakeLinear()
    experts.down_linear = FakeLinear()

    experts.configure_mxfp8_weight_cache_for_partial_graph(captured=False)
    assert experts._te_ops_mxfp8_weight_cache_mode == _TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE
    experts.configure_mxfp8_weight_cache_for_partial_graph(captured=True)
    assert experts._te_ops_mxfp8_weight_cache_mode == _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.calls == [
            (True, "", _TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE),
            (True, "", _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE),
        ]


def _install_te_216_mxfp8_cache_stubs(monkeypatch):
    """Install only the TE 2.16 APIs used by the persistent compute cache."""
    import sys
    import types

    class FakeMXFP8Member:
        def __init__(self, rowwise_data, columnwise_data):
            self.rowwise_data = rowwise_data
            self.columnwise_data = columnwise_data
            self.value = None

    class FakeGroupedTensor:
        def __init__(self, num_tensors, *, optimize_for_gemm=False):
            self.num_tensors = num_tensors
            data_stride = 32 * 32
            self.rowwise_data = torch.zeros(num_tensors * data_stride, dtype=torch.uint8)
            self.columnwise_data = torch.zeros(num_tensors * data_stride, dtype=torch.uint8)
            # TE pads both scale buffers for a 32x32 MXFP8 member to 128x4
            # elements; optimize_for_gemm changes their layout, not capacity.
            scale_stride = 512
            columnwise_scale_stride = 512
            self.scale_inv = torch.zeros(num_tensors * scale_stride, dtype=torch.uint8)
            self.columnwise_scale_inv = torch.zeros(num_tensors * columnwise_scale_stride, dtype=torch.uint8)
            self.tensor_offsets = None
            self.offsets = [group_idx * data_stride for group_idx in range(num_tensors + 1)]
            self.scale_inv_offsets = [group_idx * scale_stride for group_idx in range(num_tensors + 1)]
            self.columnwise_scale_inv_offsets = [
                group_idx * columnwise_scale_stride for group_idx in range(num_tensors + 1)
            ]
            self._with_gemm_swizzled_scales = optimize_for_gemm
            self.quantized_tensors = [
                FakeMXFP8Member(
                    self.rowwise_data[group_idx * data_stride : (group_idx + 1) * data_stride],
                    self.columnwise_data[group_idx * data_stride : (group_idx + 1) * data_stride],
                )
                for group_idx in range(num_tensors)
            ]
            self.requires_grad = False

        def requires_grad_(self, requires_grad):
            self.requires_grad = requires_grad
            return self

    class FakeMXFP8Quantizer:
        def __init__(self, fp8_dtype, *, rowwise, columnwise):
            self.fp8_dtype = fp8_dtype
            self.rowwise_usage = rowwise
            self.columnwise_usage = columnwise
            self.internal = True
            self.optimize_for_gemm = True

        def update_quantized(self, source, destination):
            destination.value = source.value
            destination.rowwise_data.fill_(source.value)
            destination.columnwise_data.fill_(source.value)
            return destination

    class FakeGroupedTensorStorage:
        calls = []
        group_quantize_calls = []

        @staticmethod
        def make_grouped_tensor_with_shapes(*, num_tensors, shapes, quantizer, device, dtype):
            FakeGroupedTensorStorage.calls.append(
                {
                    "num_tensors": num_tensors,
                    "shapes": shapes,
                    "quantizer": quantizer,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeGroupedTensor(num_tensors, optimize_for_gemm=quantizer.optimize_for_gemm)

    def fake_group_quantize(source, quantizer, num_tensors, first_dims):
        FakeGroupedTensorStorage.group_quantize_calls.append(
            {
                "source": source,
                "quantizer": quantizer,
                "num_tensors": num_tensors,
                "first_dims": first_dims,
            }
        )
        grouped = FakeGroupedTensor(num_tensors, optimize_for_gemm=quantizer.optimize_for_gemm)
        for destination, value in zip(grouped.quantized_tensors, source.values):
            destination.value = value
            destination.rowwise_data.fill_(value)
            destination.columnwise_data.fill_(value)
        return grouped

    tex_stub = types.ModuleType("transformer_engine_torch")
    tex_stub.DType = types.SimpleNamespace(kFloat8E4M3=object())
    tex_stub.group_quantize = fake_group_quantize
    te_stub = types.ModuleType("transformer_engine")
    te_pytorch_stub = types.ModuleType("transformer_engine.pytorch")
    te_tensor_stub = types.ModuleType("transformer_engine.pytorch.tensor")
    te_tensor_stub.GroupedTensorStorage = FakeGroupedTensorStorage
    te_tensor_stub.MXFP8Quantizer = FakeMXFP8Quantizer

    monkeypatch.setitem(sys.modules, "transformer_engine_torch", tex_stub)
    monkeypatch.setitem(sys.modules, "transformer_engine", te_stub)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_pytorch_stub)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor", te_tensor_stub)
    return FakeGroupedTensorStorage


def test_te_ops_mxfp8_cache_ep2_refreshes_once_ac_recompute_and_keeps_storage(monkeypatch):
    """EP2 cache refresh is versioned, non-reentrant-AC safe, and address stable."""
    from types import SimpleNamespace

    from torch.utils.checkpoint import checkpoint

    from nemo_automodel.components.moe.experts import _TeOpsMXFP8WeightCache

    storage_factory = _install_te_216_mxfp8_cache_stubs(monkeypatch)

    class FakeOwner:
        def __init__(self, *, data_ptr=1200, version=7, values=(3, 5)):
            self.shape = (2, 32, 32)
            self.device = SimpleNamespace(type="cuda")
            self.dtype = torch.bfloat16
            self.requires_grad = True
            self._data_ptr = data_ptr
            self._version = version
            self.values = values

        def data_ptr(self):
            return self._data_ptr

        def is_contiguous(self):
            return True

        def unbind(self, dim):
            assert dim == 0
            return tuple(SimpleNamespace(value=value) for value in self.values)

    owner = FakeOwner()
    cache = _TeOpsMXFP8WeightCache(
        owner,
        num_groups=2,
        out_features=32,
        in_features=32,
    )

    assert len(storage_factory.calls) == 1
    quantizer = storage_factory.calls[0]["quantizer"]
    assert quantizer.rowwise_usage is True
    assert quantizer.columnwise_usage is True
    assert quantizer.internal is False
    assert quantizer.optimize_for_gemm is False
    assert cache.mode == cache.FIXED_ADDRESS_MODE
    assert cache.tensor._with_gemm_swizzled_scales is False
    assert cache.refresh_count == 1
    assert cache.group_quantize_count == 0
    assert cache.member_update_count == 2
    assert storage_factory.group_quantize_calls == []
    assert [member.value for member in cache.members] == [3, 5]
    initial_identity = cache._storage_identity()

    # A detached functional-call/checkpoint alias is a different Python object,
    # but shares data_ptr and version with the registered owner.
    recompute_alias = FakeOwner(data_ptr=owner.data_ptr(), version=owner._version, values=owner.values)
    checkpoint_refreshes = []

    def checkpointed(x):
        checkpoint_refreshes.append(cache.refresh(recompute_alias))
        return x.sin()

    x = torch.randn(8, requires_grad=True)
    checkpoint(checkpointed, x, use_reentrant=False).sum().backward()
    assert checkpoint_refreshes == [False, False]
    assert cache.refresh_count == 1

    # Simulate one optimizer update. The next outer expert forward refreshes
    # once; checkpoint recomputation then sees the same owner generation.
    owner.values = (9, 11)
    owner._version += 1
    assert cache.refresh(owner) is True
    assert cache.refresh(owner) is False
    assert cache.refresh_count == 2
    assert [member.value for member in cache.members] == [9, 11]
    assert cache._storage_identity() == initial_identity
    assert cache.has_stable_storage_identity()

    # Init/load paths can force a refresh even when their in-place copy did not
    # expose a new version through a tensor wrapper.
    assert cache.refresh(owner, force=True) is True
    assert cache.refresh_count == 3
    assert cache.member_update_count == 6
    assert cache._storage_identity() == initial_identity


def test_te_ops_mxfp8_fixed_cache_graph_refresh_tracks_changed_weights_and_python_generation(monkeypatch):
    """Captured kernels and replay bookkeeping match repeated eager force refreshes."""
    from types import SimpleNamespace

    from nemo_automodel.components.moe.experts import _TeOpsMXFP8WeightCache

    _install_te_216_mxfp8_cache_stubs(monkeypatch)

    class FakeOwner:
        def __init__(self):
            self.shape = (2, 32, 32)
            self.device = SimpleNamespace(type="cuda")
            self.dtype = torch.bfloat16
            self.requires_grad = True
            self._version = 1
            self.values = (2, 4)

        def data_ptr(self):
            return 9100

        def is_contiguous(self):
            return True

        def unbind(self, dim):
            assert dim == 0
            return tuple(SimpleNamespace(value=value) for value in self.values)

    graph_owner = FakeOwner()
    eager_owner = FakeOwner()
    graph_cache = _TeOpsMXFP8WeightCache(
        graph_owner,
        num_groups=2,
        out_features=32,
        in_features=32,
    )
    eager_cache = _TeOpsMXFP8WeightCache(
        eager_owner,
        num_groups=2,
        out_features=32,
        in_features=32,
    )
    graph_identity = graph_cache._storage_identity()

    for generation, values in enumerate(((7, 9), (11, 13), (17, 19)), start=2):
        graph_owner.values = values
        graph_owner._version = generation
        eager_owner.values = values
        eager_owner._version = generation

        refreshes_before = graph_cache.refresh_count
        member_updates_before = graph_cache.member_update_count
        graph_cache.capture_fixed_address_refresh(graph_owner)
        # Stream capture launches kernels but must not pretend they executed.
        assert graph_cache.refresh_count == refreshes_before
        assert graph_cache.member_update_count == member_updates_before
        graph_cache.mark_fixed_address_graph_replayed(graph_owner)
        assert eager_cache.refresh(eager_owner, force=True)

        assert [member.value for member in graph_cache.members] == [member.value for member in eager_cache.members]
        assert graph_cache.is_current(graph_owner)
        assert graph_cache.refresh_count == eager_cache.refresh_count
        assert graph_cache.member_update_count == eager_cache.member_update_count
        assert graph_cache._storage_identity() == graph_identity


@pytest.mark.parametrize("mode", ["group_quantize", "gemm_ready_fixed"])
def test_te_ops_mxfp8_cache_refresh_graph_rejects_lazy_modes(monkeypatch, mode):
    from types import SimpleNamespace

    from nemo_automodel.components.moe.experts import _TeOpsMXFP8WeightCache

    _install_te_216_mxfp8_cache_stubs(monkeypatch)

    class FakeOwner:
        shape = (2, 32, 32)
        device = SimpleNamespace(type="cuda")
        dtype = torch.bfloat16
        requires_grad = True
        _version = 1
        values = (2, 4)

        def data_ptr(self):
            return 9200

        def is_contiguous(self):
            return True

        def unbind(self, dim):
            assert dim == 0
            return tuple(SimpleNamespace(value=value) for value in self.values)

        def view(self, *_shape):
            return self

    owner = FakeOwner()
    cache = _TeOpsMXFP8WeightCache(
        owner,
        num_groups=2,
        out_features=32,
        in_features=32,
        mode=mode,
    )

    with pytest.raises(RuntimeError, match="require fixed_address mode"):
        cache.capture_fixed_address_refresh(owner)


def test_te_ops_mxfp8_gemm_ready_fixed_cache_is_lazy_stable_and_padded(monkeypatch):
    """Graph-off default keeps GEMM-ready buffers stable across lazy refreshes."""
    from types import SimpleNamespace

    from torch.utils.checkpoint import checkpoint

    from nemo_automodel.components.moe.experts import _TeOpsMXFP8WeightCache

    storage_factory = _install_te_216_mxfp8_cache_stubs(monkeypatch)

    class FakeOwner:
        def __init__(self, *, data_ptr=1800, version=3, values=(4, 7)):
            self.shape = (2, 32, 32)
            self.device = SimpleNamespace(type="cuda")
            self.dtype = torch.bfloat16
            self.requires_grad = True
            self._data_ptr = data_ptr
            self._version = version
            self.values = values

        def data_ptr(self):
            return self._data_ptr

        def is_contiguous(self):
            return True

        def unbind(self, dim):
            assert dim == 0
            return tuple(SimpleNamespace(value=value) for value in self.values)

    owner = FakeOwner()
    cache = _TeOpsMXFP8WeightCache(
        owner,
        num_groups=2,
        out_features=32,
        in_features=32,
        mode=_TeOpsMXFP8WeightCache.GEMM_READY_FIXED_MODE,
    )

    assert len(storage_factory.calls) == 1
    quantizer = storage_factory.calls[0]["quantizer"]
    assert quantizer.optimize_for_gemm is True
    assert cache.mode == cache.GEMM_READY_FIXED_MODE
    assert cache.tensor._with_gemm_swizzled_scales is True
    assert cache.group_quantize_count == 0
    assert cache.member_update_count == 2
    assert storage_factory.group_quantize_calls == []
    assert [member.value for member in cache.members] == [4, 7]

    initial_tensor = cache.tensor
    initial_identity = cache._storage_identity()
    initial_offsets = (
        tuple(cache.tensor.offsets),
        tuple(cache.tensor.scale_inv_offsets),
        tuple(cache.tensor.columnwise_scale_inv_offsets),
    )
    initial_buffer_sizes = (
        cache.tensor.rowwise_data.numel(),
        cache.tensor.columnwise_data.numel(),
        cache.tensor.scale_inv.numel(),
        cache.tensor.columnwise_scale_inv.numel(),
    )
    assert initial_buffer_sizes == (2048, 2048, 1024, 1024)

    # Model a fused optimizer update that does not bump owner._version. The
    # post-step invalidation itself launches no quantization.
    owner.values = (12, 15)
    cache.invalidate()
    assert cache.invalidated
    assert not cache.is_current(owner)
    assert cache.refresh_count == 1
    assert cache.member_update_count == 2

    # The first next expert call updates every preallocated member in place.
    assert cache.refresh(owner) is True
    assert cache.refresh(owner) is False
    assert cache.tensor is initial_tensor
    assert cache._storage_identity() == initial_identity
    assert cache.has_stable_storage_identity()
    assert cache.member_update_count == 4
    assert [member.value for member in cache.members] == [12, 15]
    assert (
        tuple(cache.tensor.offsets),
        tuple(cache.tensor.scale_inv_offsets),
        tuple(cache.tensor.columnwise_scale_inv_offsets),
    ) == initial_offsets
    assert (
        cache.tensor.rowwise_data.numel(),
        cache.tensor.columnwise_data.numel(),
        cache.tensor.scale_inv.numel(),
        cache.tensor.columnwise_scale_inv.numel(),
    ) == initial_buffer_sizes

    recompute_alias = FakeOwner(data_ptr=owner.data_ptr(), version=owner._version, values=owner.values)
    checkpoint_refreshes = []

    def checkpointed(x):
        checkpoint_refreshes.append(cache.refresh(recompute_alias))
        return x.sin()

    x = torch.randn(8, requires_grad=True)
    checkpoint(checkpointed, x, use_reentrant=False).sum().backward()
    assert checkpoint_refreshes == [False, False]
    assert cache.refresh_count == 2
    assert cache.member_update_count == 4


def test_te_ops_mxfp8_cache_group_quantizes_once_per_owner_generation(monkeypatch):
    """Graph-off cache uses one grouped launch and reuses it through AC recompute."""
    from types import SimpleNamespace

    from torch.utils.checkpoint import checkpoint

    from nemo_automodel.components.moe.experts import _TeOpsMXFP8WeightCache

    storage_factory = _install_te_216_mxfp8_cache_stubs(monkeypatch)

    class FakeOwner:
        def __init__(self, *, data_ptr=2200, version=4, values=(2, 6)):
            self.shape = (2, 32, 32)
            self.device = SimpleNamespace(type="cuda")
            self.dtype = torch.bfloat16
            self.requires_grad = True
            self._data_ptr = data_ptr
            self._version = version
            self.values = values

        def data_ptr(self):
            return self._data_ptr

        def is_contiguous(self):
            return True

        def view(self, *shape):
            assert shape == (64, 32)
            return self

    owner = FakeOwner()
    cache = _TeOpsMXFP8WeightCache(
        owner,
        num_groups=2,
        out_features=32,
        in_features=32,
        mode=_TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE,
    )

    assert storage_factory.calls == []
    assert len(storage_factory.group_quantize_calls) == 1
    call = storage_factory.group_quantize_calls[0]
    assert call["source"] is owner
    assert call["num_tensors"] == 2
    assert call["first_dims"] is None
    assert call["quantizer"].rowwise_usage is True
    assert call["quantizer"].columnwise_usage is True
    assert call["quantizer"].optimize_for_gemm is False
    assert cache.group_quantize_count == 1
    assert cache.member_update_count == 0
    assert cache.buffer_replacement_count == 0
    assert [member.value for member in cache.members] == [2, 6]
    initial_tensor = cache.tensor

    recompute_alias = FakeOwner(data_ptr=owner.data_ptr(), version=owner._version, values=owner.values)
    checkpoint_refreshes = []

    def checkpointed(x):
        checkpoint_refreshes.append(cache.refresh(recompute_alias))
        return x.sin()

    x = torch.randn(8, requires_grad=True)
    checkpoint(checkpointed, x, use_reentrant=False).sum().backward()
    assert checkpoint_refreshes == [False, False]
    assert len(storage_factory.group_quantize_calls) == 1

    # FusedAdam may update parameter.data without changing PyTorch's version.
    # The optimizer boundary must explicitly invalidate the cache without
    # quantizing; the first subsequent expert forward then refreshes it once.
    owner.values = (8, 10)
    recompute_alias.values = owner.values
    cache.invalidate()
    assert cache.invalidated is True
    assert cache.is_current(owner) is False
    assert len(storage_factory.group_quantize_calls) == 1
    assert cache.refresh(owner) is True
    assert cache.refresh(owner) is False
    assert cache.invalidated is False
    assert cache.tensor is not initial_tensor
    assert cache.group_quantize_count == 2
    assert cache.buffer_replacement_count == 1
    assert [member.value for member in cache.members] == [8, 10]
    assert not cache.has_stable_storage_identity()

    checkpoint_refreshes.clear()
    checkpoint(checkpointed, x, use_reentrant=False).sum().backward()
    assert checkpoint_refreshes == [False, False]
    assert len(storage_factory.group_quantize_calls) == 2


def test_te_ops_mxfp8_cache_is_not_optimizer_or_checkpoint_state(monkeypatch):
    """Only the ordinary stacked nn.Parameter is registered model state."""
    from types import SimpleNamespace

    from nemo_automodel.components.moe.experts import _TeOpsMXFP8WeightCache

    _install_te_216_mxfp8_cache_stubs(monkeypatch)

    class FakeOwner:
        shape = (2, 32, 32)
        device = SimpleNamespace(type="cuda")
        dtype = torch.bfloat16
        requires_grad = True
        _version = 0

        def data_ptr(self):
            return 3400

        def is_contiguous(self):
            return True

        def unbind(self, dim):
            assert dim == 0
            return (SimpleNamespace(value=1), SimpleNamespace(value=2))

    cache = _TeOpsMXFP8WeightCache(FakeOwner(), num_groups=2, out_features=32, in_features=32)
    module = torch.nn.Module()
    module.register_parameter("stacked_weight", torch.nn.Parameter(torch.randn(2, 32, 32)))
    module.__dict__["_mxfp8_weight_cache"] = cache

    assert set(dict(module.named_parameters())) == {"stacked_weight"}
    assert set(module.state_dict()) == {"stacked_weight"}


def test_collect_te_ops_mxfp8_weight_cache_diagnostics_is_bounded():
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        collect_te_ops_mxfp8_weight_cache_diagnostics,
    )

    class FakeLinear(torch.nn.Module):
        def __init__(self, cache_id, base_ptr):
            super().__init__()
            self.cache_id = cache_id
            self.base_ptr = base_ptr

        def mxfp8_weight_cache_diagnostics(self):
            return {
                "enabled": True,
                "mode": "fixed_address",
                "optimize_for_gemm": False,
                "allocations": 1,
                "refreshes": 4,
                "group_quantize_calls": 0,
                "member_update_calls": 8,
                "buffer_replacements": 0,
                "optimizer_invalidations": 0,
                "optimizer_refreshes": 3,
                "hits": 7,
                "fallbacks": 0,
                "current": True,
                "dirty": False,
                "requires_fixed_identity": True,
                "identity_policy_satisfied": True,
                "fallback_reason": "",
                "storage": {
                    "cache_id": self.cache_id,
                    "member_ids": (self.cache_id + 1,),
                    "rowwise_data_ptr": self.base_ptr,
                    "columnwise_data_ptr": self.base_ptr + 1,
                    "rowwise_scale_ptr": self.base_ptr + 2,
                    "columnwise_scale_ptr": self.base_ptr + 3,
                    "identity_stable": True,
                },
            }

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_requested = True
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts._te_ops_mxfp8_weight_cache_mode = "fixed_address"
    experts._te_ops_mxfp8_weight_cache_fallback_reason = ""
    experts.gate_up_linear = FakeLinear(10, 100)
    experts.down_linear = FakeLinear(20, 200)
    model = torch.nn.Sequential(experts)

    diagnostics = collect_te_ops_mxfp8_weight_cache_diagnostics((model, model))

    assert diagnostics == {
        "expert_layers": 1,
        "unstacked_layers": 0,
        "requested_layers": 1,
        "enabled_layers": 1,
        "fallback_layers": 0,
        "projection_caches": 2,
        "allocations": 2,
        "refreshes": 8,
        "group_quantize_calls": 0,
        "member_update_calls": 16,
        "buffer_replacements": 0,
        "optimizer_invalidations": 0,
        "optimizer_refreshes": 6,
        "hits": 14,
        "fallbacks": 0,
        "current_caches": 2,
        "dirty_caches": 0,
        "group_quantize_caches": 0,
        "gemm_ready_fixed_caches": 0,
        "fixed_address_caches": 2,
        "gemm_optimized_caches": 0,
        "storage_identity_stable": True,
        "identity_policy_satisfied": True,
        "unique_cache_objects": 2,
        "unique_buffer_sets": 2,
        "fallback_reasons": {},
    }


def test_collect_te_ops_mxfp8_weight_cache_diagnostics_distinguishes_all_modes():
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        collect_te_ops_mxfp8_weight_cache_diagnostics,
    )

    class FakeLinear(torch.nn.Module):
        def __init__(self, mode, optimize_for_gemm, base_ptr):
            super().__init__()
            self.mode = mode
            self.optimize_for_gemm = optimize_for_gemm
            self.base_ptr = base_ptr

        def mxfp8_weight_cache_diagnostics(self):
            return {
                "enabled": True,
                "mode": self.mode,
                "optimize_for_gemm": self.optimize_for_gemm,
                "allocations": 1,
                "refreshes": 1,
                "group_quantize_calls": int(self.mode == "group_quantize"),
                "member_update_calls": int(self.mode != "group_quantize"),
                "buffer_replacements": 0,
                "optimizer_invalidations": int(self.mode != "fixed_address"),
                "optimizer_refreshes": int(self.mode == "fixed_address"),
                "hits": 0,
                "fallbacks": 0,
                "current": True,
                "dirty": False,
                "requires_fixed_identity": self.mode != "group_quantize",
                "identity_policy_satisfied": True,
                "fallback_reason": "",
                "storage": {
                    "cache_id": self.base_ptr,
                    "member_ids": (),
                    "rowwise_data_ptr": self.base_ptr,
                    "columnwise_data_ptr": self.base_ptr + 1,
                    "rowwise_scale_ptr": self.base_ptr + 2,
                    "columnwise_scale_ptr": self.base_ptr + 3,
                    "identity_stable": self.mode != "group_quantize",
                },
            }

    experts_by_mode = []
    for index, (mode, optimize_for_gemm) in enumerate(
        (("group_quantize", False), ("gemm_ready_fixed", True), ("fixed_address", False))
    ):
        experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
        torch.nn.Module.__init__(experts)
        experts._te_ops_mxfp8_weight_cache_requested = True
        experts._te_ops_mxfp8_weight_cache_enabled = True
        experts._te_ops_mxfp8_weight_cache_mode = mode
        experts._te_ops_mxfp8_weight_cache_fallback_reason = ""
        experts.gate_up_linear = FakeLinear(mode, optimize_for_gemm, 100 + index * 20)
        experts.down_linear = FakeLinear(mode, optimize_for_gemm, 110 + index * 20)
        experts_by_mode.append(experts)

    diagnostics = collect_te_ops_mxfp8_weight_cache_diagnostics(torch.nn.Sequential(*experts_by_mode))

    assert diagnostics["group_quantize_caches"] == 2
    assert diagnostics["gemm_ready_fixed_caches"] == 2
    assert diagnostics["fixed_address_caches"] == 2
    assert diagnostics["gemm_optimized_caches"] == 2
    assert diagnostics["optimizer_invalidations"] == 4
    assert diagnostics["optimizer_refreshes"] == 2


@pytest.mark.parametrize("cache_mode", ["group_quantize", "gemm_ready_fixed", "fixed_address"])
def test_te_ops_mxfp8_weight_cache_ep2_shard_policy_falls_back_transparently(cache_mode):
    from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

    class FakeLinear:
        def __init__(self):
            self.calls = []

        def set_mxfp8_weight_cache_enabled(self, enabled, *, fallback_reason, mode):
            self.calls.append((enabled, fallback_reason, mode))

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    experts._te_ops_mxfp8_weight_cache_requested = True
    experts._te_ops_mxfp8_weight_cache_mode = cache_mode
    experts.gate_up_linear = FakeLinear()
    experts.down_linear = FakeLinear()

    # EP2 with ep_shard=1 owns complete local expert matrices.
    experts.configure_mxfp8_weight_cache_for_ep_shard(ep_shard_enabled=False)
    assert experts._te_ops_mxfp8_weight_cache_enabled is True
    assert experts._te_ops_mxfp8_weight_cache_fallback_reason == ""

    # EP2 with ep_shard=2 keeps the same registered owners and uses TE's
    # ordinary post-unshard quantization path.
    experts.configure_mxfp8_weight_cache_for_ep_shard(ep_shard_enabled=True)
    expected_reason = "ep_shard>1 requires FSDP unshard; using eager TE weight quantization"
    assert experts._te_ops_mxfp8_weight_cache_enabled is False
    assert experts._te_ops_mxfp8_weight_cache_fallback_reason == expected_reason
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.calls == [
            (True, "", cache_mode),
            (False, expected_reason, cache_mode),
        ]


@pytest.mark.parametrize("lazy_mode", ["group_quantize", "gemm_ready_fixed"])
def test_te_ops_mxfp8_weight_cache_optimizer_hook_lazily_invalidates_graph_off_cache(lazy_mode):
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        register_te_ops_mxfp8_weight_cache_optimizer_hooks,
    )

    class FakeCachedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("_stacked_weight", torch.nn.Parameter(torch.ones(2, 2)))
            self.__dict__["_mxfp8_weight_cache_enabled"] = True
            self.__dict__["_mxfp8_weight_cache_mode"] = "fixed_address"
            self.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] = 0
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] = 0
            self.dirty = False
            self.invalidate_calls = 0
            self.refresh_calls = []
            self.lazy_refresh_count = 0

        def invalidate_mxfp8_weight_cache(self):
            self.invalidate_calls += 1
            self.dirty = True
            return True

        def refresh_mxfp8_weight_cache_if_needed(self, *, force=False):
            self.refresh_calls.append(force)
            if not force and not self.dirty:
                return False
            self.dirty = False
            self.lazy_refresh_count += 1
            return True

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts.gate_up_linear = FakeCachedLinear()
    experts.down_linear = FakeCachedLinear()
    model = torch.nn.Sequential(experts)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Repeated roots must not register duplicate projection refreshes.
    handles = register_te_ops_mxfp8_weight_cache_optimizer_hooks((model, model), optimizer)
    assert len(handles) == 1
    # Graph discovery runs after hook registration and may leave an unselected
    # expert layer eager. The callback must read its current mode at step time.
    for linear in (experts.gate_up_linear, experts.down_linear):
        linear.__dict__["_mxfp8_weight_cache_mode"] = lazy_mode

    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.invalidate_calls == 1
        assert linear.refresh_calls == []
        assert linear.lazy_refresh_count == 0
        assert linear.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] == 1
        assert linear.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] == 0

    # The first subsequent layer forward refreshes both projections. Further
    # GA microbatches and checkpoint recomputation reuse the same generation.
    assert experts.refresh_mxfp8_weight_cache_if_needed() == 2
    assert experts.refresh_mxfp8_weight_cache_if_needed() == 0
    assert experts.refresh_mxfp8_weight_cache_if_needed() == 0
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.lazy_refresh_count == 1
        assert linear.refresh_calls == [False, False, False]

    # A returned optimizer-internal no-op still advances the generation marker,
    # but performs no grouped quantization until another expert forward.
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.invalidate_calls == 2
        assert linear.lazy_refresh_count == 1
        assert linear.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] == 2

    for handle in handles:
        handle.remove()
    optimizer.step()
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.invalidate_calls == 2


def test_te_ops_mxfp8_fixed_cache_graph_target_excludes_only_owned_fixed_hooks():
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        _TeOpsMXFP8WeightCache,
        build_te_ops_mxfp8_weight_cache_refresh_target,
        register_te_ops_mxfp8_weight_cache_optimizer_hooks,
    )

    class FakeFixedLinear(torch.nn.Module):
        def __init__(self, pointer):
            super().__init__()
            self.register_parameter("_stacked_weight", torch.nn.Parameter(torch.ones(2, 2)))
            self.__dict__["_mxfp8_weight_cache_mode"] = _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] = 0
            self.pointer = pointer
            self.events = []

        def mxfp8_weight_cache_graph_signature(self):
            return (self.pointer,)

        def refresh_mxfp8_weight_cache_if_needed(self, *, force=False):
            self.events.append(("eager", force))
            return True

        def capture_mxfp8_weight_cache_refresh(self):
            self.events.append("capture")

        def mark_mxfp8_weight_cache_refresh_graph_replayed(self):
            self.events.append("mark")
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] += 1

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts.gate_up_linear = FakeFixedLinear(101)
    experts.down_linear = FakeFixedLinear(103)
    optimizer = torch.optim.SGD(experts.parameters(), lr=0.1)

    target = build_te_ops_mxfp8_weight_cache_refresh_target(experts, optimizer)
    assert target is not None
    assert target.managed_owner_ids == frozenset(id(parameter) for parameter in experts.parameters())
    assert target.graph_signature() == ((101,), (103,))
    assert target.eager_refresh() == 2
    target.capture_refresh()
    assert target.mark_replayed() == 2
    assert experts.gate_up_linear.events == [("eager", True), "capture", "mark"]
    assert experts.down_linear.events == [("eager", True), "capture", "mark"]

    handles = register_te_ops_mxfp8_weight_cache_optimizer_hooks(
        experts,
        optimizer,
        excluded_owner_ids=target.managed_owner_ids,
    )
    assert handles == ()
    optimizer.step()
    assert experts.gate_up_linear.events == [("eager", True), "capture", "mark"]
    assert experts.down_linear.events == [("eager", True), "capture", "mark"]


def test_te_ops_mxfp8_weight_cache_optimizer_hook_synchronously_refreshes_fixed_address_cache():
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        register_te_ops_mxfp8_weight_cache_optimizer_hooks,
    )

    class FakeCachedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("_stacked_weight", torch.nn.Parameter(torch.ones(2, 2)))
            self.__dict__["_mxfp8_weight_cache_enabled"] = True
            self.__dict__["_mxfp8_weight_cache_mode"] = "fixed_address"
            self.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] = 0
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] = 0
            self.refresh_calls = []

        def invalidate_mxfp8_weight_cache(self):
            raise AssertionError("fixed-address graph caches must not be invalidated")

        def refresh_mxfp8_weight_cache_if_needed(self, *, force=False):
            self.refresh_calls.append(force)
            return True

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts.gate_up_linear = FakeCachedLinear()
    experts.down_linear = FakeCachedLinear()
    optimizer = torch.optim.SGD(experts.parameters(), lr=0.1)
    handles = register_te_ops_mxfp8_weight_cache_optimizer_hooks(experts, optimizer)

    optimizer.step()
    for linear in (experts.gate_up_linear, experts.down_linear):
        assert linear.refresh_calls == [True]
        assert linear.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] == 0
        assert linear.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] == 1

    for handle in handles:
        handle.remove()


@pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
def test_te_ops_mxfp8_cache_refresh_graph_matches_eager_after_remainder_fused_adam_steps():
    """Real SM100 quantization replay follows changing BF16 owners over multiple steps."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 grouped quantization requires SM100+")

    from transformer_engine.pytorch.optimizers import FusedAdam

    from nemo_automodel.components.moe.experts import (
        TeOpsMXFP8WeightCacheRefreshTarget,
        _get_stacked_te_ops_grouped_linear_class,
        _TeOpsMXFP8WeightCache,
    )
    from nemo_automodel.recipes.llm.mxfp8_cache_refresh_cuda_graph import (
        MXFP8CacheRefreshCudaGraphManager,
    )

    linear_cls = _get_stacked_te_ops_grouped_linear_class()
    eager_linear = linear_cls(2, 32, 32, bias=False, dtype=torch.bfloat16, device="cuda")
    graph_linear = linear_cls(2, 32, 32, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        graph_linear._parameters["_stacked_weight"].copy_(eager_linear._parameters["_stacked_weight"])
    for linear in (eager_linear, graph_linear):
        linear.set_mxfp8_weight_cache_enabled(
            True,
            fallback_reason="",
            mode=_TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE,
        )

    eager_optimizer = FusedAdam(
        eager_linear.parameters(),
        lr=0.01,
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    graph_optimizer = FusedAdam(
        graph_linear.parameters(),
        lr=0.01,
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    graph_owner = graph_linear._parameters["_stacked_weight"]
    target = TeOpsMXFP8WeightCacheRefreshTarget((graph_linear,), (graph_owner,), graph_optimizer)
    manager = MXFP8CacheRefreshCudaGraphManager(target)

    try:
        for step, capture_allowed in enumerate((False, True, True, True), start=1):
            eager_owner = eager_linear._parameters["_stacked_weight"]
            gradient = torch.linspace(-1.0, 1.0, eager_owner.numel(), device="cuda", dtype=torch.bfloat16)
            gradient = gradient.reshape_as(eager_owner).mul_(step)
            eager_owner.grad = gradient.clone()
            graph_owner.grad = gradient.clone()

            eager_optimizer.step()
            graph_optimizer.step()
            assert eager_linear.refresh_mxfp8_weight_cache_if_needed(force=True)
            manager.run(capture_allowed=capture_allowed)
            torch.cuda.synchronize()

            torch.testing.assert_close(graph_owner, eager_owner, rtol=0, atol=0)
            eager_cache = eager_linear.__dict__["_mxfp8_weight_cache"].tensor
            graph_cache = graph_linear.__dict__["_mxfp8_weight_cache"].tensor
            for graph_buffer, eager_buffer in (
                (graph_cache.rowwise_data, eager_cache.rowwise_data),
                (graph_cache.columnwise_data, eager_cache.columnwise_data),
                (graph_cache.scale_inv, eager_cache.scale_inv),
                (graph_cache.columnwise_scale_inv, eager_cache.columnwise_scale_inv),
            ):
                torch.testing.assert_close(graph_buffer, eager_buffer, rtol=0, atol=0)
            assert graph_linear.__dict__["_mxfp8_weight_cache"].is_current(graph_owner)

        assert graph_optimizer.store_param_remainders is True
        assert manager.capture_count == 1
        assert manager.replay_count == 2
    finally:
        manager.close()


def test_te_ops_mxfp8_weight_cache_optimizer_hook_rejects_unowned_trainable_params():
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        register_te_ops_mxfp8_weight_cache_optimizer_hooks,
    )

    class FakeCachedLinear(torch.nn.Module):
        def __init__(self, *, requires_grad):
            super().__init__()
            self.register_parameter(
                "_stacked_weight",
                torch.nn.Parameter(torch.ones(2, 2), requires_grad=requires_grad),
            )

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts.gate_up_linear = FakeCachedLinear(requires_grad=True)
    experts.down_linear = FakeCachedLinear(requires_grad=False)
    unrelated = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([unrelated], lr=0.1)

    with pytest.raises(RuntimeError, match="trainable stacked owners missing"):
        register_te_ops_mxfp8_weight_cache_optimizer_hooks(experts, optimizer)

    # Intentionally frozen caches need no generation hook.
    experts.gate_up_linear._parameters["_stacked_weight"].requires_grad_(False)
    assert register_te_ops_mxfp8_weight_cache_optimizer_hooks(experts, optimizer) == ()


def test_te_ops_mxfp8_weight_cache_optimizer_hooks_reject_overlapping_owners():
    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        register_te_ops_mxfp8_weight_cache_optimizer_hooks,
    )

    class FakeCachedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("_stacked_weight", torch.nn.Parameter(torch.ones(2, 2)))

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts.gate_up_linear = FakeCachedLinear()
    experts.down_linear = FakeCachedLinear()
    parameters = list(experts.parameters())
    optimizers = (torch.optim.SGD(parameters, lr=0.1), torch.optim.Adam(parameters, lr=0.1))

    with pytest.raises(RuntimeError, match="multiple optimizer param_groups"):
        register_te_ops_mxfp8_weight_cache_optimizer_hooks(experts, optimizers)


@pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
@pytest.mark.parametrize("lazy_mode", ["group_quantize", "gemm_ready_fixed"])
def test_te_fused_adam_post_step_lazily_invalidates_graph_off_cache(lazy_mode):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 grouped quantization requires SM100+")

    import transformer_engine_torch as tex
    from transformer_engine.pytorch.optimizers import FusedAdam
    from transformer_engine.pytorch.tensor import GroupedTensorStorage, MXFP8Quantizer

    from nemo_automodel.components.moe.experts import (
        GroupedExpertsTeOps,
        _TeOpsMXFP8WeightCache,
        register_te_ops_mxfp8_weight_cache_optimizer_hooks,
    )

    class CachedLinear(torch.nn.Module):
        def __init__(self, mode):
            super().__init__()
            owner = torch.nn.Parameter(torch.randn(2, 32, 32, device="cuda", dtype=torch.bfloat16))
            self.register_parameter("_stacked_weight", owner)
            self.__dict__["_mxfp8_weight_cache_enabled"] = True
            self.__dict__["_mxfp8_weight_cache_mode"] = mode
            self.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] = 0
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] = 0
            self.cache = _TeOpsMXFP8WeightCache(
                owner,
                num_groups=2,
                out_features=32,
                in_features=32,
                mode=mode,
            )

        def refresh_mxfp8_weight_cache_if_needed(self, *, force=False):
            return self.cache.refresh(self._parameters["_stacked_weight"], force=force)

        def invalidate_mxfp8_weight_cache(self):
            self.cache.invalidate()
            return True

    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    torch.nn.Module.__init__(experts)
    experts._te_ops_mxfp8_weight_cache_enabled = True
    experts.gate_up_linear = CachedLinear(lazy_mode)
    experts.down_linear = CachedLinear(lazy_mode)
    optimizer = FusedAdam(
        experts.parameters(),
        lr=0.01,
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    handles = register_te_ops_mxfp8_weight_cache_optimizer_hooks(experts, optimizer)
    assert len(handles) == 1

    before = {name: parameter.detach().clone() for name, parameter in experts.named_parameters()}
    before_refreshes = {
        id(linear): linear.cache.refresh_count for linear in (experts.gate_up_linear, experts.down_linear)
    }
    before_group_quantize_calls = {
        id(linear): linear.cache.group_quantize_count for linear in (experts.gate_up_linear, experts.down_linear)
    }
    before_member_update_calls = {
        id(linear): linear.cache.member_update_count for linear in (experts.gate_up_linear, experts.down_linear)
    }
    for parameter in experts.parameters():
        parameter.grad = torch.ones_like(parameter)

    try:
        optimizer.step()
        torch.cuda.synchronize()
        for name, parameter in experts.named_parameters():
            assert not torch.equal(parameter, before[name])
        for linear in (experts.gate_up_linear, experts.down_linear):
            owner = linear._parameters["_stacked_weight"]
            cache = linear.cache
            assert cache.refresh_count == before_refreshes[id(linear)]
            assert cache.group_quantize_count == before_group_quantize_calls[id(linear)]
            assert not cache.is_current(owner)
            assert cache.invalidated
            assert linear.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] == 1
            assert linear.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] == 0

        assert experts.refresh_mxfp8_weight_cache_if_needed() == 2
        torch.cuda.synchronize()
        assert experts.refresh_mxfp8_weight_cache_if_needed() == 0
        for linear in (experts.gate_up_linear, experts.down_linear):
            owner = linear._parameters["_stacked_weight"]
            cache = linear.cache
            assert cache.refresh_count == before_refreshes[id(linear)] + 1
            if lazy_mode == _TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE:
                assert cache.group_quantize_count == before_group_quantize_calls[id(linear)] + 1
                assert cache.member_update_count == before_member_update_calls[id(linear)]
            else:
                assert cache.group_quantize_count == before_group_quantize_calls[id(linear)]
                assert cache.member_update_count == before_member_update_calls[id(linear)] + 2
            assert cache.is_current(owner)
            assert not cache.invalidated

            reference_quantizer = MXFP8Quantizer(
                tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=True,
            )
            reference_quantizer.internal = False
            reference_quantizer.optimize_for_gemm = lazy_mode == _TeOpsMXFP8WeightCache.GEMM_READY_FIXED_MODE
            if lazy_mode == _TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE:
                reference = tex.group_quantize(owner.detach().view(64, 32), reference_quantizer, 2, None)
            else:
                # Optimized bidirectional grouped quantization requires per-member
                # padded offsets. Build the same supported storage shape explicitly;
                # group_quantize(..., first_dims=None) has no such offsets for E>1.
                reference = GroupedTensorStorage.make_grouped_tensor_with_shapes(
                    num_tensors=2,
                    shapes=[(32, 32)] * 2,
                    quantizer=reference_quantizer,
                    device=owner.device,
                    dtype=owner.dtype,
                )
                with torch.no_grad():
                    for source, destination in zip(owner.detach().unbind(0), reference.quantized_tensors):
                        reference_quantizer.update_quantized(source, destination)
            assert torch.equal(cache.tensor.rowwise_data, reference.rowwise_data)
            assert torch.equal(cache.tensor.columnwise_data, reference.columnwise_data)
            assert torch.equal(cache.tensor.scale_inv, reference.scale_inv)
            assert torch.equal(cache.tensor.columnwise_scale_inv, reference.columnwise_scale_inv)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(SKIP_TE_TESTS, reason="TransformerEngine and CUDA required")
class TestGroupedExpertsTeOps:
    """Test the stacked-owner contract for TE fusible expert ops."""

    @pytest.fixture
    def te_ops_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=0,
            n_activated_experts=1,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=64,
            inter_dim=64,
            moe_inter_dim=64,
            norm_topk_prob=False,
            router_bias=False,
            expert_bias=True,
            expert_activation="quick_geglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            dtype=torch.bfloat16,
        )

    @pytest.mark.parametrize(
        ("activation", "expected_ops", "expected_gate_up_width"),
        [
            ("swiglu", ["_StackedTeOpsGroupedLinear", "ScaledSwiGLU", "_StackedTeOpsGroupedLinear"], 128),
            (
                "swiglu_step",
                ["_StackedTeOpsGroupedLinear", "ScaledSwiGLU", "_StackedTeOpsGroupedLinear"],
                128,
            ),
            (
                "swigluoai",
                ["_StackedTeOpsGroupedLinear", "ScaledClampedQGeGLU", "_StackedTeOpsGroupedLinear"],
                128,
            ),
            (
                "quick_geglu",
                ["_StackedTeOpsGroupedLinear", "ScaledClampedQGeGLU", "_StackedTeOpsGroupedLinear"],
                128,
            ),
            (
                "geglu",
                ["_StackedTeOpsGroupedLinear", "GEGLU", "_TeOpsRowScale", "_StackedTeOpsGroupedLinear"],
                128,
            ),
            (
                "relu2",
                ["_StackedTeOpsGroupedLinear", "ScaledSReLU", "_StackedTeOpsGroupedLinear"],
                64,
            ),
        ],
    )
    @pytest.mark.parametrize("expert_bias", [False, True])
    def test_generic_activation_pipeline_and_projection_width(
        self,
        te_ops_config,
        activation,
        expected_ops,
        expected_gate_up_width,
        expert_bias,
    ):
        """TE-ops constructs the right native activation and gated/non-gated width."""
        from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

        te_ops_config.expert_activation = activation
        te_ops_config.expert_bias = expert_bias
        backend = BackendConfig(experts="te_ops", dispatcher="hybridep")
        experts = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")

        assert experts.gate_up_linear.out_features == expected_gate_up_width
        assert experts.gate_up_linear.use_bias is expert_bias
        assert experts.down_linear.use_bias is expert_bias
        if activation == "relu2":
            from transformer_engine.pytorch import ops as te_ops

            if not hasattr(te_ops, "ScaledSReLU"):
                expected_ops = [
                    "_StackedTeOpsGroupedLinear",
                    "SReLU",
                    "_TeOpsRowScale",
                    "_StackedTeOpsGroupedLinear",
                ]
        assert [type(op).__name__ for op in experts._te_grouped_mlp] == expected_ops
        assert sum(op.num_extra_inputs for op in experts._te_grouped_mlp) == (4 if expert_bias else 3)
        expected_parameters = {
            "gate_up_linear._stacked_weight",
            "down_linear._stacked_weight",
        }
        if expert_bias:
            expected_parameters.update(("gate_up_linear._stacked_bias", "down_linear._stacked_bias"))
        assert set(dict(experts.named_parameters())) == expected_parameters

    @pytest.mark.parametrize(
        ("input_dtype", "scale_dtype", "grad_dtype"),
        [
            (torch.float32, torch.float32, torch.float32),
            (torch.float32, torch.bfloat16, torch.bfloat16),
        ],
    )
    def test_custom_row_scale_forward_backward_parity(self, input_dtype, scale_dtype, grad_dtype):
        """The fusible fallback scale differentiates both activations and router probs."""
        from nemo_automodel.components.moe.experts import _get_te_ops_custom_classes

        row_scale, _ = _get_te_ops_custom_classes()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        x_ref = torch.randn(7, 13, device=device, dtype=input_dtype, requires_grad=True)
        scales_ref = torch.randn(7, device=device, dtype=scale_dtype, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_()
        scales_test = scales_ref.detach().clone().requires_grad_()
        grad = torch.randn(7, 13, device=device, dtype=grad_dtype)

        output_ref = x_ref * scales_ref.unsqueeze(-1)
        output_test = row_scale()(x_test, scales_test)
        output_ref.backward(grad)
        output_test.backward(grad)

        torch.testing.assert_close(output_test, output_ref)
        torch.testing.assert_close(x_test.grad, x_ref.grad)
        torch.testing.assert_close(scales_test.grad, scales_ref.grad)

    @pytest.mark.parametrize(
        ("alpha", "linear_offset", "limit"),
        [(1.0, 0.0, 1.5), (1.3, 1.0, None)],
    )
    def test_custom_exact_gated_activation_forward_backward_parity(self, alpha, linear_offset, limit):
        """Exact fallbacks preserve FP32 through activation and backward."""
        from nemo_automodel.components.moe.experts import _get_te_ops_custom_classes

        _, exact_gated_activation = _get_te_ops_custom_classes()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        x_ref = (torch.randn(7, 26, device=device, dtype=torch.bfloat16) * 2.0).requires_grad_()
        x_test = x_ref.detach().clone().requires_grad_()
        grad = torch.randn(7, 13, device=device, dtype=torch.float32)

        gate, up = x_ref.chunk(2, dim=-1)
        gate = gate.float()
        up = up.float()
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        activated_gate = torch.nn.functional.silu(gate) if alpha == 1.0 else gate * torch.sigmoid(alpha * gate)
        output_ref = activated_gate * (up + linear_offset)
        output_test = exact_gated_activation(alpha=alpha, linear_offset=linear_offset, limit=limit)(x_test)
        output_ref.backward(grad)
        output_test.backward(grad)

        assert output_test.dtype == torch.float32
        torch.testing.assert_close(output_test, output_ref)
        torch.testing.assert_close(x_test.grad, x_ref.grad, atol=2e-2, rtol=2e-2)

    def test_custom_step_gated_activation_bf16_forward_backward_parity(self):
        """Step's post-SiLU clamp remains exact in the TE-ops fallback."""
        from nemo_automodel.components.moe.experts import _get_te_ops_custom_classes

        _, exact_gated_activation = _get_te_ops_custom_classes()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        x_ref = (torch.randn(7, 26, device=device, dtype=torch.bfloat16) * 3).requires_grad_()
        x_test = x_ref.detach().clone().requires_grad_()
        grad = torch.randn(7, 13, device=device, dtype=torch.bfloat16)
        limit = 1.5

        gate, up = x_ref.chunk(2, dim=-1)
        output_ref = torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)
        output_test = exact_gated_activation(
            alpha=1.0,
            linear_offset=0.0,
            limit=limit,
            clamp_after_gate_activation=True,
            use_input_dtype=True,
        )(x_test)
        output_ref.backward(grad)
        output_test.backward(grad)

        torch.testing.assert_close(output_test, output_ref, atol=0, rtol=0)
        torch.testing.assert_close(x_test.grad, x_ref.grad, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize(
        ("activation", "swiglu_limit", "activation_limit"),
        [("swiglu", 1.5, 7.0), ("swigluoai", 0.0, 0.0)],
    )
    def test_exact_fp32_full_expert_forward_backward_parity(
        self,
        te_ops_config,
        activation,
        swiglu_limit,
        activation_limit,
    ):
        """Full TE experts match eager math when routes scale FP32 activations."""
        from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
        from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

        class LocalPermutationDispatcher:
            def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
                self.num_tokens = num_local_tokens
                flat_indices = token_indices.reshape(-1)
                flat_probs = token_probs.reshape(-1)
                token_ids = (
                    torch.arange(num_local_tokens, device=hidden_states.device)
                    .unsqueeze(1)
                    .expand_as(token_indices)
                    .reshape(-1)
                )
                order = torch.argsort(flat_indices, stable=True)
                self.token_ids = token_ids[order]
                splits = torch.bincount(flat_indices, minlength=te_ops_config.n_routed_experts)
                return hidden_states[self.token_ids], splits, flat_probs[order]

            def token_unpermutation(self, hidden_states):
                output = torch.zeros(
                    self.num_tokens,
                    hidden_states.shape[-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                return output.index_add(0, self.token_ids, hidden_states)

        te_ops_config.n_activated_experts = 2
        te_ops_config.expert_bias = False
        te_ops_config.expert_activation = activation
        te_ops_config.swiglu_limit = swiglu_limit
        te_ops_config.activation_limit = activation_limit
        backend = BackendConfig(experts="te_ops", dispatcher="hybridep")
        experts = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        to_empty_parameters_only(experts, device=device)
        experts.ep_size = 1
        experts.token_dispatcher = LocalPermutationDispatcher()

        generator = torch.Generator(device="cpu").manual_seed(9817)

        def sample(shape, scale=1.0):
            return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=torch.bfloat16)

        num_tokens = 6
        gate_up = sample((2, 64, 128), scale=0.2)
        down = sample((2, 64, 64), scale=0.1)
        experts.gate_and_up_projs = gate_up
        experts.down_projs = down

        gate_up_ref = gate_up.detach().clone().requires_grad_()
        down_ref = down.detach().clone().requires_grad_()
        x_ref = sample((num_tokens, 64)).requires_grad_()
        x_test = x_ref.detach().clone().requires_grad_()
        indices = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
            device=device,
            dtype=torch.int64,
        )
        probs_ref = torch.tensor(
            [[0.13, 0.87], [0.71, 0.29], [0.43, 0.57], [0.19, 0.81], [0.62, 0.38], [0.34, 0.66]],
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        probs_test = probs_ref.detach().clone().requires_grad_()

        flat_experts = indices.reshape(-1)
        flat_token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand_as(indices).reshape(-1)
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts = flat_experts[order]
        sorted_token_ids = flat_token_ids[order]
        sorted_probs = probs_ref.reshape(-1)[order]
        splits = torch.bincount(sorted_experts, minlength=te_ops_config.n_routed_experts).tolist()
        grouped_hidden = x_ref[sorted_token_ids]
        projected = torch.cat(
            [chunk @ gate_up_ref[expert_idx] for expert_idx, chunk in enumerate(grouped_hidden.split(splits))]
        )
        gate, up = projected.chunk(2, dim=-1)
        gate = gate.float()
        up = up.float()
        if activation == "swiglu":
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            activated = torch.nn.functional.silu(gate) * up
        else:
            activated = gate * torch.sigmoid(te_ops_config.activation_alpha * gate) * (up + 1.0)
        routed = (activated * sorted_probs.float().unsqueeze(-1)).to(torch.bfloat16)
        route_outputs = torch.cat(
            [chunk @ down_ref[expert_idx] for expert_idx, chunk in enumerate(routed.split(splits))]
        )
        output_ref = torch.zeros_like(x_ref).index_add(0, sorted_token_ids, route_outputs)

        token_mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
        output_test = experts(x_test, token_mask, probs_test, indices)
        grad_output = sample(output_ref.shape)
        output_ref.backward(grad_output)
        output_test.backward(grad_output)

        torch.testing.assert_close(output_test, output_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(x_test.grad, x_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(probs_test.grad, probs_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            experts.gate_up_linear._stacked_weight.grad.transpose(-1, -2),
            gate_up_ref.grad,
            atol=3e-2,
            rtol=3e-2,
        )
        torch.testing.assert_close(
            experts.down_linear._stacked_weight.grad.transpose(-1, -2),
            down_ref.grad,
            atol=3e-2,
            rtol=3e-2,
        )
        assert torch.count_nonzero(probs_test.grad).item() == probs_test.numel()

    def test_stacked_owner_alias_and_materialization_identity(self, te_ops_config):
        from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
        from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

        backend = BackendConfig(experts="te_ops", dispatcher="hybridep")
        experts = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")
        expected_names = {
            "gate_up_linear._stacked_weight",
            "gate_up_linear._stacked_bias",
            "down_linear._stacked_weight",
            "down_linear._stacked_bias",
        }
        assert set(dict(experts.named_parameters())) == expected_names

        parameter_ids = {name: id(parameter) for name, parameter in experts.named_parameters()}
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        to_empty_parameters_only(experts, device=device)
        experts.init_weights(device)
        assert {name: id(parameter) for name, parameter in experts.named_parameters()} == parameter_ids

        for linear in (experts.gate_up_linear, experts.down_linear):
            assert [id(parameter) for parameter in linear.parameters()] == [
                id(linear._stacked_weight),
                id(linear._stacked_bias),
            ]
            weight_alias = linear.weight
            bias_alias = linear.bias
            assert type(weight_alias).__name__ == "GroupedTensor"
            assert type(bias_alias).__name__ == "GroupedTensor"
            assert weight_alias.rowwise_data.data_ptr() == linear._stacked_weight.data_ptr()
            assert bias_alias.rowwise_data.data_ptr() == linear._stacked_bias.data_ptr()
            assert weight_alias.requires_grad == linear._stacked_weight.requires_grad
            assert bias_alias.requires_grad == linear._stacked_bias.requires_grad

    def test_unstacked_owner_materialization_and_virtual_state_dict(self, te_ops_config):
        """Native TE owners stay stable while exposing the canonical expert checkpoint."""
        from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
        from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

        backend = BackendConfig(
            experts="te_ops",
            dispatcher="hybridep",
            te_ops_unstacked_parameters=True,
        )
        source = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")
        destination = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")
        expected_names = {
            "gate_up_linear.weight0",
            "gate_up_linear.weight1",
            "gate_up_linear.bias0",
            "gate_up_linear.bias1",
            "down_linear.weight0",
            "down_linear.weight1",
            "down_linear.bias0",
            "down_linear.bias1",
        }
        assert set(dict(source.named_parameters())) == expected_names

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        to_empty_parameters_only(source, device=device)
        to_empty_parameters_only(destination, device=device)
        source_ids = {name: id(parameter) for name, parameter in source.named_parameters()}
        source.init_weights(device)
        destination.init_weights(device)
        assert {name: id(parameter) for name, parameter in source.named_parameters()} == source_ids

        state = source.state_dict()
        destination.load_state_dict(state)
        assert set(state) == {
            "gate_and_up_projs",
            "down_projs",
            "gate_up_proj_bias",
            "down_proj_bias",
        }
        for key, value in state.items():
            torch.testing.assert_close(destination.state_dict()[key], value)

    @pytest.mark.parametrize("activation", ["swiglu", "swiglu_step", "swigluoai", "quick_geglu", "geglu", "relu2"])
    @pytest.mark.parametrize("expert_bias", [False, True])
    def test_virtual_state_dict_roundtrip_preserves_stacked_owners(
        self,
        te_ops_config,
        activation,
        expert_bias,
    ):
        from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
        from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

        te_ops_config.expert_activation = activation
        te_ops_config.expert_bias = expert_bias
        backend = BackendConfig(experts="te_ops", dispatcher="hybridep")
        source = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")
        destination = GroupedExpertsTeOps(te_ops_config, backend=backend, dispatcher_backend="hybridep")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        to_empty_parameters_only(source, device=device)
        to_empty_parameters_only(destination, device=device)
        source.init_weights(device)
        destination.init_weights(device)

        state = source.state_dict()
        expected_keys = {"gate_and_up_projs", "down_projs"}
        if expert_bias:
            expected_keys.update(("gate_up_proj_bias", "down_proj_bias"))
        assert set(state) == expected_keys
        expected_gate_up_width = te_ops_config.moe_inter_dim * (2 if is_gated_activation(activation) else 1)
        assert state["gate_and_up_projs"].shape == (
            te_ops_config.n_routed_experts,
            te_ops_config.expert_dim,
            expected_gate_up_width,
        )
        destination_ids = {name: id(parameter) for name, parameter in destination.named_parameters()}
        destination.load_state_dict(state)
        assert {name: id(parameter) for name, parameter in destination.named_parameters()} == destination_ids
        for key, value in state.items():
            torch.testing.assert_close(destination.state_dict()[key], value)


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
        expected_gate_up_bias_shape = (
            te_moe_config_with_bias.n_routed_experts,
            te_moe_config_with_bias.moe_inter_dim * 2,
        )
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

        experts._load_from_state_dict(state_dict, "", None, True, missing_keys, unexpected_keys, error_msgs)

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
        torch.testing.assert_close(experts1.gate_and_up_projs, experts2.gate_and_up_projs, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(experts1.down_projs, experts2.down_projs, rtol=1e-4, atol=1e-4)

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


class TestPermuteTokensForGroupedMM:
    """Test _permute_tokens_for_grouped_mm helper function."""

    def test_basic_permutation(self, device):
        """Test tokens are sorted by expert and outputs have correct shapes."""
        n_local_experts = 4
        num_tokens = 8
        topk = 2
        indices = torch.tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3], [0, 1], [2, 3], [0, 2], [1, 3]],
            device=device,
        )
        weights = torch.rand(num_tokens, topk, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts, experts_start_idx=0
        )

        # All 16 slots should be assigned (8 tokens * topk=2, all local)
        assert sorted_ids.shape[0] == num_tokens * topk
        assert sorted_weights.shape == sorted_ids.shape
        assert tpe.shape == (n_local_experts,)
        assert offs.shape == (n_local_experts,)
        assert offs.dtype == torch.int32
        assert tpe.sum().item() == num_tokens * topk
        # offs is cumulative sum
        torch.testing.assert_close(offs, tpe.cumsum(0).to(torch.int32))

    def test_expert_offset(self, device):
        """Test that experts_start_idx correctly filters to local experts."""
        # 8 experts total, local experts are 4..7
        indices = torch.tensor([[0, 5], [3, 7], [4, 6]], device=device)
        weights = torch.ones(3, 2, device=device)
        token_mask = torch.ones(3, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=4, experts_start_idx=4
        )

        # Only experts 4,5,6,7 are local. Assignments: token0->5, token1->7, token2->4, token2->6
        assert tpe.sum().item() == 4
        assert tpe[0].item() == 1  # expert 4: token2
        assert tpe[1].item() == 1  # expert 5: token0
        assert tpe[2].item() == 1  # expert 6: token2
        assert tpe[3].item() == 1  # expert 7: token1

    def test_masked_tokens_excluded(self, device):
        """Test that masked tokens are excluded from permutation."""
        indices = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], device=device)
        weights = torch.ones(4, 2, device=device)
        token_mask = torch.tensor([True, True, False, False], device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=2, experts_start_idx=0
        )

        # Only first 2 tokens are valid -> 4 assignments (2 tokens * topk=2)
        assert tpe.sum().item() == 4

    def test_no_local_tokens(self, device):
        """Test when no tokens route to local experts."""
        # Local experts 0..1, all indices go to experts 2,3
        indices = torch.tensor([[2, 3], [2, 3]], device=device)
        weights = torch.ones(2, 2, device=device)
        token_mask = torch.ones(2, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=2, experts_start_idx=0
        )

        assert tpe.sum().item() == 0
        assert sorted_ids.shape[0] == 0

    def test_weights_preserved(self, device):
        """Test that sorted weights correspond to the correct tokens."""
        indices = torch.tensor([[1, 0]], device=device)  # token 0 -> expert 1 (slot0), expert 0 (slot1)
        weights = torch.tensor([[0.7, 0.3]], device=device)
        token_mask = torch.ones(1, dtype=torch.bool, device=device)

        sorted_ids, sorted_weights, tpe, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts=2, experts_start_idx=0
        )

        # Sorted by expert: expert 0 first (weight 0.3), expert 1 second (weight 0.7)
        assert tpe[0].item() == 1  # expert 0
        assert tpe[1].item() == 1  # expert 1
        torch.testing.assert_close(sorted_weights[0], torch.tensor(0.3, device=device))
        torch.testing.assert_close(sorted_weights[1], torch.tensor(0.7, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
class TestTorchGroupedMM:
    """Test GroupedExperts with torch._grouped_mm backend (use_torch_mm=True)."""

    @pytest.fixture
    def torch_mm_backend(self):
        return BackendConfig(experts="torch_mm", dispatcher="torch")

    @pytest.fixture
    def torch_mm_config(self):
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
    def torch_mm_config_with_bias(self, torch_mm_config):
        torch_mm_config.expert_bias = True
        return torch_mm_config

    @staticmethod
    def _unwrap_compiled(fn):
        """Unwrap a torch.compile decorated function to its eager version."""
        from functools import partial

        if isinstance(fn, partial):
            inner = TestTorchGroupedMM._unwrap_compiled(fn.func)
            if inner is not fn.func:
                return partial(inner, *fn.args, **fn.keywords)
            return fn
        if hasattr(fn, "_torchdynamo_orig_callable"):
            return fn._torchdynamo_orig_callable
        return fn

    def _init_experts(self, config, backend, device):
        """Create and initialize GroupedExperts on device."""
        experts = GroupedExperts(config, backend=backend).to(device)
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            if experts.expert_bias:
                experts.gate_up_proj_bias.zero_()
                experts.down_proj_bias.zero_()
        # Use eager (non-compiled) activation functions to avoid recompilation issues in tests
        experts.expert_activation_grouped = self._unwrap_compiled(experts.expert_activation_grouped)
        return experts

    def test_init_sets_use_torch_mm(self, torch_mm_config, torch_mm_backend):
        """Test that use_torch_mm flag is set correctly."""
        experts = GroupedExperts(torch_mm_config, backend=torch_mm_backend)
        assert experts.use_torch_mm is True
        assert hasattr(experts, "expert_activation_grouped")

    def test_init_without_backend_disables_torch_mm(self, torch_mm_config):
        """Test that use_torch_mm is False without backend."""
        experts = GroupedExperts(torch_mm_config)
        assert experts.use_torch_mm is False
        # expert_activation_grouped is always initialized (used by both loop and grouped_mm paths)
        assert hasattr(experts, "expert_activation_grouped")

    def test_forward_shape(self, torch_mm_config, torch_mm_backend, device):
        """Test grouped_mm forward produces correct output shape."""
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert output.device == device
        assert not torch.isnan(output).any()

    def test_forward_with_bias(self, torch_mm_config_with_bias, torch_mm_backend, device):
        """Test grouped_mm forward with expert bias."""
        experts = self._init_experts(torch_mm_config_with_bias, torch_mm_backend, device)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config_with_bias.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(
            num_tokens, torch_mm_config_with_bias.n_activated_experts, dtype=torch.bfloat16, device=device
        )
        indices = torch.randint(
            0,
            torch_mm_config_with_bias.n_routed_experts,
            (num_tokens, torch_mm_config_with_bias.n_activated_experts),
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_matches_loop_path(self, torch_mm_config, torch_mm_backend, device):
        """Test that torch_mm and loop paths produce similar outputs."""
        torch_mm_config.dtype = torch.float32

        experts_mm = self._init_experts(torch_mm_config, torch_mm_backend, device)
        experts_loop = GroupedExperts(torch_mm_config).to(device)
        experts_loop.expert_activation_grouped = self._unwrap_compiled(experts_loop.expert_activation_grouped)
        # Copy weights
        with torch.no_grad():
            experts_loop.gate_and_up_projs.copy_(experts_mm.gate_and_up_projs)
            experts_loop.down_projs.copy_(experts_mm.down_projs)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        out_mm = experts_mm(x, token_mask, weights, indices)
        out_loop = experts_loop(x, token_mask, weights, indices)

        torch.testing.assert_close(out_mm, out_loop, rtol=1e-3, atol=1e-3)

    def test_backward(self, torch_mm_config, torch_mm_backend, device):
        """Test backward pass completes and produces gradients."""
        torch_mm_config.dtype = torch.float32
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 8
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert experts.gate_and_up_projs.grad is not None
        assert experts.down_projs.grad is not None

    def test_zero_active_experts(self, torch_mm_config, torch_mm_backend, device):
        """Test grouped_mm path when no tokens route to any expert."""
        torch_mm_config.dtype = torch.float32
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 8
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.float32, device=device)
        # Route all tokens to non-existent experts
        indices = torch.full(
            (num_tokens, torch_mm_config.n_activated_experts),
            fill_value=torch_mm_config.n_routed_experts + 100,
            dtype=torch.long,
            device=device,
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Backward should still work (dummy computation for gradient flow)
        residual = x.mean(dim=-1, keepdim=True).expand_as(x)
        (output + residual).sum().backward()
        assert x.grad is not None

    def test_partial_token_mask(self, torch_mm_config, torch_mm_backend, device):
        """Test grouped_mm with partially masked tokens."""
        experts = self._init_experts(torch_mm_config, torch_mm_backend, device)

        num_tokens = 16
        x = torch.randn(num_tokens, torch_mm_config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        token_mask[: num_tokens // 2] = True
        weights = torch.rand(num_tokens, torch_mm_config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(
            0, torch_mm_config.n_routed_experts, (num_tokens, torch_mm_config.n_activated_experts), device=device
        )

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_relu2_activation(self, torch_mm_backend, device):
        """Test grouped_mm with ReLU² (non-gated) activation."""
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
        experts = self._init_experts(config, torch_mm_backend, device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_deepep_init_with_torch_mm(self, torch_mm_config, torch_mm_backend):
        """Test GroupedExpertsDeepEP initializes with torch_mm backend."""
        experts = GroupedExpertsDeepEP(torch_mm_config, backend=torch_mm_backend)
        assert experts.use_torch_mm is True

    def test_deepep_init_without_torch_mm(self, torch_mm_config):
        """Test GroupedExpertsDeepEP defaults to gmm without torch_mm backend."""
        experts = GroupedExpertsDeepEP(torch_mm_config)
        assert experts.use_torch_mm is False


class TestTorchMMExpertsFwd:
    """Test _torch_mm_experts_fwd helper function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_basic_forward(self, device):
        """Test _torch_mm_experts_fwd produces correct shape output."""
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


class TestGroupedExpertsConvergenceFixes:
    """Test fixes for GroupedExperts convergence with expert parallelism.

    These tests verify:
    1. expert_activation_grouped is always initialized (needed for restructured loop path)
    2. Loop path uses WeightedSwiGLUFunction (matching DeepEP compute pattern)
    3. Float32 scatter_add accumulation with correct output dtype
    4. Backward gradient flow through both loop and grouped_mm paths
    """

    @pytest.fixture
    def config(self):
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
    def torch_mm_backend(self):
        return BackendConfig(experts="torch_mm", dispatcher="torch")

    @staticmethod
    def _unwrap_compiled(fn):
        from functools import partial

        if isinstance(fn, partial):
            inner = TestGroupedExpertsConvergenceFixes._unwrap_compiled(fn.func)
            if inner is not fn.func:
                return partial(inner, *fn.args, **fn.keywords)
            return fn
        if hasattr(fn, "_torchdynamo_orig_callable"):
            return fn._torchdynamo_orig_callable
        return fn

    def _init_experts(self, config, backend=None, device=None):
        experts = GroupedExperts(config, backend=backend)
        if device:
            experts = experts.to(device)
        with torch.no_grad():
            experts.gate_and_up_projs.normal_(0, 0.02)
            experts.down_projs.normal_(0, 0.02)
            if experts.expert_bias:
                experts.gate_up_proj_bias.zero_()
                experts.down_proj_bias.zero_()
        experts.expert_activation_grouped = self._unwrap_compiled(experts.expert_activation_grouped)
        return experts

    # --- Test 1: expert_activation_grouped always initialized ---

    def test_expert_activation_grouped_always_present(self, config):
        """expert_activation_grouped must be available for both loop and grouped_mm paths."""
        experts_no_backend = GroupedExperts(config)
        assert hasattr(experts_no_backend, "expert_activation_grouped")
        assert callable(experts_no_backend.expert_activation_grouped)

        experts_with_backend = GroupedExperts(config, backend=BackendConfig(experts="torch_mm", dispatcher="torch"))
        assert hasattr(experts_with_backend, "expert_activation_grouped")
        assert callable(experts_with_backend.expert_activation_grouped)

    # --- Test 2: Output dtype matches input dtype (float32 accumulation cast back) ---

    def test_output_dtype_matches_input_bf16(self, config, device):
        """Output should be bf16 when input is bf16 (float32 accumulation cast back)."""
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.dtype == torch.bfloat16

    def test_output_dtype_matches_input_fp32(self, config, device):
        """Output should be float32 when input is float32."""
        config.dtype = torch.float32
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_output_dtype_grouped_mm_bf16(self, config, torch_mm_backend, device):
        """grouped_mm path output should be bf16 when input is bf16."""
        experts = self._init_experts(config, backend=torch_mm_backend, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.bfloat16, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.bfloat16, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.dtype == torch.bfloat16

    # --- Test 3: Loop path matches grouped_mm (restructured to use WeightedSwiGLUFunction) ---

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_loop_path_matches_grouped_mm_path(self, config, torch_mm_backend, device):
        """Loop path (restructured) should produce similar output to grouped_mm path."""
        config.dtype = torch.float32

        experts_mm = self._init_experts(config, backend=torch_mm_backend, device=device)
        experts_loop = self._init_experts(config, device=device)
        with torch.no_grad():
            experts_loop.gate_and_up_projs.copy_(experts_mm.gate_and_up_projs)
            experts_loop.down_projs.copy_(experts_mm.down_projs)

        num_tokens = 16
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        out_mm = experts_mm(x, token_mask, weights, indices)
        out_loop = experts_loop(x, token_mask, weights, indices)

        torch.testing.assert_close(out_mm, out_loop, rtol=1e-3, atol=1e-3)

    # --- Test 4: Backward produces correct gradients ---

    def test_loop_path_backward_all_params_have_grad(self, config, device):
        """Loop path backward should produce gradients for input and all expert params."""
        config.dtype = torch.float32
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        output.sum().backward()

        assert x.grad is not None, "Input x should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients should not be NaN"
        assert experts.gate_and_up_projs.grad is not None, "gate_and_up_projs should have gradients"
        assert experts.down_projs.grad is not None, "down_projs should have gradients"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for torch._grouped_mm")
    def test_loop_and_grouped_mm_backward_gradients_match(self, config, torch_mm_backend, device):
        """Loop and grouped_mm paths should produce similar gradients."""
        config.dtype = torch.float32

        experts_mm = self._init_experts(config, backend=torch_mm_backend, device=device)
        experts_loop = self._init_experts(config, device=device)
        with torch.no_grad():
            experts_loop.gate_and_up_projs.copy_(experts_mm.gate_and_up_projs)
            experts_loop.down_projs.copy_(experts_mm.down_projs)

        num_tokens = 16
        x_mm = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device, requires_grad=True)
        x_loop = x_mm.detach().clone().requires_grad_(True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        experts_mm(x_mm, token_mask, weights, indices).sum().backward()
        experts_loop(x_loop, token_mask, weights, indices).sum().backward()

        torch.testing.assert_close(x_mm.grad, x_loop.grad, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            experts_mm.gate_and_up_projs.grad, experts_loop.gate_and_up_projs.grad, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(experts_mm.down_projs.grad, experts_loop.down_projs.grad, rtol=1e-3, atol=1e-3)

    # --- Test 5: Loop path with bias ---

    def test_loop_path_with_bias_forward_and_backward(self, config, device):
        """Loop path should work correctly with expert bias."""
        config.dtype = torch.float32
        config.expert_bias = True
        experts = self._init_experts(config, device=device)

        num_tokens = 8
        x = torch.randn(num_tokens, config.dim, dtype=torch.float32, device=device, requires_grad=True)
        token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
        weights = torch.rand(num_tokens, config.n_activated_experts, dtype=torch.float32, device=device)
        indices = torch.randint(0, config.n_routed_experts, (num_tokens, config.n_activated_experts), device=device)

        output = experts(x, token_mask, weights, indices)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        output.sum().backward()
        assert x.grad is not None
        assert experts.gate_up_proj_bias.grad is not None, "gate_up_proj_bias should have gradients"
        assert experts.down_projs.grad is not None
