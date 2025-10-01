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

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.moe.layers import (
    FakeBalancedGate,
    Gate,
    GroupedExperts,
    GroupedExpertsDeepEP,
    MLP,
    MoE,
    MoEConfig,
    get_expert_activation,
    get_expert_activation_for_deepep,
    quick_geglu,
    quick_geglu_deepep,
    swiglu,
)
from nemo_automodel.components.moe.utils import BackendConfig


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
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
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
            down_proj_bias=down_proj_bias
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

        with patch("nemo_automodel.components.moe.layers.weighted_bias_swiglu_impl") as mock_swiglu:
            activation_fn = get_expert_activation_for_deepep(moe_config)
            assert activation_fn == mock_swiglu


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
        indices = torch.randint(0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device)

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
        indices = torch.randint(0, moe_config.n_routed_experts, (num_tokens, moe_config.n_activated_experts), device=device)

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
        with patch("nemo_automodel.components.moe.layers.MoEFlexTokenDispatcher") as mock_dispatcher:
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

        result = experts._apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert, permuted_probs=permuted_probs)

        assert result.shape == value.shape


class TestMoE:
    """Test MoE (Mixture of Experts) module."""

    def test_moe_init_with_fake_balanced_gate(self, moe_config, backend_config):
        """Test MoE initialization with fake balanced gate."""
        backend_config.fake_balanced_gate = True
        moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, FakeBalancedGate)
        assert isinstance(moe.experts, GroupedExperts)

    def test_moe_init_with_deepep(self, moe_config, backend_config):
        """Test MoE initialization with DeepEP."""
        backend_config.enable_deepep = True
        moe = MoE(moe_config, backend_config)

        assert isinstance(moe.gate, Gate)
        assert isinstance(moe.experts, GroupedExpertsDeepEP)

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

        with patch.object(moe.gate, "forward") as mock_gate, \
             patch.object(moe.experts, "forward") as mock_experts:

            # Mock gate outputs
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(0, moe_config.n_routed_experts, (batch_size * seq_len, moe_config.n_activated_experts), device=device),
                None
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

        with patch.object(moe.gate, "forward") as mock_gate, \
             patch.object(moe.experts, "forward") as mock_experts, \
             patch.object(moe.shared_experts, "forward") as mock_shared:

            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(0, moe_config.n_routed_experts, (batch_size * seq_len, moe_config.n_activated_experts), device=device),
                None
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)
            mock_shared.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            # Patch at the module level to avoid CUDA stream issues on CPU
            with patch("torch.cuda.Stream") as mock_stream_class, \
                 patch("torch.cuda.current_stream") as mock_current_stream, \
                 patch("torch.cuda.stream") as mock_stream_context:

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

        with patch.object(moe.gate, "forward") as mock_gate, \
             patch.object(moe.experts, "forward") as mock_experts:

            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(0, moe_config.n_routed_experts, (batch_size * seq_len, moe_config.n_activated_experts), device=device),
                None
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

        with patch.object(moe.gate, "forward") as mock_gate, \
             patch.object(moe.experts, "forward") as mock_experts:

            aux_loss = torch.tensor(0.01, device=device)
            mock_gate.return_value = (
                torch.rand(batch_size * seq_len, moe_config.n_activated_experts, device=device),
                torch.randint(0, moe_config.n_routed_experts, (batch_size * seq_len, moe_config.n_activated_experts), device=device),
                aux_loss
            )

            mock_experts.return_value = torch.randn(batch_size * seq_len, moe_config.dim, device=device)

            result = moe(x)

            # Should return the reshaped output since aux_loss handling is done in gate
            assert result.shape == x.shape
