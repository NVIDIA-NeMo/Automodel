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

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from nemo_automodel.components.models.nemotron_v3.layers import (
    NemotronV3Attention,
    NemotronV3Block,
)
from nemo_automodel.components.moe.utils import BackendConfig
from nemo_automodel.components.moe.layers import MoEConfig

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class MockNemotronV3Config:
    """Mock configuration for NemotronV3 model."""

    def __init__(self, **overrides):
        # Attention configuration
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.head_dim = 64
        self.hidden_size = 512
        self.attention_bias = False
        self.attention_dropout = 0.0

        # MLP/MoE configuration
        self.intermediate_size = 1024
        self.mlp_bias = False
        self.mlp_hidden_act = "relu2"

        # Mamba configuration
        self.mamba_num_heads = 8
        self.mamba_head_dim = 64
        self.ssm_state_size = 16
        self.n_groups = 1
        self.chunk_size = 256
        self.conv_kernel = 4
        self.use_conv_bias = True
        self.mamba_hidden_act = "silu"
        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.time_step_floor = 1e-4
        self.use_bias = False

        # General configuration
        self.layer_norm_epsilon = 1e-5
        self.num_hidden_layers = 4
        self.vocab_size = 1000
        self.torch_dtype = "bfloat16"
        self.initializer_range = 0.02
        self.rescale_prenorm_residual = True
        self.residual_in_fp32 = False

        # Layer types for hybrid architecture
        self.layers_block_type = ["attention", "mlp", "moe", "mamba"]

        # MoE configuration
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.moe_intermediate_size = 256
        self.norm_topk_prob = False
        self.moe_shared_expert_intermediate_size = 256

        # Apply overrides
        for key, value in overrides.items():
            setattr(self, key, value)


class TestNemotronV3Attention:
    """Test NemotronV3Attention module."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    def test_attention_init(self, config):
        """Test attention initialization."""
        attn = NemotronV3Attention(config)

        assert attn.num_attention_heads == config.num_attention_heads
        assert attn.num_key_value_heads == config.num_key_value_heads
        assert attn.head_dim == config.head_dim
        assert attn.hidden_size == config.hidden_size

        # Check projections exist
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "o_proj")

    def test_attention_init_with_bias(self, config):
        """Test attention initialization with bias enabled."""
        config.attention_bias = True
        attn = NemotronV3Attention(config)

        assert attn.q_proj.bias is not None
        assert attn.k_proj.bias is not None
        assert attn.v_proj.bias is not None
        assert attn.o_proj.bias is not None

    def test_attention_forward_shape(self, config):
        """Test attention forward pass produces correct shapes."""
        attn = NemotronV3Attention(config)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_attention_forward_with_mask(self, config):
        """Test attention forward pass with attention mask."""
        attn = NemotronV3Attention(config)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        # Create 4D causal mask
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        attention_mask = attention_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), float("-inf")
        )

        output = attn(hidden_states, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_attention_gqa_with_different_kv_heads(self):
        """Test GQA with different number of key-value heads."""
        config = MockNemotronV3Config(
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=32,
            hidden_size=512,
        )
        attn = NemotronV3Attention(config)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)
        # Verify projection dimensions
        assert attn.q_proj.out_features == config.num_attention_heads * config.head_dim
        assert attn.k_proj.out_features == config.num_key_value_heads * config.head_dim
        assert attn.v_proj.out_features == config.num_key_value_heads * config.head_dim

    def test_attention_init_weights(self, config):
        """Test attention weight initialization."""
        config.attention_bias = True
        attn = NemotronV3Attention(config)

        device = torch.device("cpu")
        attn.init_weights(
            num_hidden_layers=config.num_hidden_layers,
            rescale_prenorm_residual=True,
            buffer_device=device,
        )

        # Check biases are zeroed
        assert torch.allclose(attn.q_proj.bias, torch.zeros_like(attn.q_proj.bias))
        assert torch.allclose(attn.k_proj.bias, torch.zeros_like(attn.k_proj.bias))

    def test_attention_forward_single_token(self, config):
        """Test attention with single token (seqlen=1)."""
        attn = NemotronV3Attention(config)

        batch_size, seq_len = 2, 1
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestNemotronV3Block:
    """Test NemotronV3Block module."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            enable_deepep=False,
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    @pytest.fixture
    def moe_config(self, config):
        return MoEConfig(
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=1,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=config.routed_scaling_factor,
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=config.mlp_bias,
            expert_activation="relu2",
            dtype=torch.bfloat16,
            shared_expert_gate=False,
            shared_expert_inter_dim=config.moe_shared_expert_intermediate_size,
            shared_expert_activation="relu2",
        )

    def test_block_init_attention(self, config, backend):
        """Test block initialization with attention layer type."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.block_type == "attention"
        assert isinstance(block.mixer, NemotronV3Attention)

    def test_block_init_mlp(self, config, backend):
        """Test block initialization with MLP layer type."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.block_type == "mlp"
        # MLP should use relu2 activation by default for NemotronV3
        assert hasattr(block.mixer, "up_proj")
        assert hasattr(block.mixer, "down_proj")

    def test_block_init_moe(self, config, backend, moe_config):
        """Test block initialization with MoE layer type."""
        config.layers_block_type = ["moe"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=moe_config, backend=backend)

        assert block.block_type == "moe"
        assert hasattr(block.mixer, "gate")
        assert hasattr(block.mixer, "experts")

    def test_block_init_invalid_type(self, config, backend):
        """Test block initialization with invalid layer type raises error."""
        config.layers_block_type = ["invalid"]

        with pytest.raises(ValueError, match="Invalid block_type"):
            NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

    def test_block_forward_attention(self, config, backend):
        """Test block forward pass with attention layer."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_forward_mlp(self, config, backend):
        """Test block forward pass with MLP layer."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_forward_moe(self, config, backend, moe_config):
        """Test block forward pass with MoE layer."""
        config.layers_block_type = ["moe"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=moe_config, backend=backend)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = block(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_residual_connection(self, config, backend):
        """Test that block applies residual connection."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
        hidden_states_clone = hidden_states.clone()

        output = block(hidden_states)

        # Output should differ from input due to MLP transformation
        assert not torch.allclose(output, hidden_states_clone)

    def test_block_residual_fp32(self, config, backend):
        """Test block with fp32 residual option."""
        config.layers_block_type = ["mlp"]
        config.residual_in_fp32 = True
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.residual_in_fp32 is True

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)

        output = block(hidden_states)

        # Should still produce correct shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_block_mlp_property(self, config, backend, moe_config):
        """Test mlp property returns mixer for MoE blocks."""
        config.layers_block_type = ["moe"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=moe_config, backend=backend)

        assert block.mlp is not None
        assert block.mlp is block.mixer

    def test_block_mlp_property_non_moe(self, config, backend):
        """Test mlp property returns None for non-MoE blocks."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        assert block.mlp is None

    def test_block_init_weights_attention(self, config, backend):
        """Test weight initialization for attention block."""
        config.layers_block_type = ["attention"]
        config.attention_bias = True
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        device = torch.device("cpu")
        block.init_weights(buffer_device=device)

        # Verify biases are zeroed
        assert torch.allclose(
            block.mixer.q_proj.bias, torch.zeros_like(block.mixer.q_proj.bias)
        )

    def test_block_init_weights_mlp(self, config, backend):
        """Test weight initialization for MLP block."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        device = torch.device("cpu")
        block.init_weights(buffer_device=device)

        # Weights should be initialized (not all zeros)
        assert not torch.allclose(
            block.mixer.up_proj.weight, torch.zeros_like(block.mixer.up_proj.weight)
        )

    def test_block_uses_relu2_for_mlp(self, config, backend):
        """Test that MLP block uses relu2 activation by default."""
        config.layers_block_type = ["mlp"]
        config.mlp_hidden_act = "relu2"
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        # MLP should not have gate_proj (non-gated activation)
        assert block.mixer.gate_proj is None
        assert block.mixer.up_proj is not None
        assert block.mixer.down_proj is not None


class TestNemotronV3MambaRMSNormGated:
    """Test NemotronV3MambaRMSNormGated module."""

    def test_gated_rmsnorm_init(self):
        """Test gated RMSNorm initialization."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3MambaRMSNormGated

        hidden_size = 512
        group_size = 128
        norm = NemotronV3MambaRMSNormGated(hidden_size, group_size, eps=1e-5)

        assert norm.weight.shape == (hidden_size,)
        assert norm.variance_epsilon == 1e-5
        assert norm.group_size == group_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Mamba Triton kernels")
    def test_gated_rmsnorm_forward_requires_cuda(self):
        """Test that forward requires CUDA for Triton kernels."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3MambaRMSNormGated

        hidden_size = 512
        group_size = 128
        norm = NemotronV3MambaRMSNormGated(hidden_size, group_size).cuda()

        # Forward requires CUDA tensors for Triton kernels
        hidden_states = torch.randn(2, 8, hidden_size, device="cuda")
        output = norm(hidden_states)
        assert output.shape == hidden_states.shape


class TestNemotronV3Mamba2Mixer:
    """Test NemotronV3Mamba2Mixer module initialization."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    def test_mamba2_mixer_init(self, config):
        """Test Mamba2Mixer initialization."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        assert mixer.layer_idx == 0
        assert mixer.hidden_size == config.hidden_size
        assert mixer.num_heads == config.mamba_num_heads
        assert mixer.head_dim == config.mamba_head_dim
        assert mixer.ssm_state_size == config.ssm_state_size
        assert mixer.n_groups == config.n_groups
        assert mixer.chunk_size == config.chunk_size
        assert mixer.conv_kernel_size == config.conv_kernel

    def test_mamba2_mixer_derived_dimensions(self, config):
        """Test Mamba2Mixer derived dimensions."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        expected_intermediate = config.mamba_num_heads * config.mamba_head_dim
        expected_conv_dim = expected_intermediate + 2 * config.n_groups * config.ssm_state_size

        assert mixer.intermediate_size == expected_intermediate
        assert mixer.conv_dim == expected_conv_dim

    def test_mamba2_mixer_projection_sizes(self, config):
        """Test Mamba2Mixer projection dimensions."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        expected_proj_size = mixer.intermediate_size + mixer.conv_dim + mixer.num_heads
        assert mixer.in_proj.in_features == config.hidden_size
        assert mixer.in_proj.out_features == expected_proj_size

        assert mixer.out_proj.in_features == mixer.intermediate_size
        assert mixer.out_proj.out_features == config.hidden_size

    def test_mamba2_mixer_ssm_parameters(self, config):
        """Test Mamba2Mixer SSM parameters."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        assert mixer.dt_bias.shape == (config.mamba_num_heads,)
        assert mixer.A_log.shape == (config.mamba_num_heads,)
        assert mixer.D.shape == (config.mamba_num_heads,)

        # Check no_weight_decay attributes
        assert getattr(mixer.A_log, "_no_weight_decay", False)
        assert getattr(mixer.D, "_no_weight_decay", False)

    def test_mamba2_mixer_conv1d(self, config):
        """Test Mamba2Mixer conv1d layer."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        assert mixer.conv1d.in_channels == mixer.conv_dim
        assert mixer.conv1d.out_channels == mixer.conv_dim
        assert mixer.conv1d.kernel_size[0] == config.conv_kernel
        assert mixer.conv1d.groups == mixer.conv_dim
