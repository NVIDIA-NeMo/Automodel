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


import copy
import math

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3 import layers as nemotron_layers
from nemo_automodel.components.models.nemotron_v3.layers import (
    NemotronV3Attention,
    NemotronV3BackendConfig,
    NemotronV3Block,
    NemotronV3Mamba2Mixer,
)
from nemo_automodel.components.models.nemotron_v3.mamba_scan import split_conv1d_scan
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM, NemotronV3Model
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.parallelizer import _get_moe_module, apply_ac
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mamba_out_projection import (
    _install_cpu_scan,
    _pre_projection,
)
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_model import MockNemotronV3Config as ModelMockNemotronV3Config
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config as MtpMockNemotronV3Config

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")

try:
    import mamba_ssm  # noqa: F401

    _has_mamba_ssm = True
except ImportError:
    _has_mamba_ssm = False

skip_if_no_mamba = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_mamba_ssm, reason="CUDA and mamba_ssm required for Mamba Triton kernels"
)


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

    @skip_if_no_gpu
    def test_attention_forward_shape(self, config):
        """Test attention forward pass produces correct shapes."""
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_forward_with_mask(self, config):
        """Test attention forward pass with attention mask."""
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        # 2D padding mask (1=valid, 0=pad) — TE handles causality internally
        attention_mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.long)

        output = attn(hidden_states, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_gqa_with_different_kv_heads(self):
        """Test GQA with different number of key-value heads."""
        config = MockNemotronV3Config(
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=32,
            hidden_size=512,
        )
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

        output = attn(hidden_states)

        assert output.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_init_weights(self, config):
        """Test attention weight initialization."""
        config.attention_bias = True
        attn = NemotronV3Attention(config).cuda()

        device = torch.device("cuda")
        attn.init_weights(
            num_hidden_layers=config.num_hidden_layers,
            rescale_prenorm_residual=True,
            buffer_device=device,
        )

        # Check biases are zeroed
        assert torch.allclose(attn.q_proj.bias, torch.zeros_like(attn.q_proj.bias))
        assert torch.allclose(attn.k_proj.bias, torch.zeros_like(attn.k_proj.bias))

    @pytest.mark.parametrize("rescale", [False, True])
    def test_attention_initializes_materialized_projection_weights(self, config, rescale: bool) -> None:
        """Meta construction must not leave attention projections as allocator contents."""
        config.torch_dtype = "float32"
        config.attention_bias = True
        with torch.device("meta"):
            attn = NemotronV3Attention(config, backend=BackendConfig(linear="torch", attn="sdpa"))
        attn.to_empty(device=torch.device("cpu"))
        with torch.no_grad():
            for parameter in attn.parameters():
                parameter.fill_(float("nan"))
        attn.init_weights(
            num_hidden_layers=config.num_hidden_layers,
            rescale_prenorm_residual=rescale,
            buffer_device=torch.device("cpu"),
        )
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            assert torch.isfinite(projection.weight).all()
            assert torch.count_nonzero(projection.bias) == 0
            expected_std = config.initializer_range
            if projection is attn.o_proj and rescale:
                expected_std /= config.num_hidden_layers**0.5
            assert projection.weight.std().item() == pytest.approx(expected_std, rel=0.05)

    @skip_if_no_gpu
    def test_attention_forward_single_token(self, config):
        """Test attention with single token (seqlen=1)."""
        attn = NemotronV3Attention(config).cuda()

        batch_size, seq_len = 2, 1
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

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
            dispatcher="torch",
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
            train_gate=True,
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

    @skip_if_no_gpu
    def test_block_forward_attention(self, config, backend):
        """Test block forward pass with attention layer."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend).cuda()

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)

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

    @skip_if_no_gpu
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
        assert torch.allclose(block.mixer.q_proj.bias, torch.zeros_like(block.mixer.q_proj.bias))

    def test_block_init_weights_mlp(self, config, backend):
        """Test weight initialization for MLP block."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)

        device = torch.device("cpu")
        block.init_weights(buffer_device=device)

        # Weights should be initialized (not all zeros)
        assert not torch.allclose(block.mixer.up_proj.weight, torch.zeros_like(block.mixer.up_proj.weight))

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

    @skip_if_no_mamba
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

    def test_full_tensor_if_dtensor_clones_gathered_value(self, monkeypatch):
        """Gathered fp32 holder outputs must not alias storage FSDP may reshard."""
        from nemo_automodel.components.models.nemotron_v3 import layers as layers_mod

        full = torch.arange(4, dtype=torch.float32, requires_grad=True)

        class FakeDTensor:
            def full_tensor(self):
                return full

        monkeypatch.setattr(layers_mod, "DTensor", FakeDTensor)

        gathered = layers_mod._full_tensor_if_dtensor(FakeDTensor())

        assert torch.equal(gathered, full)
        assert gathered.data_ptr() != full.data_ptr()
        gathered.sum().backward()
        assert torch.equal(full.grad, torch.ones_like(full))

    def test_full_tensor_if_dtensor_clones_regular_tensor(self):
        """FSDP can reshard a regular-looking tensor returned from an fp32 holder."""
        from nemo_automodel.components.models.nemotron_v3 import layers as layers_mod

        tensor = torch.arange(4, dtype=torch.float32, requires_grad=True)

        gathered = layers_mod._full_tensor_if_dtensor(tensor)

        assert torch.equal(gathered, tensor)
        assert gathered.data_ptr() != tensor.data_ptr()
        gathered.sum().backward()
        assert torch.equal(tensor.grad, torch.ones_like(tensor))

    def test_get_fp32_ssm_params_passes_reference(self, config):
        """FSDP2 root pre-forward expects a non-empty args tuple for the holder."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)
        reference = torch.randn(1, 2, mixer.in_proj.out_features)
        seen = {}

        def fake_forward(ref=None):
            seen["reference"] = ref
            return mixer.A_log, mixer.dt_bias, mixer.D

        mixer._fp32_params.forward = fake_forward

        mixer._get_fp32_ssm_params(reference)

        assert seen["reference"] is reference

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

        params = dict(mixer.named_parameters())
        assert "_fp32_params.A_log" in params
        assert "_fp32_params.dt_bias" in params
        assert "_fp32_params.D" in params
        assert "A_log" not in mixer._parameters
        assert "dt_bias" not in mixer._parameters
        assert "D" not in mixer._parameters

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


class TestBackwardCompatibility:
    """Verify that attention and block layers still work without cache args."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    @skip_if_no_gpu
    def test_attention_no_cache_args(self, config):
        """Verify attn(hidden) still works without cache args."""
        attn = NemotronV3Attention(config).cuda()
        hidden = torch.randn(2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out = attn(hidden)
        assert out.shape == (2, 8, config.hidden_size)

    @skip_if_no_gpu
    def test_attention_mask_only(self, config):
        """Verify attn(hidden, attention_mask=...) still works without cache args."""
        attn = NemotronV3Attention(config).cuda()
        batch_size, seq_len = 2, 8
        hidden = torch.randn(batch_size, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # 2D padding mask (1=valid, 0=pad) — TE handles causality internally
        mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.long)
        out = attn(hidden, attention_mask=mask)
        assert out.shape == (batch_size, seq_len, config.hidden_size)

    @skip_if_no_gpu
    def test_block_attention_no_cache_args(self, config, backend):
        """Verify block(hidden) still works for attention block without cache args."""
        config.layers_block_type = ["attention"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend).cuda()
        hidden = torch.randn(2, 8, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        out = block(hidden)
        assert out.shape == (2, 8, config.hidden_size)

    def test_block_mlp_no_cache_args(self, config, backend):
        """Verify block(hidden) still works for mlp block without cache args."""
        config.layers_block_type = ["mlp"]
        block = NemotronV3Block(config, layer_idx=0, moe_config=None, backend=backend)
        block = block.to(torch.bfloat16)
        hidden = torch.randn(2, 8, config.hidden_size, dtype=torch.bfloat16)
        out = block(hidden)
        assert out.shape == (2, 8, config.hidden_size)


def _mamba_init_config() -> MockNemotronV3Config:
    """A small mixer: 8 heads of 16 channels in 2 groups."""
    return MockNemotronV3Config(hidden_size=64, mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=8, n_groups=2)


def _zeroed_mamba_mixer() -> NemotronV3Mamba2Mixer:
    """A mixer whose parameters hold what a fresh uninitialized allocation hands out: zeros."""
    mixer = NemotronV3Mamba2Mixer(_mamba_init_config(), layer_idx=0)
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.zero_()
    return mixer


@pytest.mark.cpu_only
def test_mamba_mixer_init_weights_initializes_every_parameter_of_an_uninitialized_mixer():
    """init_weights must not rely on the constructors' default init: a mixer built without it starts at zero."""
    num_hidden_layers = 4
    mixer = _zeroed_mamba_mixer()

    with torch.no_grad():
        mixer.init_weights(num_hidden_layers=num_hidden_layers, buffer_device=torch.device("cpu"))

    config = _mamba_init_config()
    intermediate = config.mamba_num_heads * config.mamba_head_dim
    in_bound = 1.0 / math.sqrt(config.hidden_size)
    out_bound = 1.0 / math.sqrt(intermediate) / math.sqrt(num_hidden_layers)
    conv_bound = 1.0 / math.sqrt(config.conv_kernel)
    for name, weight, bound in (
        ("in_proj.weight", mixer.in_proj.weight, in_bound),
        ("out_proj.weight", mixer.out_proj.weight, out_bound),
        ("conv1d.weight", mixer.conv1d.weight, conv_bound),
        ("conv1d.bias", mixer.conv1d.bias, conv_bound),
    ):
        # A uniform(-bound, bound) sample: inside the bound, with standard deviation bound / sqrt(3).
        assert float(weight.detach().abs().max()) <= bound, name
        assert float(weight.detach().std()) == pytest.approx(bound / math.sqrt(3), rel=0.2), name
    torch.testing.assert_close(mixer.norm.weight.detach(), torch.ones(intermediate))
    torch.testing.assert_close(mixer.D.detach(), torch.ones(config.mamba_num_heads))
    torch.testing.assert_close(mixer.A_log.detach(), torch.log(torch.arange(1, config.mamba_num_heads + 1).float()))
    step_size = torch.nn.functional.softplus(mixer.dt_bias.detach())
    assert float(step_size.min()) >= config.time_step_floor
    assert float(step_size.max()) <= config.time_step_max * 1.001


def _sharded_ssm_parameters_worker(rank: int, rendezvous: str) -> None:
    """One of two Gloo ranks: initialize a mixer whose A_log and dt_bias are sharded over the ranks."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        # What one process draws from this random state.
        reference = _zeroed_mamba_mixer()
        torch.manual_seed(20261123)
        with torch.no_grad():
            reference.init_weights(num_hidden_layers=4, buffer_device=torch.device("cpu"))

        mixer = _zeroed_mamba_mixer()
        holder = mixer._fp32_params
        holder.A_log = torch.nn.Parameter(distribute_tensor(holder.A_log.detach(), mesh, [Shard(0)]))
        holder.dt_bias = torch.nn.Parameter(distribute_tensor(holder.dt_bias.detach(), mesh, [Shard(0)]))
        # Both ranks start from one random state, as they do in training once a sharded tensor has been drawn.
        torch.manual_seed(20261123)
        with torch.no_grad():
            mixer.init_weights(num_hidden_layers=4, buffer_device=torch.device("cpu"))
        expected = torch.log(torch.arange(1, _mamba_init_config().mamba_num_heads + 1).float())
        torch.testing.assert_close(mixer.A_log.detach().full_tensor(), expected)
        torch.testing.assert_close(mixer.A_log.detach().to_local(), expected.chunk(2)[rank])
        # dt_bias is random per head: each rank holds its half of the single-process draw, not the same half.
        torch.testing.assert_close(mixer.dt_bias.detach().to_local(), reference.dt_bias.detach().chunk(2)[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
def test_mamba_mixer_init_weights_gives_each_rank_its_slice_of_sharded_ssm_parameters(tmp_path):
    """Per-head parameters follow the global head index: a sharded one must not restart at head 0 on every rank."""
    import torch.multiprocessing as mp

    mp.spawn(_sharded_ssm_parameters_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


class TestMambaMixerSeqIdxConstruction:
    """Regression tests for the seq_idx construction in NemotronV3Mamba2Mixer.

    The mamba kernel asserts ``seq_idx.shape == (batch_size, seqlen)``. When
    the model receives a (B, S, H) BSHD batch with a globally-cumulated
    ``cu_seqlens`` (derived in ``model.py`` from a 2D attention_mask), the
    pre-fix mixer produced ``seq_idx`` of shape ``(1, S)`` and crashed. After
    the fix the mixer must produce a per-row ``seq_idx`` of shape ``(B, S)``.
    """

    @pytest.fixture
    def config(self):
        # Small dimensions so CPU init is fast.
        return MockNemotronV3Config(
            hidden_size=64,
            mamba_num_heads=4,
            mamba_head_dim=16,
            n_groups=1,
            ssm_state_size=8,
            chunk_size=8,
            conv_kernel=4,
        )

    @staticmethod
    def _patch_mamba_kernel(monkeypatch, capture):
        """Install a fake ``mamba_split_conv1d_scan_combined`` that records
        the ``seq_idx`` kwarg and returns a correctly-shaped output tensor."""
        import sys
        import types

        def _fake_kernel(projected_states, *args, **kwargs):
            capture["seq_idx"] = kwargs.get("seq_idx", None)
            # Out shape matches the projected gate path: (B, S, intermediate_size)
            # after the outproj_weight matmul → (B, S, hidden_size).
            outproj_weight = kwargs["outproj_weight"]
            B, S, _ = projected_states.shape
            return projected_states.new_zeros((B, S, outproj_weight.shape[0]))

        # Stub the import chain so the deferred ``from mamba_ssm.ops.triton.
        # ssd_combined import mamba_split_conv1d_scan_combined`` resolves to
        # our fake without requiring the real package.
        for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        fake_ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
        fake_ssd.mamba_split_conv1d_scan_combined = _fake_kernel
        monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", fake_ssd)

    def test_bshd_b_gt_1_with_cu_seqlens_yields_per_row_seq_idx(self, monkeypatch, config):
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        B, S = 2, 16
        hidden_states = torch.randn(B, S, config.hidden_size)
        # cu_seqlens as model.py derives it from a 2D attention_mask: a global
        # cumsum across the batch (one boundary per row).
        cu_seqlens = torch.tensor([0, S, 2 * S], dtype=torch.int32)

        capture: dict = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        mixer.forward(hidden_states, cu_seqlens=cu_seqlens)

        seq_idx = capture["seq_idx"]
        assert seq_idx is not None, "mamba kernel did not receive a seq_idx"
        assert seq_idx.shape == (B, S), f"expected (B, S)=({B}, {S}), got {tuple(seq_idx.shape)}"
        assert seq_idx.dtype == torch.int32

    def test_bshd_b_eq_1_with_cu_seqlens_keeps_flat_construction(self, monkeypatch, config):
        """When B == 1 (THD flat layout), the legacy searchsorted construction
        still applies and should yield shape (1, S)."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        B, S = 1, 16
        hidden_states = torch.randn(B, S, config.hidden_size)
        # Two intra-row sub-sequences: boundaries at positions 8 and 16.
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

        capture: dict = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        mixer.forward(hidden_states, cu_seqlens=cu_seqlens)

        seq_idx = capture["seq_idx"]
        assert seq_idx is not None
        assert seq_idx.shape == (1, S)
        # right=True so boundary positions map to the new sub-seq.
        assert seq_idx[0, :8].tolist() == [0] * 8
        assert seq_idx[0, 8:].tolist() == [1] * 8

    def test_cp_bshd_uses_sequence_transport_while_preserving_packed_seq_idx(self, monkeypatch, config):
        """BSHD packing metadata must not select the CP helper's 2D THD path."""
        import sys
        import types

        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)
        calls: dict[str, object] = {}

        class FakeCP:
            cp_size = 2
            n_groups_local = mixer.n_groups

            def pre_conv_ssm(self, tensor, *, cu_seqlens=None):
                calls["pre_cu_seqlens"] = cu_seqlens
                calls["pre_shape"] = tensor.shape
                return tensor

            def post_conv_ssm(self, tensor, *, cu_seqlens=None):
                calls["post_cu_seqlens"] = cu_seqlens
                return tensor

            @staticmethod
            def get_A_log(tensor):
                return tensor

            @staticmethod
            def get_dt_bias(tensor):
                return tensor

            @staticmethod
            def get_D(tensor):
                return tensor

            def get_conv1d_weight(self):
                return mixer.conv1d.weight.squeeze(1)

            def get_conv1d_bias(self):
                return mixer.conv1d.bias

        def fake_kernel(projected_states, *args, **kwargs):
            del args
            calls["seq_idx"] = kwargs["seq_idx"]
            return projected_states[..., : mixer.intermediate_size]

        for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            if name not in sys.modules:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        fake_ssd = types.ModuleType("mamba_ssm.ops.triton.ssd_combined")
        fake_ssd.mamba_split_conv1d_scan_combined = fake_kernel
        monkeypatch.setitem(sys.modules, "mamba_ssm.ops.triton.ssd_combined", fake_ssd)

        class GateIdentity(torch.nn.Module):
            def forward(self, tensor, gate=None):
                del gate
                return tensor

        mixer.norm = GateIdentity()
        mixer.cp = FakeCP()
        hidden_states = torch.randn(1, 8, config.hidden_size)
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)

        output = mixer(hidden_states, cu_seqlens=cu_seqlens)

        assert output.shape == hidden_states.shape
        assert calls["pre_shape"] == mixer.in_proj(hidden_states).shape
        assert calls["pre_cu_seqlens"] is None
        assert calls["post_cu_seqlens"] is None
        # The local BSHD shard has 8 positions, while CP=2 transport presents
        # 16 positions to the kernel. cu_seqlens describes 8 real packed tokens;
        # searchsorted assigns the trailing 8 CP-padding positions the sentinel
        # sequence ID immediately after the two real documents.
        assert calls["seq_idx"].tolist() == [[0] * 4 + [1] * 4 + [2] * 8]

    def test_bshd_b_gt_1_passes_through_upstream_seq_idx(self, monkeypatch, config):
        """If seq_idx is supplied upstream (e.g. via _packed_seq_ids), the
        construction path must NOT override it — neat-packing semantics."""
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer

        mixer = NemotronV3Mamba2Mixer(config, layer_idx=0)

        B, S = 2, 16
        hidden_states = torch.randn(B, S, config.hidden_size)
        # Pretend the wrapper has already set seq_idx from _packed_seq_ids.
        upstream = torch.tensor(
            [[0] * 8 + [1] * 8, [0] * 6 + [1] * 5 + [2] * 5],
            dtype=torch.int32,
        )

        capture: dict = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        mixer.forward(hidden_states, cu_seqlens=torch.tensor([0, 16, 32], dtype=torch.int32), seq_idx=upstream)

        seq_idx = capture["seq_idx"]
        assert seq_idx is upstream, "upstream seq_idx must not be replaced by the construction path"

    def test_projected_states_too_large_for_the_scan_kernels_are_rejected(self, monkeypatch, config):
        """The scan kernels address the projected states with 32-bit offsets; a larger tensor must not reach them."""
        from nemo_automodel.components.models.nemotron_v3 import layers

        mixer = layers.NemotronV3Mamba2Mixer(config, layer_idx=0)
        capture: dict[str, torch.Tensor | None] = {}
        self._patch_mamba_kernel(monkeypatch, capture)
        projection_size = mixer.in_proj.out_features
        # 2 x 16 tokens of the projection are the limit here; a third sequence is one too many.
        monkeypatch.setattr(layers, "_MAMBA_SCAN_MAX_ELEMENTS", 2 * 16 * projection_size)

        mixer.forward(torch.randn(2, 16, config.hidden_size))
        assert "seq_idx" in capture
        capture.clear()
        message = f"Mamba scan received 3 x 16 x {projection_size} = {3 * 16 * projection_size} projected elements"
        with pytest.raises(ValueError, match=message):
            mixer.forward(torch.randn(3, 16, config.hidden_size))
        assert capture == {}


def test_mamba_scan_size_limit_is_the_largest_32_bit_offset():
    from nemo_automodel.components.models.nemotron_v3.layers import _check_mamba_scan_size

    # A 4096 -> 18560 projection of 8192-token sequences: 14 sequences are 2,128,609,280 elements, 15 are
    # 2,280,652,800. Meta tensors carry the shape without the memory.
    assert 14 * 8192 * 18560 < 2**31 < 15 * 8192 * 18560
    _check_mamba_scan_size(torch.empty(14, 8192, 18560, device="meta"))
    with pytest.raises(
        ValueError, match=r"15 x 8192 x 18560 = 2280652800 projected elements, more than the 2147483647"
    ):
        _check_mamba_scan_size(torch.empty(15, 8192, 18560, device="meta"))
    # Exactly 2**31 - 1 elements pass, 2**31 do not.
    _check_mamba_scan_size(torch.empty(1, 2**31 - 1, 1, device="meta"))
    with pytest.raises(ValueError, match="2147483648 projected elements"):
        _check_mamba_scan_size(torch.empty(1, 2**31, 1, device="meta"))


# ``NemotronV3BackendConfig.mamba_linear``: the Mamba projections as ``torch.nn.Linear`` or Transformer Engine linears.
#
# The comparison of an output projection applied after the scan kernel with the one applied inside it, on the
# real kernel, is in ``tests/functional_tests/kernel_parity/test_nemotron_v3_mamba_scan.py``.


def _mamba_linear_backend(mamba_linear: str) -> NemotronV3BackendConfig:
    return NemotronV3BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        mamba_linear=mamba_linear,
    )


@pytest.mark.cpu_only
def test_default_backend_keeps_torch_projections():
    # The shared config has no Mamba field; a mixer built with it uses the defaults of the model's config.
    shared = BackendConfig(attn="sdpa", linear="torch", experts="torch", dispatcher="torch")
    assert _backend().mamba_linear == "torch"
    for mixer in (
        NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0),
        NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0, backend=shared),
        NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0, backend=_mamba_linear_backend("torch")),
    ):
        assert type(mixer.in_proj) is nn.Linear
        assert type(mixer.out_proj) is nn.Linear


@pytest.mark.cpu_only
@pytest.mark.parametrize("with_backend", [False, True])
def test_torch_projection_construction_draws_the_same_initial_values(with_backend):
    """The default path must not change the initialization stream of existing recipes.

    The expectation is built here from plain PyTorch modules, in the order the mixer has always constructed
    them: input projection, depthwise convolution, output projection. The SSM parameters and the norm weight
    draw nothing.
    """
    config = MtpMockNemotronV3Config()
    intermediate = config.mamba_num_heads * config.mamba_head_dim
    conv_dim = intermediate + 2 * config.n_groups * config.ssm_state_size
    torch.manual_seed(7)
    in_proj = nn.Linear(config.hidden_size, intermediate + conv_dim + config.mamba_num_heads, bias=config.use_bias)
    conv1d = nn.Conv1d(
        conv_dim,
        conv_dim,
        kernel_size=config.conv_kernel,
        groups=conv_dim,
        padding=config.conv_kernel - 1,
        bias=config.use_conv_bias,
    )
    out_proj = nn.Linear(intermediate, config.hidden_size, bias=config.use_bias)
    expected = {
        **{f"in_proj.{name}": value for name, value in in_proj.named_parameters()},
        **{f"conv1d.{name}": value for name, value in conv1d.named_parameters()},
        **{f"out_proj.{name}": value for name, value in out_proj.named_parameters()},
    }

    torch.manual_seed(7)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_mamba_linear_backend("torch") if with_backend else None)
    actual = dict(mixer.named_parameters())
    assert set(expected) == {name for name in actual if not name.startswith(("_fp32_params.", "norm."))}
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)
    # Nothing else was drawn: the next random number is the one that follows the three modules.
    after_mixer = torch.rand(())
    torch.manual_seed(7)
    nn.Linear(config.hidden_size, intermediate + conv_dim + config.mamba_num_heads, bias=config.use_bias)
    nn.Conv1d(conv_dim, conv_dim, kernel_size=config.conv_kernel, groups=conv_dim, bias=config.use_conv_bias)
    nn.Linear(intermediate, config.hidden_size, bias=config.use_bias)
    assert torch.equal(after_mixer, torch.rand(()))


@pytest.mark.cpu_only
def test_block_forwards_the_backend_to_the_mixer(monkeypatch):
    seen = []
    original = NemotronV3Mamba2Mixer._build_projection

    def record(mamba_linear, in_features, out_features, config):
        seen.append(mamba_linear)
        return original("torch", in_features, out_features, config)

    monkeypatch.setattr(NemotronV3Mamba2Mixer, "_build_projection", staticmethod(record))
    backend = _mamba_linear_backend("torch")
    backend.mamba_linear = "te"  # bypass the constructor check: only the plumbing is under test here
    NemotronV3Block(MtpMockNemotronV3Config(layers_block_type=["mamba", "moe"]), layer_idx=0, backend=backend)
    assert seen == ["te", "te"]


@pytest.mark.cpu_only
def test_unknown_mamba_linear_is_rejected():
    with pytest.raises(ValueError, match="mamba_linear"):
        _mamba_linear_backend("quack")


@pytest.mark.parametrize("mamba_linear", ["torch", pytest.param("te", marks=pytest.mark.run_only_on("GPU"))])
def test_init_weights_restores_every_parameter_of_a_zeroed_mixer(mamba_linear):
    """Parameters that hold no usable values get their construction-time values, with either projection type.

    A model on more than one rank is built on the meta device and materialized before ``init_weights``
    runs, so the values its parameters had at construction are gone; zeros stand in for that here. A mixer
    left like that returns zeros and passes no gradient to its parameters.
    """
    if mamba_linear == "te":
        pytest.importorskip("transformer_engine")
    config = MtpMockNemotronV3Config(hidden_size=256, mamba_num_heads=8, mamba_head_dim=64)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_mamba_linear_backend(mamba_linear))
    assert (type(mixer.in_proj) is nn.Linear) == (mamba_linear == "torch")
    assert (type(mixer.out_proj) is nn.Linear) == (mamba_linear == "torch")
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.zero_()
    torch.manual_seed(0)
    num_hidden_layers = 4
    mixer.init_weights(num_hidden_layers=num_hidden_layers, buffer_device=torch.device("cpu"))
    # Uniform in +-1/sqrt(fan_in): standard deviation bound / sqrt(3). The output projection is rescaled.
    # Transformer Engine projections hold bfloat16 values, which round up to 1% past the bound.
    for weight, scale in ((mixer.in_proj.weight, 1.0), (mixer.out_proj.weight, math.sqrt(num_hidden_layers))):
        bound = 1.0 / math.sqrt(weight.shape[1]) / scale
        assert float(weight.float().abs().max()) <= bound * 1.01
        assert float(weight.float().std()) == pytest.approx(bound / math.sqrt(3), rel=0.03)
    # The depthwise convolution has a fan-in of its kernel size.
    conv_bound = 1.0 / math.sqrt(config.conv_kernel)
    assert 0.8 * conv_bound < float(mixer.conv1d.weight.abs().max()) <= conv_bound
    assert 0.8 * conv_bound < float(mixer.conv1d.bias.abs().max()) <= conv_bound
    torch.testing.assert_close(mixer.norm.weight, torch.ones_like(mixer.norm.weight))
    torch.testing.assert_close(mixer.D, torch.ones_like(mixer.D))
    torch.testing.assert_close(mixer.A_log, torch.log(torch.arange(1, config.mamba_num_heads + 1.0)))
    assert float(mixer.dt_bias.abs().min()) > 0


@pytest.mark.run_only_on("GPU")
def test_te_projections_start_from_the_torch_linear_distribution():
    pytest.importorskip("transformer_engine")
    config = MtpMockNemotronV3Config(hidden_size=256, mamba_num_heads=8, mamba_head_dim=64)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_mamba_linear_backend("te"))
    mixer.init_weights(num_hidden_layers=4, rescale_prenorm_residual=True, buffer_device=torch.device("cpu"))
    # torch.nn.Linear draws U(-1/sqrt(fan_in), 1/sqrt(fan_in)); the output projection is then divided by sqrt(layers).
    in_bound = 1.0 / math.sqrt(config.hidden_size)
    out_bound = 1.0 / math.sqrt(mixer.intermediate_size) / math.sqrt(4)
    for weight, bound in ((mixer.in_proj.weight.float(), in_bound), (mixer.out_proj.weight.float(), out_bound)):
        assert weight.abs().max().item() <= bound * 1.01
        assert weight.std().item() == pytest.approx(bound / math.sqrt(3), rel=0.05)


# Which padding mask the backbone hands to its Mamba layers.
#
# A ``[batch, sequence]`` mask of ones multiplies every hidden state by one, so the Mamba layers get no
# mask at all. Any other mask (padding zeros, or values other than one) must reach them unchanged.


@pytest.fixture
def model_and_masks(monkeypatch):
    seen: list[tuple[str, torch.Tensor | None]] = []

    def record(self, hidden_states, attention_mask=None, **kwargs):
        seen.append((self.block_type, attention_mask))
        return hidden_states

    monkeypatch.setattr(NemotronV3Block, "forward", record)
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    config = MtpMockNemotronV3Config(layers_block_type=["mamba", "attention"])
    return NemotronV3Model(config, backend=backend).float(), seen


def _mamba_masks(seen: list[tuple[str, torch.Tensor | None]]) -> list[torch.Tensor | None]:
    return [mask for block_type, mask in seen if block_type == "mamba"]


@pytest.mark.cpu_only
def test_mask_of_ones_is_not_passed_to_mamba_layers(model_and_masks):
    model, seen = model_and_masks
    input_ids = torch.randint(1, 64, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    model(input_ids, attention_mask=mask)
    assert _mamba_masks(seen) == [None]
    # Attention layers are unaffected: without a causal mask they still fall back to the 2D mask.
    assert [m for block_type, m in seen if block_type == "attention"][0] is mask


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=dtype)
        for dtype in (torch.long, torch.bool, torch.float32)
    ]
    # Only a multiply by exactly one is dropped; an indexed mask keeps its existing behaviour.
    + [torch.tensor([[1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]])],
    ids=["padding_long", "padding_bool", "padding_float", "values_other_than_one"],
)
def test_other_masks_reach_mamba_layers(model_and_masks, mask):
    model, seen = model_and_masks
    model(torch.randint(1, 64, (2, 6)), attention_mask=mask)
    (mamba_mask,) = _mamba_masks(seen)
    assert mamba_mask is mask


# ``NemotronV3BackendConfig.mamba_norm_backward_grid_scale`` and the mixer's ``scan_ac_exempt`` on a CPU stand-in.
#
# Which scan the mixer runs, how often an output projection that saves its input after computing (as a
# Transformer Engine linear does) runs under a non-reentrant activation checkpoint, and what context
# parallelism refuses. The comparisons with mamba_ssm's own scan on the real kernels are in
# ``tests/functional_tests/kernel_parity/test_nemotron_v3_mamba_scan.py``.


def _backend(**mamba: int | float | bool | str) -> NemotronV3BackendConfig:
    return NemotronV3BackendConfig(
        attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", **mamba
    )


@pytest.mark.cpu_only
def test_defaults_keep_the_library_scan_and_checkpointed_scan_state():
    backend = _backend()
    assert backend.mamba_norm_backward_grid_scale == 1
    for mixer in (
        NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0),
        NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0, backend=backend),
    ):
        assert mixer.norm_backward_grid_scale == 1
        assert mixer.scan_ac_exempt is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [2.5, 8.0, True, "8"])
def test_grid_scale_that_is_not_an_integer_is_rejected(scale):
    """YAML can deliver a float, a boolean or a string; the value multiplies a kernel's launch grid."""
    with pytest.raises(ValueError, match="mamba_norm_backward_grid_scale must be an integer"):
        _backend(mamba_norm_backward_grid_scale=scale)


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [0, -3])
def test_grid_scale_below_one_is_rejected(scale):
    with pytest.raises(ValueError, match="mamba_norm_backward_grid_scale"):
        _backend(mamba_norm_backward_grid_scale=scale)
    with pytest.raises(ValueError, match="norm_backward_grid_scale"):
        split_conv1d_scan(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1),
            None,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            chunk_size=1,
            seq_idx=None,
            dt_limit=(0.0, float("inf")),
            activation="silu",
            rmsnorm_weight=torch.zeros(1),
            rmsnorm_eps=1e-5,
            outproj_weight=None,
            outproj_bias=None,
            headdim=1,
            ngroups=1,
            norm_backward_grid_scale=scale,
            checkpoint_exempt=False,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(("scale", "scan_exempt"), [(1, False), (8, False), (1, True), (8, True)])
@pytest.mark.parametrize("fused", [True, False], ids=["torch_linear", "module_after_kernel"])
def test_mixer_selects_the_scan_by_grid_scale_and_exemption(monkeypatch, scale, scan_exempt, fused):
    torch.manual_seed(3)
    backend = _backend(mamba_norm_backward_grid_scale=scale)
    mixer = NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0, backend=backend).float()
    if not fused:
        # Only a plain ``torch.nn.Linear`` is applied inside the kernel; any other module, as a
        # Transformer Engine linear, runs after it.
        mixer.out_proj = _SaveAfterLinear(mixer.out_proj.weight)
    mixer.scan_ac_exempt = scan_exempt
    library_calls = _install_cpu_scan(monkeypatch, mixer)
    own_calls = []

    def own_scan(projected, conv_weight, conv_bias, dt_bias, a, d, **kwargs):
        own_calls.append({"conv_weight": conv_weight, "conv_bias": conv_bias, "dt_bias": dt_bias, "D": d, **kwargs})
        result = _pre_projection(projected, mixer.intermediate_size, kwargs["rmsnorm_weight"], kwargs["rmsnorm_eps"])
        if kwargs["outproj_weight"] is not None:
            result = result @ kwargs["outproj_weight"].T
        return result

    monkeypatch.setattr(nemotron_layers, "split_conv1d_scan", own_scan)
    x = torch.randn(2, 7, mixer.hidden_size)
    actual = mixer(x)

    # Independent expectation: the stand-in scan without a projection, then plain matrix algebra.
    expected = (
        _pre_projection(mixer.in_proj(x), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon)
        @ mixer.out_proj.weight.T
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if scale == 1 and not scan_exempt:
        assert len(library_calls) == 1 and own_calls == []
        assert (library_calls[0]["outproj_weight"] is mixer.out_proj.weight) == fused
        return
    assert library_calls == [] and len(own_calls) == 1
    call = own_calls[0]
    assert call["norm_backward_grid_scale"] == scale
    assert call["checkpoint_exempt"] is scan_exempt
    assert call["rmsnorm_weight"] is mixer.norm.weight
    assert call["rmsnorm_eps"] == mixer.norm.variance_epsilon
    assert (call["outproj_weight"] is mixer.out_proj.weight) == fused
    assert call["outproj_bias"] is None
    assert call["conv_weight"].shape == (mixer.conv_dim, mixer.conv_kernel_size)
    assert call["conv_bias"] is mixer.conv1d.bias
    assert call["chunk_size"] == mixer.chunk_size
    assert call["headdim"] == mixer.head_dim and call["ngroups"] == mixer.n_groups
    assert call["dt_limit"] == (0.0, float("inf"))
    assert call["activation"] == "silu" and call["seq_idx"] is None


class _SaveAfterMatmul(torch.autograd.Function):
    """``x @ weight.T`` that saves its inputs after computing, as a Transformer Engine linear does."""

    @staticmethod
    def forward(ctx, x, weight):
        out = x @ weight.T
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        return grad @ weight, torch.einsum("bso,bsi->oi", grad, x)


class _SaveAfterLinear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.computed = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.computed += 1
        return _SaveAfterMatmul.apply(x, self.weight)


@pytest.mark.cpu_only
@pytest.mark.parametrize("scan_exempt", [False, True])
def test_scan_ac_exempt_keeps_what_a_projection_after_the_kernel_saves(monkeypatch, scan_exempt):
    """The GEMM of a projection applied after the kernel runs twice under a checkpoint unless the scan is exempt.

    The stand-in scans are plain autograd operations, so they run in the forward and in the recompute either
    way: here only the tensors saved by the projection are kept. That the real scan runs once is checked on
    the real kernels.
    """
    torch.manual_seed(17)
    mixer = NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0, backend=_backend()).float()
    mixer.scan_ac_exempt = scan_exempt
    mixer.out_proj = _SaveAfterLinear(mixer.out_proj.weight)
    library_calls = _install_cpu_scan(monkeypatch, mixer)
    own_calls = []

    def own_scan(projected, conv_weight, conv_bias, dt_bias, a, d, **kwargs):
        own_calls.append(kwargs)
        return _pre_projection(projected, mixer.intermediate_size, kwargs["rmsnorm_weight"], kwargs["rmsnorm_eps"])

    monkeypatch.setattr(nemotron_layers, "split_conv1d_scan", own_scan)

    x = torch.randn(2, 7, mixer.hidden_size)
    hidden = x.clone().requires_grad_()
    out = checkpoint(mixer, hidden, use_reentrant=False)
    assert mixer.out_proj.computed == 1
    out.square().sum().backward()

    # An exempt scan goes through ``split_conv1d_scan``, the other one through the library scan.
    scan_calls, other_calls = (own_calls, library_calls) if scan_exempt else (library_calls, own_calls)
    assert len(scan_calls) == 2, "the stand-in scan runs in the forward and in the recompute"
    assert other_calls == []
    assert all(call["outproj_weight"] is None for call in scan_calls)
    assert mixer.out_proj.computed == (1 if scan_exempt else 2)

    # Independent expectation: an uncheckpointed stand-in scan and plain matrix algebra under autograd.
    reference_hidden = x.clone().requires_grad_()
    reference = (
        _pre_projection(
            mixer.in_proj(reference_hidden), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon
        )
        @ mixer.out_proj.weight.T
    )
    torch.testing.assert_close(out, reference, rtol=1e-6, atol=1e-6)
    names = ["hidden", "in_proj.weight", "norm.weight", "out_proj.weight"]
    actual_grads = [hidden.grad, mixer.in_proj.weight.grad, mixer.norm.weight.grad, mixer.out_proj.weight.grad]
    expected_grads = torch.autograd.grad(
        reference.square().sum(),
        (reference_hidden, mixer.in_proj.weight, mixer.norm.weight, mixer.out_proj.weight),
    )
    for name, actual, expected in zip(names, actual_grads, expected_grads):
        assert actual is not None and torch.count_nonzero(actual) > 0, name
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)


class _UnusedContextParallel:
    """Stands in for the context-parallel transport of a mixer; a refused forward never calls it."""


@pytest.mark.cpu_only
@pytest.mark.parametrize("scan_exempt", [True, False])
def test_checkpoint_exempt_scan_state_is_refused_with_context_parallelism(monkeypatch, scan_exempt):
    """The context-parallel scan has no checkpoint-exempt path: it must not silently recompute the state."""
    torch.manual_seed(23)
    mixer = NemotronV3Mamba2Mixer(MtpMockNemotronV3Config(), layer_idx=0, backend=_backend()).float()
    mixer.scan_ac_exempt = scan_exempt
    _install_cpu_scan(monkeypatch, mixer)
    x = torch.randn(2, 7, mixer.hidden_size)
    mixer.cp = _UnusedContextParallel()
    if scan_exempt:
        with pytest.raises(ValueError, match="not supported with context parallelism"):
            mixer(x)
    else:
        # The transport is reached: the stand-in has none of its methods.
        with pytest.raises(AttributeError, match="pre_conv_ssm"):
            mixer(x)


# Activation checkpointing of NemotronV3 MoE blocks in MoE-only mode (``mlp`` aliases ``mixer``).


def _moe_checkpointing_backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.mark.cpu_only
def test_moe_only_wraps_the_registered_moe_child() -> None:
    config = ModelMockNemotronV3Config(layers_block_type=["mamba", "attention", "moe"], num_hidden_layers=3)
    model = NemotronHForCausalLM(config, backend=_moe_checkpointing_backend())

    apply_ac(model, moe_only=True, hidden_size=config.hidden_size, num_experts=config.n_routed_experts)

    mamba_block, attention_block, moe_block = model.model.layers.values()
    assert all(isinstance(block, NemotronV3Block) for block in (mamba_block, attention_block, moe_block))
    # ``mlp`` is a property that aliases ``mixer``: the wrapper has to replace the registered child.
    assert isinstance(moe_block.mixer, CheckpointWrapper)
    assert moe_block.mlp is moe_block.mixer
    assert isinstance(_get_moe_module(moe_block), MoE)
    assert not isinstance(mamba_block.mixer, CheckpointWrapper)
    assert not isinstance(attention_block.mixer, CheckpointWrapper)


@pytest.mark.cpu_only
def test_moe_only_keeps_loss_and_gradients() -> None:
    """One CPU forward/backward with and without the MoE-only checkpoint: loss and gradients are bitwise equal."""
    config = ModelMockNemotronV3Config(layers_block_type=["attention", "moe"], torch_dtype="float32")
    torch.manual_seed(0)
    reference = NemotronHForCausalLM(config, backend=_moe_checkpointing_backend()).to(torch.float32)
    # The constructor leaves the routed-expert weights at zero; without this the expert gradients would be zero too.
    reference.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    reference.train()
    checkpointed = copy.deepcopy(reference)
    apply_ac(checkpointed, moe_only=True, hidden_size=config.hidden_size, num_experts=config.n_routed_experts)

    input_ids = torch.randint(0, config.vocab_size, (2, 8), generator=torch.Generator().manual_seed(3))

    def loss_and_grads(model: NemotronHForCausalLM) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1))
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref_loss, ref_grads = loss_and_grads(reference)
    loss, grads = loss_and_grads(checkpointed)

    assert torch.equal(loss, ref_loss)
    # The wrapper adds one path component to the names of the MoE parameters.
    assert {name.replace("._checkpoint_wrapped_module", "") for name in grads} == set(ref_grads)
    assert all(grad.abs().sum() > 0 for name, grad in ref_grads.items() if ".experts." in name)
    for name, grad in grads.items():
        assert torch.equal(grad, ref_grads[name.replace("._checkpoint_wrapped_module", "")]), name
