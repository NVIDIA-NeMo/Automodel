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

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.model import (
    Block,
    Step3p5ForCausalLM,
    Step3p5Model,
    parse_moe_layers_enum,
)
from nemo_automodel.components.moe.layers import MoEConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@dataclass
class MockStep3p5Config:
    """Mock configuration for Step3p5 model."""
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_attention_groups: int = 2
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    partial_rotary_factors: list = None
    layer_types: list = None
    attention_other_setting: dict = None
    sliding_window: int = None
    use_head_wise_attn_gate: bool = False
    use_rope_layers: list = None
    head_dim: int = 16
    attention_bias: bool = False
    torch_dtype: str = "bfloat16"
    moe_layers_enum: str = None
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_intermediate_size: int = 64
    moe_router_activation: str = "sigmoid"
    moe_router_scaling_factor: float = 1.0
    use_moe_router_bias: bool = False
    share_expert_dim: int = 64
    swiglu_limits: list = None
    swiglu_limits_shared: list = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention", "sliding_attention"]
        if self.attention_other_setting is None:
            self.attention_other_setting = {
                "num_attention_heads": 2,
                "num_attention_groups": 1,
            }
        if self.swiglu_limits is None:
            self.swiglu_limits = [None, None]
        if self.swiglu_limits_shared is None:
            self.swiglu_limits_shared = [None, None]


@pytest.fixture
def config():
    return MockStep3p5Config()


@pytest.fixture
def sdpa_backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


class TestParseMoELayersEnum:
    def test_with_none_returns_default(self):
        result = parse_moe_layers_enum(None, num_hidden_layers=4)
        assert result == {1, 2, 3}  # All except layer 0

    def test_with_valid_enum_string(self):
        result = parse_moe_layers_enum("1,3,5", num_hidden_layers=6)
        assert result == {1, 3, 5}

    def test_with_single_layer_string(self):
        result = parse_moe_layers_enum("2", num_hidden_layers=4)
        assert result == {2}

    def test_with_tuple_format(self):
        """Test HF Step-3.5-Flash style tuple format."""
        result = parse_moe_layers_enum((3, 4, 5, 6, 7), num_hidden_layers=10)
        assert result == {3, 4, 5, 6, 7}

    def test_with_list_format(self):
        result = parse_moe_layers_enum([1, 2, 3], num_hidden_layers=5)
        assert result == {1, 2, 3}

    def test_with_hf_step3p5_style_enum(self):
        """Test with Step-3.5-Flash style moe_layers_enum (layers 3-44)."""
        moe_layers = tuple(range(3, 45))  # (3, 4, 5, ..., 44)
        result = parse_moe_layers_enum(moe_layers, num_hidden_layers=45)
        assert result == set(range(3, 45))
        assert 0 not in result
        assert 1 not in result
        assert 2 not in result
        assert 3 in result
        assert 44 in result


class TestBlock:
    def test_initialization_non_moe_layer(self, config, moe_config, sdpa_backend):
        config.moe_layers_enum = "1"  # Only layer 1 is MoE
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=sdpa_backend)

        assert block.is_moe_layer is False
        assert block.mlp is not None
        assert block.moe is None
        assert block.share_expert is None

    def test_initialization_moe_layer(self, config, moe_config, sdpa_backend):
        config.moe_layers_enum = "1"  # Only layer 1 is MoE
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=sdpa_backend)

        assert block.is_moe_layer is True
        assert block.mlp is None
        assert block.moe is not None
        assert block.share_expert is not None

    def test_initialization_moe_layer_with_tuple_enum(self, config, moe_config, sdpa_backend):
        """Test MoE layer initialization with tuple format (HF Step-3.5-Flash style)."""
        config.moe_layers_enum = (1,)  # Only layer 1 is MoE
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=sdpa_backend)

        assert block.is_moe_layer is True
        assert block.moe is not None

    def test_shared_expert_uses_share_expert_dims(self, config, moe_config, sdpa_backend):
        """Test that HF's share_expert_dims config is used for shared expert."""
        config.moe_layers_enum = "1"
        # HF uses share_expert_dims (plural)
        config.share_expert_dims = 256
        delattr(config, "share_expert_dim")  # Remove the singular form

        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=sdpa_backend)

        assert block.share_expert is not None
        assert block.share_expert.intermediate_size == 256

    def test_forward_non_moe_shape_preserved(self, config, moe_config, sdpa_backend):
        config.moe_layers_enum = "1"  # Only layer 1 is MoE
        block = Block(layer_idx=0, config=config, moe_config=moe_config, backend=sdpa_backend)

        batch, seq = 2, 10
        x = torch.randn(batch, seq, config.hidden_size).to(torch.bfloat16)
        freqs_cis = torch.randn(batch, seq, config.head_dim)

        # Mock the attention and mlp to avoid actual computation
        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn:
            with patch.object(block.mlp, "forward", return_value=torch.zeros_like(x)):
                out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()

    def test_forward_moe_shape_preserved(self, config, moe_config, sdpa_backend):
        sdpa_backend.fake_balanced_gate = True
        config.moe_layers_enum = "1"  # Only layer 1 is MoE
        block = Block(layer_idx=1, config=config, moe_config=moe_config, backend=sdpa_backend)

        batch, seq = 2, 10
        x = torch.randn(batch, seq, config.hidden_size)
        freqs_cis = torch.randn(batch, seq, config.head_dim)

        # Mock the attention to avoid actual computation
        with patch.object(block.self_attn, "forward", return_value=torch.zeros_like(x)) as mock_attn:
            with patch.object(block.share_expert, "forward", return_value=torch.zeros_like(x)):
                with patch.object(block.moe, "forward", return_value=torch.zeros_like(x)):
                    out = block(x, freqs_cis=freqs_cis)

        assert out.shape == x.shape
        mock_attn.assert_called_once()


class TestStep3p5Model:
    def test_initialization(self, config, sdpa_backend):
        model = Step3p5Model(config, sdpa_backend)

        assert model.embed_tokens is not None
        assert len(model.layers) == config.num_hidden_layers
        assert model.norm is not None

    def test_forward_runs_all_layers(self, config, sdpa_backend):
        model = Step3p5Model(config, sdpa_backend)

        batch, seq = 2, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))
        freqs_mock = MagicMock(return_value=(1.0, torch.ones(config.head_dim // 2)))

        with patch.object(model.rotary_emb, "_compute_concentration_and_inv_freq", freqs_mock):
            with patch.object(Block, "forward", side_effect=lambda *_, **__: torch.randn(batch, seq, config.hidden_size)) as mock_block:
                out = model(input_ids)

        assert out.shape == (batch, seq, config.hidden_size)
        assert mock_block.call_count == config.num_hidden_layers


class TestStep3p5ForCausalLM:
    def test_initialization(self, config, sdpa_backend):
        model = Step3p5ForCausalLM(config, backend=sdpa_backend)

        assert model.model is not None
        assert model.lm_head is not None

    def test_forward_returns_logits(self, config, sdpa_backend):
        model = Step3p5ForCausalLM(config, backend=sdpa_backend)

        batch, seq = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))

        with patch.object(model.model, "forward", return_value=torch.randn(batch, seq, config.hidden_size).to(torch.bfloat16)):
            logits = model(input_ids)

        assert logits.shape == (batch, seq, config.vocab_size)

    def test_from_config(self, config, sdpa_backend):
        model = Step3p5ForCausalLM.from_config(config, backend=sdpa_backend)

        assert isinstance(model, Step3p5ForCausalLM)
        assert model.config == config

    def test_state_dict_adapter_created_when_enabled(self, config, sdpa_backend):
        sdpa_backend.enable_hf_state_dict_adapter = True
        model = Step3p5ForCausalLM(config, backend=sdpa_backend)

        assert hasattr(model, "state_dict_adapter")

    def test_need_fp32_gate_sets_gate_precision(self, config, sdpa_backend):
        """Test that need_fp32_gate config sets gate_precision to float32."""
        import torch

        config.need_fp32_gate = True
        config.moe_layers_enum = "1"  # Layer 1 is MoE
        model = Step3p5ForCausalLM(config, backend=sdpa_backend)

        # Check that backend.gate_precision was set
        assert sdpa_backend.gate_precision == torch.float32

    def test_config_num_experts_set_from_moe_num_experts(self, config, sdpa_backend):
        """Test that model.config.num_experts is set from config.moe_num_experts.

        This is important for activation checkpointing (apply_ac) which looks for
        config.num_experts to determine the router weight shape for selective AC.
        """
        config.moe_num_experts = 8  # Set explicitly
        model = Step3p5ForCausalLM(config, backend=sdpa_backend)

        # Verify config.num_experts is set to moe_num_experts value
        assert hasattr(model.model.config, "num_experts")
        assert model.model.config.num_experts == 8

    def test_moe_config_uses_num_experts_from_config(self, config, sdpa_backend):
        """Test that moe_config.n_routed_experts is set from config.num_experts."""
        config.moe_num_experts = 16
        model = Step3p5ForCausalLM(config, backend=sdpa_backend)

        assert model.model.moe_config.n_routed_experts == 16
