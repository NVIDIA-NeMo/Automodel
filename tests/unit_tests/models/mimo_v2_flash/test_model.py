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

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.model import (
    MiMoV2FlashAttention,
    MiMoV2FlashBlock,
    MiMoV2FlashForCausalLM,
    MiMoV2FlashModel,
    ModelClass,
)
from nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter import (
    MiMoV2FlashStateDictAdapter,
)
from nemo_automodel.components.moe.layers import MLP, MoE


@pytest.fixture
def tiny_config():
    """Tiny config exercising the dense-then-MoE pattern across full+sliding layers."""
    return MiMoV2FlashConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        swa_v_head_dim=8,
        max_position_embeddings=64,
        layernorm_epsilon=1e-6,
        rope_theta=10000.0,
        swa_rope_theta=10000.0,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        partial_rotary_factor=0.5,
        sliding_window=4,
        sliding_window_size=4,
        attention_chunk_size=4,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_layer_freq=[0, 1, 1, 1],  # layer 0 dense; rest MoE
        hybrid_layer_pattern=[0, 1, 0, 1],  # alternating full/sliding
        torch_dtype="float32",
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class TestMiMoV2FlashAttention:
    def test_full_attention_projection_shapes(self, tiny_config, backend_config):
        attn = MiMoV2FlashAttention(tiny_config, backend_config, is_swa=False, layer_idx=0)
        assert attn.num_attention_heads == tiny_config.num_attention_heads
        assert attn.num_key_value_heads == tiny_config.num_key_value_heads
        assert attn.head_dim == tiny_config.head_dim
        assert attn.v_head_dim == tiny_config.v_head_dim
        assert attn.q_proj.weight.shape == (
            tiny_config.num_attention_heads * tiny_config.head_dim,
            tiny_config.hidden_size,
        )
        assert attn.k_proj.weight.shape == (
            tiny_config.num_key_value_heads * tiny_config.head_dim,
            tiny_config.hidden_size,
        )
        assert attn.v_proj.weight.shape == (
            tiny_config.num_key_value_heads * tiny_config.v_head_dim,
            tiny_config.hidden_size,
        )
        assert attn.o_proj.weight.shape == (
            tiny_config.hidden_size,
            tiny_config.num_attention_heads * tiny_config.v_head_dim,
        )

    def test_swa_attention_uses_swa_head_dims(self, tiny_config, backend_config):
        tiny_config.swa_num_attention_heads = 8
        tiny_config.swa_head_dim = 4
        attn = MiMoV2FlashAttention(tiny_config, backend_config, is_swa=True, layer_idx=1)
        assert attn.num_attention_heads == 8
        assert attn.head_dim == 4

    def test_swa_layer_has_attention_sink_when_configured(self, tiny_config, backend_config):
        """SWA sink bias defaults on; full attention defaults off."""
        attn_swa = MiMoV2FlashAttention(tiny_config, backend_config, is_swa=True, layer_idx=1)
        attn_full = MiMoV2FlashAttention(tiny_config, backend_config, is_swa=False, layer_idx=0)
        assert attn_swa.attention_sink_bias is not None
        assert attn_full.attention_sink_bias is None

    def test_v_scale_attribute_picked_up(self, tiny_config, backend_config):
        attn = MiMoV2FlashAttention(tiny_config, backend_config, is_swa=False, layer_idx=0)
        assert attn.v_scale == tiny_config.attention_value_scale

    def test_rope_dim_partial_factor(self, tiny_config, backend_config):
        """rope_dim = head_dim * partial_rotary_factor, rounded down to even."""
        # head_dim=8, partial_rotary_factor=0.5 → rope_dim=4
        attn = MiMoV2FlashAttention(tiny_config, backend_config, is_swa=False, layer_idx=0)
        assert attn.rope_dim == 4


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class TestMiMoV2FlashBlock:
    def test_layer_0_is_dense_full_attention(self, tiny_config, backend_config):
        block = MiMoV2FlashBlock(0, tiny_config, _moe_config(tiny_config), backend_config)
        # moe_layer_freq[0]==0 → MLP; hybrid_layer_pattern[0]==0 → full_attention
        assert isinstance(block.mlp, MLP)
        assert block.attention_type == "full_attention"

    def test_layer_1_is_moe_sliding(self, tiny_config, backend_config):
        block = MiMoV2FlashBlock(1, tiny_config, _moe_config(tiny_config), backend_config)
        assert isinstance(block.mlp, MoE)
        assert block.attention_type == "sliding_attention"

    def test_layer_2_is_moe_full(self, tiny_config, backend_config):
        block = MiMoV2FlashBlock(2, tiny_config, _moe_config(tiny_config), backend_config)
        assert isinstance(block.mlp, MoE)
        assert block.attention_type == "full_attention"

    def test_block_records_layer_idx(self, tiny_config, backend_config):
        block = MiMoV2FlashBlock(3, tiny_config, _moe_config(tiny_config), backend_config)
        assert block.layer_idx == 3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TestMiMoV2FlashModel:
    def test_moe_config_defaults_from_hf(self, tiny_config, backend_config):
        model = MiMoV2FlashModel(tiny_config, backend_config)
        cfg = model.moe_config
        assert cfg.dim == tiny_config.hidden_size
        assert cfg.inter_dim == tiny_config.intermediate_size
        assert cfg.moe_inter_dim == tiny_config.moe_intermediate_size
        assert cfg.n_routed_experts == tiny_config.n_routed_experts
        assert cfg.n_activated_experts == tiny_config.num_experts_per_tok
        assert cfg.score_func == "sigmoid_with_bias"  # mapped from scoring_func="sigmoid"
        assert cfg.norm_topk_prob == tiny_config.norm_topk_prob
        assert cfg.force_e_score_correction_bias is True

    def test_scoring_func_passthrough_when_not_sigmoid(self, tiny_config, backend_config):
        tiny_config.scoring_func = "softmax"
        model = MiMoV2FlashModel(tiny_config, backend_config)
        assert model.moe_config.score_func == "softmax"

    def test_sets_backend_gate_precision_to_fp32(self, tiny_config, backend_config):
        """MiMo follows Pattern A — gate compute in fp32 via backend.gate_precision."""
        assert backend_config.gate_precision is None
        MiMoV2FlashModel(tiny_config, backend_config)
        assert backend_config.gate_precision == torch.float32

    def test_respects_existing_gate_precision(self, tiny_config, backend_config):
        backend_config.gate_precision = torch.bfloat16
        MiMoV2FlashModel(tiny_config, backend_config)
        # Existing override is preserved
        assert backend_config.gate_precision == torch.bfloat16

    def test_moe_overrides_apply(self, tiny_config, backend_config):
        model = MiMoV2FlashModel(tiny_config, backend_config, moe_overrides={"aux_loss_coeff": 0.5})
        assert model.moe_config.aux_loss_coeff == 0.5

    def test_rejects_both_moe_config_and_overrides(self, tiny_config, backend_config):
        from nemo_automodel.components.moe.config import MoEConfig

        cfg = MoEConfig(
            dim=32,
            inter_dim=64,
            moe_inter_dim=16,
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            norm_topk_prob=True,
        )
        with pytest.raises(ValueError, match="Cannot pass both"):
            MiMoV2FlashModel(tiny_config, backend_config, moe_config=cfg, moe_overrides={"aux_loss_coeff": 0.1})

    def test_layer_count_matches_config(self, tiny_config, backend_config):
        model = MiMoV2FlashModel(tiny_config, backend_config)
        assert len(model.layers) == tiny_config.num_hidden_layers

    def test_has_swa_rotary_emb_only_when_sliding_present(self, tiny_config, backend_config):
        """swa_rotary_emb is always constructed (used by sliding layers)."""
        model = MiMoV2FlashModel(tiny_config, backend_config)
        assert hasattr(model, "swa_rotary_emb")
        assert hasattr(model, "rotary_emb")

    def test_has_sliding_layers_flag(self, tiny_config, backend_config):
        model = MiMoV2FlashModel(tiny_config, backend_config)
        # hybrid_layer_pattern=[0,1,0,1] has sliding layers
        assert model.has_sliding_layers is True

        tiny_config.hybrid_layer_pattern = [0, 0, 0, 0]
        # Recompute derived layer_types
        tiny_config.layer_types = ["full_attention"] * 4
        model_no_swa = MiMoV2FlashModel(tiny_config, backend_config)
        assert model_no_swa.has_sliding_layers is False


# ---------------------------------------------------------------------------
# Causal-LM head
# ---------------------------------------------------------------------------
class TestMiMoV2FlashForCausalLM:
    def test_from_config_builds_model(self, tiny_config, backend_config):
        model = MiMoV2FlashForCausalLM.from_config(tiny_config, backend=backend_config)
        assert isinstance(model, MiMoV2FlashForCausalLM)
        assert model.config is tiny_config

    def test_from_pretrained_uses_classmethod(self, tiny_config, backend_config):
        with patch(
            "nemo_automodel.components.models.mimo_v2_flash.model.MiMoV2FlashConfig.from_pretrained",
            return_value=tiny_config,
        ) as mock_from_pretrained:
            model = MiMoV2FlashForCausalLM.from_pretrained("XiaomiMiMo/MiMo-V2-Flash", backend=backend_config)
        mock_from_pretrained.assert_called_once_with("XiaomiMiMo/MiMo-V2-Flash")
        assert isinstance(model, MiMoV2FlashForCausalLM)

    def test_lm_head_shape(self, tiny_config, backend_config):
        model = MiMoV2FlashForCausalLM(tiny_config, backend=backend_config)
        assert model.lm_head.weight.shape == (tiny_config.vocab_size, tiny_config.hidden_size)

    def test_state_dict_adapter_off_by_default(self, tiny_config, backend_config):
        model = MiMoV2FlashForCausalLM(tiny_config, backend=backend_config)
        assert not hasattr(model, "state_dict_adapter")

    def test_state_dict_adapter_when_enabled(self, tiny_config, backend_config):
        backend_config.enable_hf_state_dict_adapter = True
        model = MiMoV2FlashForCausalLM(tiny_config, backend=backend_config)
        assert isinstance(model.state_dict_adapter, MiMoV2FlashStateDictAdapter)

    def test_input_output_embeddings_accessors(self, tiny_config, backend_config):
        model = MiMoV2FlashForCausalLM(tiny_config, backend=backend_config)
        assert model.get_input_embeddings() is model.model.embed_tokens
        assert model.get_output_embeddings() is model.lm_head

        new_emb = torch.nn.Embedding(tiny_config.vocab_size, tiny_config.hidden_size)
        model.set_input_embeddings(new_emb)
        assert model.model.embed_tokens is new_emb

        new_head = torch.nn.Linear(tiny_config.hidden_size, tiny_config.vocab_size, bias=False)
        model.set_output_embeddings(new_head)
        assert model.lm_head is new_head

    def test_keep_in_fp32_modules_strict_includes_buffers(self):
        """Buffers stay in fp32 regardless of activation dtype; gate weight is bf16 (Pattern A)."""
        assert "mlp.gate.e_score_correction_bias" in MiMoV2FlashForCausalLM._keep_in_fp32_modules_strict
        assert "attention_sink_bias" in MiMoV2FlashForCausalLM._keep_in_fp32_modules_strict
        # gate weight is no longer kept in fp32 (Pattern A uses gate_precision instead).
        assert "mlp.gate.weight" not in MiMoV2FlashForCausalLM._keep_in_fp32_modules_strict


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------
class TestModelClassExport:
    def test_modelclass_points_to_causal_lm(self):
        assert ModelClass is MiMoV2FlashForCausalLM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _moe_config(cfg: MiMoV2FlashConfig):
    from nemo_automodel.components.moe.config import MoEConfig

    return MoEConfig(
        dim=cfg.hidden_size,
        inter_dim=cfg.intermediate_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts or 0,
        n_activated_experts=cfg.num_experts_per_tok,
        n_expert_groups=cfg.n_group,
        n_limited_groups=cfg.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid_with_bias",
        route_scale=cfg.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=cfg.norm_topk_prob,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
        dtype=torch.float32,
    )
