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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
from nemo_automodel.components.models.ling_v2.state_dict_adapter import BailingMoeV2StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


@pytest.fixture
def config():
    # Tiny but realistic: 2 layers, first dense, second MoE, 4 experts, 1 shared.
    return BailingMoeV2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        partial_rotary_factor=0.5,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )


@pytest.fixture
def moe_config(config):
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=config.num_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.0,
        force_e_score_correction_bias=True,
        score_func="sigmoid",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=config.norm_topk_prob,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        shared_expert_inter_dim=config.moe_intermediate_size,
        shared_expert_activation="swiglu",
        softmax_before_topk=False,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(config, moe_config, backend_config):
    return BailingMoeV2StateDictAdapter(
        config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32
    )


def _make_hf_state_dict(config):
    """Build a synthetic HF BailingMoeV2 checkpoint matching the public layout."""
    H = config.hidden_size
    D = config.head_dim
    Q = config.num_attention_heads * D
    K = config.num_key_value_heads * D
    sd = {}
    sd["model.word_embeddings.weight"] = torch.randn(config.vocab_size, H)
    sd["model.norm.weight"] = torch.ones(H)
    sd["lm_head.weight"] = torch.randn(config.vocab_size, H)
    for L in range(config.num_hidden_layers):
        # Attention: fused QKV + dense
        sd[f"model.layers.{L}.attention.query_key_value.weight"] = torch.randn(Q + 2 * K, H)
        sd[f"model.layers.{L}.attention.dense.weight"] = torch.randn(H, Q)
        sd[f"model.layers.{L}.attention.query_layernorm.weight"] = torch.ones(D)
        sd[f"model.layers.{L}.attention.key_layernorm.weight"] = torch.ones(D)
        # Block norms
        sd[f"model.layers.{L}.input_layernorm.weight"] = torch.ones(H)
        sd[f"model.layers.{L}.post_attention_layernorm.weight"] = torch.ones(H)
        # MLP: dense for first_k_dense_replace, MoE otherwise
        if L < config.first_k_dense_replace:
            sd[f"model.layers.{L}.mlp.gate_proj.weight"] = torch.randn(config.intermediate_size, H)
            sd[f"model.layers.{L}.mlp.up_proj.weight"] = torch.randn(config.intermediate_size, H)
            sd[f"model.layers.{L}.mlp.down_proj.weight"] = torch.randn(H, config.intermediate_size)
        else:
            sd[f"model.layers.{L}.mlp.gate.weight"] = torch.randn(config.num_experts, H)
            sd[f"model.layers.{L}.mlp.gate.expert_bias"] = torch.zeros(config.num_experts)
            for E in range(config.num_experts):
                sd[f"model.layers.{L}.mlp.experts.{E}.gate_proj.weight"] = torch.randn(config.moe_intermediate_size, H)
                sd[f"model.layers.{L}.mlp.experts.{E}.up_proj.weight"] = torch.randn(config.moe_intermediate_size, H)
                sd[f"model.layers.{L}.mlp.experts.{E}.down_proj.weight"] = torch.randn(H, config.moe_intermediate_size)
            sd[f"model.layers.{L}.mlp.shared_experts.gate_proj.weight"] = torch.randn(config.moe_intermediate_size, H)
            sd[f"model.layers.{L}.mlp.shared_experts.up_proj.weight"] = torch.randn(config.moe_intermediate_size, H)
            sd[f"model.layers.{L}.mlp.shared_experts.down_proj.weight"] = torch.randn(H, config.moe_intermediate_size)
    return sd


class TestSplitFusedQKV:
    def test_splits_into_q_k_v_in_order(self, adapter, config):
        hf_sd = _make_hf_state_dict(config)
        native = adapter._split_fused_qkv_and_rename(hf_sd)

        for L in range(config.num_hidden_layers):
            assert f"model.layers.{L}.self_attn.q_proj.weight" in native
            assert f"model.layers.{L}.self_attn.k_proj.weight" in native
            assert f"model.layers.{L}.self_attn.v_proj.weight" in native
            assert f"model.layers.{L}.attention.query_key_value.weight" not in native

            fused = hf_sd[f"model.layers.{L}.attention.query_key_value.weight"]
            q_size = config.num_attention_heads * config.head_dim
            kv_size = config.num_key_value_heads * config.head_dim
            assert torch.equal(native[f"model.layers.{L}.self_attn.q_proj.weight"], fused[:q_size])
            assert torch.equal(native[f"model.layers.{L}.self_attn.k_proj.weight"], fused[q_size : q_size + kv_size])
            assert torch.equal(native[f"model.layers.{L}.self_attn.v_proj.weight"], fused[q_size + kv_size :])

    def test_renames_attention_subkeys(self, adapter, config):
        hf_sd = _make_hf_state_dict(config)
        native = adapter._split_fused_qkv_and_rename(hf_sd)

        for L in range(config.num_hidden_layers):
            assert f"model.layers.{L}.self_attn.o_proj.weight" in native
            assert f"model.layers.{L}.self_attn.q_norm.weight" in native
            assert f"model.layers.{L}.self_attn.k_norm.weight" in native
            assert f"model.layers.{L}.attention.dense.weight" not in native

    def test_renames_embed_and_expert_bias(self, adapter, config):
        hf_sd = _make_hf_state_dict(config)
        native = adapter._split_fused_qkv_and_rename(hf_sd)

        assert "model.embed_tokens.weight" in native
        assert "model.word_embeddings.weight" not in native
        # MoE layers (>= first_k_dense_replace) carry the bias buffer.
        assert "model.layers.1.mlp.gate.e_score_correction_bias" in native, (
            "expert_bias must be renamed to e_score_correction_bias"
        )
        assert "model.layers.1.mlp.gate.expert_bias" not in native

    def test_rejects_wrong_fused_qkv_shape(self, adapter, config):
        bad = {
            "model.layers.0.attention.query_key_value.weight": torch.randn(7, config.hidden_size),
        }
        with pytest.raises(ValueError, match="Fused qkv"):
            adapter._split_fused_qkv_and_rename(bad)


class TestToHFRefusion:
    def test_refuses_q_k_v_back_into_query_key_value(self, adapter, config):
        H = config.hidden_size
        D = config.head_dim
        q_size = config.num_attention_heads * D
        kv_size = config.num_key_value_heads * D
        native = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(q_size, H),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(kv_size, H),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(kv_size, H),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(H, q_size),
            "model.layers.0.self_attn.q_norm.weight": torch.ones(D),
            "model.layers.0.self_attn.k_norm.weight": torch.ones(D),
            "model.embed_tokens.weight": torch.randn(config.vocab_size, H),
        }
        hf = adapter.to_hf(native)
        assert "model.layers.0.attention.query_key_value.weight" in hf
        assert hf["model.layers.0.attention.query_key_value.weight"].shape == (q_size + 2 * kv_size, H)
        assert "model.layers.0.attention.dense.weight" in hf
        assert "model.layers.0.attention.query_layernorm.weight" in hf
        assert "model.layers.0.attention.key_layernorm.weight" in hf
        assert "model.word_embeddings.weight" in hf

        # Round-trip: re-split the just-fused tensor and ensure each slice matches the input.
        fused = hf["model.layers.0.attention.query_key_value.weight"]
        torch.testing.assert_close(fused[:q_size], native["model.layers.0.self_attn.q_proj.weight"])
        torch.testing.assert_close(fused[q_size : q_size + kv_size], native["model.layers.0.self_attn.k_proj.weight"])
        torch.testing.assert_close(fused[q_size + kv_size :], native["model.layers.0.self_attn.v_proj.weight"])


class TestRoundTrip:
    def test_hf_to_native_to_hf_is_lossless_for_qkv_path(self, adapter, config):
        """fused -> split -> fused must reproduce the original (the slowest, finickiest path)."""
        hf_sd = _make_hf_state_dict(config)
        native = adapter._split_fused_qkv_and_rename(hf_sd)
        roundtripped = adapter.to_hf(native)

        for L in range(config.num_hidden_layers):
            key = f"model.layers.{L}.attention.query_key_value.weight"
            assert key in roundtripped
            torch.testing.assert_close(roundtripped[key], hf_sd[key])
            torch.testing.assert_close(
                roundtripped[f"model.layers.{L}.attention.dense.weight"],
                hf_sd[f"model.layers.{L}.attention.dense.weight"],
            )
        torch.testing.assert_close(roundtripped["model.word_embeddings.weight"], hf_sd["model.word_embeddings.weight"])
