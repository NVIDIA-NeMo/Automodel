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

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
    _rename_hf_key,
)
from nemo_automodel.components.moe.config import MoEConfig


def _make_adapter(**config_overrides):
    config = DeepseekV4Config(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        **config_overrides,
    )
    moe_config = Mock(spec=MoEConfig)
    moe_config.n_routed_experts = 4
    moe_config.moe_inter_dim = 32
    backend = BackendConfig()
    return DeepSeekV4StateDictAdapter(config, moe_config, backend, dtype=torch.float32)


class TestRenameHfKey:
    def test_embed(self):
        assert _rename_hf_key("embed.weight") == "model.embed_tokens.weight"

    def test_norm(self):
        assert _rename_hf_key("norm.weight") == "model.norm.weight"

    def test_head(self):
        assert _rename_hf_key("head.weight") == "lm_head.weight"

    def test_attn_norm(self):
        assert _rename_hf_key("layers.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"

    def test_ffn_norm(self):
        assert _rename_hf_key("layers.2.ffn_norm.weight") == "model.layers.2.post_attention_layernorm.weight"

    def test_attn_wq_a(self):
        assert _rename_hf_key("layers.1.attn.wq_a.weight") == "model.layers.1.self_attn.wq_a.weight"

    def test_attn_wkv(self):
        assert _rename_hf_key("layers.0.attn.wkv.weight") == "model.layers.0.self_attn.wkv.weight"

    def test_attn_attn_sink(self):
        # ``attn_sink`` (HF) maps to ``self_attn.sinks`` to match the
        # ``module.sinks`` attribute consumed by ``eager_attention_with_sink``.
        assert _rename_hf_key("layers.0.attn.attn_sink") == "model.layers.0.self_attn.sinks"

    def test_gate_weight(self):
        assert _rename_hf_key("layers.1.ffn.gate.weight") == "model.layers.1.mlp.gate.weight"

    def test_gate_bias_to_e_score_correction_bias(self):
        assert _rename_hf_key("layers.1.ffn.gate.bias") == "model.layers.1.mlp.gate.e_score_correction_bias"

    def test_gate_tid2eid(self):
        assert _rename_hf_key("layers.0.ffn.gate.tid2eid") == "model.layers.0.mlp.gate.tid2eid"

    def test_shared_expert_w1(self):
        # The MoE module exposes the shared expert as ``mlp.shared_experts``
        # (plural) — the rename keeps that name.
        result = _rename_hf_key("layers.0.ffn.shared_experts.w1.weight")
        assert result == "model.layers.0.mlp.shared_experts.gate_proj.weight"

    def test_shared_expert_w3(self):
        result = _rename_hf_key("layers.0.ffn.shared_experts.w3.weight")
        assert result == "model.layers.0.mlp.shared_experts.up_proj.weight"

    def test_shared_expert_w2(self):
        result = _rename_hf_key("layers.0.ffn.shared_experts.w2.weight")
        assert result == "model.layers.0.mlp.shared_experts.down_proj.weight"

    def test_hc_attn_fn(self):
        # HF flat ``hc_attn_*`` keys map into the ``attn_hc`` HyperConnection
        # submodule (``ffn_hc`` for the MLP-site one).
        assert _rename_hf_key("layers.2.hc_attn_fn") == "model.layers.2.attn_hc.fn"

    def test_hc_ffn_scale(self):
        assert _rename_hf_key("layers.0.hc_ffn_scale") == "model.layers.0.ffn_hc.scale"

    def test_unknown_key_unchanged(self):
        assert _rename_hf_key("some.unknown.key") == "some.unknown.key"


class TestDeepSeekV4StateDictAdapterFromHF:
    def test_rename_all(self):
        adapter = _make_adapter()
        sd = {
            "embed.weight": torch.zeros(256, 64),
            "norm.weight": torch.ones(64),
            "head.weight": torch.zeros(256, 64),
            "layers.0.attn_norm.weight": torch.ones(64),
            "layers.0.attn.wq_a.weight": torch.zeros(32, 64),
            "layers.0.ffn.gate.weight": torch.zeros(4, 64),
            "layers.0.ffn.gate.bid": torch.zeros(4),
            "layers.0.hc_attn_fn": torch.zeros(24, 256),
        }
        out = adapter._rename_all(sd)
        assert "model.embed_tokens.weight" in out
        assert "model.norm.weight" in out
        assert "lm_head.weight" in out
        assert "model.layers.0.input_layernorm.weight" in out
        assert "model.layers.0.self_attn.wq_a.weight" in out
        assert "model.layers.0.mlp.gate.weight" in out
        assert "model.layers.0.attn_hc.fn" in out

    def test_tid2eid_skips_dequantize(self):
        adapter = _make_adapter()
        tid2eid = torch.randint(0, 4, (256, 2), dtype=torch.int32)
        sd = {
            "embed.weight": torch.zeros(256, 64),
            "layers.0.ffn.gate.tid2eid": tid2eid,
        }
        out = adapter._dequantize(sd)
        # tid2eid must be unchanged (int32 preserved)
        assert out["layers.0.ffn.gate.tid2eid"].dtype == torch.int32
        assert torch.equal(out["layers.0.ffn.gate.tid2eid"], tid2eid)

    def test_expert_aggregation_no_mesh(self):
        adapter = _make_adapter()
        inter_dim = 32
        hidden = 64
        n_experts = 4
        sd = {}
        for eid in range(n_experts):
            sd[f"layers.0.ffn.experts.{eid}.w1.weight"] = torch.randn(inter_dim, hidden)
            sd[f"layers.0.ffn.experts.{eid}.w3.weight"] = torch.randn(inter_dim, hidden)
            sd[f"layers.0.ffn.experts.{eid}.w2.weight"] = torch.randn(hidden, inter_dim)
        out = adapter._aggregate_experts(sd, device_mesh=None)
        assert "model.layers.0.mlp.experts.gate_and_up_projs" in out
        assert "model.layers.0.mlp.experts.down_projs" in out
        gate_up = out["model.layers.0.mlp.experts.gate_and_up_projs"]
        assert gate_up.shape == (n_experts, hidden, 2 * inter_dim)
        down = out["model.layers.0.mlp.experts.down_projs"]
        assert down.shape == (n_experts, inter_dim, hidden)

    def test_expert_fp8_base_layout_uses_fp8_dequantizer(self):
        adapter = _make_adapter()
        weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, 1, dtype=torch.float32)
        sentinel = torch.empty(4, 4)

        with patch(
            "nemo_automodel.components.models.deepseek_v4.state_dict_adapter.dequantize_from_fp8",
            return_value=sentinel,
        ) as mock_dequantize:
            out = adapter._dequantize_expert_weight("layers.0.ffn.experts.0.w1.weight", weight, scale)

        assert out is sentinel
        mock_dequantize.assert_called_once_with(
            weight,
            scale,
            dtype=adapter.dtype,
            name="layers.0.ffn.experts.0.w1.weight",
        )

    def test_expert_fp4_flash_layout_uses_fp4_dequantizer(self):
        adapter = _make_adapter()
        weight = torch.zeros(4, 2, dtype=torch.int8)
        scale = torch.ones(4, 1, dtype=torch.float8_e8m0fnu)
        sentinel = torch.empty(4, 4)

        with patch.object(
            DeepSeekV4StateDictAdapter,
            "_dequantize_expert_fp4",
            return_value=sentinel,
        ) as mock_dequantize:
            out = adapter._dequantize_expert_weight("layers.0.ffn.experts.0.w1.weight", weight, scale)

        assert out is sentinel
        mock_dequantize.assert_called_once_with(weight, scale, adapter.dtype)


class TestDeepSeekV4StateDictAdapterToHF:
    def test_split_gate_up(self):
        adapter = _make_adapter()
        n_experts, hidden, inter = 4, 64, 32
        # gate_and_up_projs is [n_experts, hidden, 2*inter] (transposed from per-expert w1/w3)
        gate_up = torch.randn(n_experts, hidden, 2 * inter)
        pairs = adapter._split_merged_expert("model.layers.0.mlp.experts.gate_and_up_projs", gate_up)
        # Should produce 2*n_experts entries: w1 and w3 for each expert
        assert len(pairs) == 2 * n_experts
        keys = {k for k, _ in pairs}
        assert "layers.0.ffn.experts.0.w1.weight" in keys
        assert "layers.0.ffn.experts.0.w3.weight" in keys

    def test_split_down(self):
        adapter = _make_adapter()
        n_experts, hidden, inter = 4, 64, 32
        down = torch.randn(n_experts, inter, hidden)
        pairs = adapter._split_merged_expert("model.layers.0.mlp.experts.down_projs", down)
        assert len(pairs) == n_experts
        keys = {k for k, _ in pairs}
        assert "layers.0.ffn.experts.0.w2.weight" in keys

    def test_internal_key_to_hf_gate(self):
        adapter = _make_adapter()
        assert (
            adapter._internal_key_to_hf("model.layers.1.mlp.gate.e_score_correction_bias") == "layers.1.ffn.gate.bias"
        )
        assert adapter._internal_key_to_hf("model.layers.1.mlp.gate.weight") == "layers.1.ffn.gate.weight"
        assert adapter._internal_key_to_hf("model.layers.1.mlp.gate.tid2eid") == "layers.1.ffn.gate.tid2eid"

    def test_non_quantized_gate_bias(self):
        adapter = _make_adapter()
        assert adapter._is_non_quantized("ffn.gate.bias")
        assert adapter._is_non_quantized("ffn.gate.tid2eid")
        assert adapter._is_non_quantized("attn.attn_sink")

    def test_to_hf_quantization_creates_fp8_expert_placeholders_for_base(self, monkeypatch):
        monkeypatch.setenv("NEMO_AUTOMODEL_DSV4_EXPERT_LAYOUT", "base")
        adapter = _make_adapter()
        down = torch.randn(4, 32, 64)

        pairs = adapter.convert_single_tensor_to_hf(
            "model.layers.0.mlp.experts.down_projs",
            down,
            quantization=True,
        )
        by_key = dict(pairs)

        assert by_key["layers.0.ffn.experts.0.w2.weight"].dtype == torch.float8_e4m3fn
        assert by_key["layers.0.ffn.experts.0.w2.weight"].shape == (64, 32)
        assert by_key["layers.0.ffn.experts.0.w2.scale"].dtype == torch.float32
        assert by_key["layers.0.ffn.experts.0.w2.scale"].shape == (1, 1)

    def test_to_hf_quantization_keeps_fp4_expert_placeholders_for_flash(self, monkeypatch):
        monkeypatch.setenv("NEMO_AUTOMODEL_DSV4_EXPERT_LAYOUT", "flash")
        adapter = _make_adapter()
        down = torch.randn(4, 32, 64)

        pairs = adapter.convert_single_tensor_to_hf(
            "model.layers.0.mlp.experts.down_projs",
            down,
            quantization=True,
        )
        by_key = dict(pairs)

        assert by_key["layers.0.ffn.experts.0.w2.weight"].dtype == torch.int8
        assert by_key["layers.0.ffn.experts.0.w2.weight"].shape == (64, 16)
        assert by_key["layers.0.ffn.experts.0.w2.scale"].dtype == torch.float8_e8m0fnu
        assert by_key["layers.0.ffn.experts.0.w2.scale"].shape == (64, 1)
