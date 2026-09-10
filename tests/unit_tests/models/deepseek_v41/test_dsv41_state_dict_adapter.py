# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU unit tests for the DeepSeek V4.1 state-dict adapter."""

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import _ExpertQuantLayout
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
    _internal_key_to_hf,
    _rename_hf_key,
    dequantize_engram_table,
    dequantize_fp8_blocks,
    infer_fp8_block_size,
)
from nemo_automodel.components.moe.config import MoEConfig
from tests.unit_tests.models.deepseek_v41.conftest import tiny_config


def _e8m0(exponents: torch.Tensor) -> torch.Tensor:
    """Build an e8m0 tensor holding ``2 ** exponents``."""
    return (exponents + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def _make_adapter(**config_overrides):
    config = tiny_config(**config_overrides)
    moe_config = Mock(spec=MoEConfig)
    moe_config.n_routed_experts = config.n_routed_experts
    moe_config.moe_inter_dim = config.moe_intermediate_size
    return DeepSeekV41StateDictAdapter(config, moe_config, BackendConfig(), dtype=torch.float32)


class TestRenames:
    @pytest.mark.parametrize(
        ("hf_key", "internal_key"),
        [
            ("embed.weight", "model.embed_tokens.weight"),
            ("norm.weight", "model.norm.weight"),
            ("head.weight", "lm_head.weight"),
            ("layers.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
            ("layers.3.ffn_norm.weight", "model.layers.3.post_attention_layernorm.weight"),
            ("layers.3.attn.attn_sink", "model.layers.3.self_attn.sinks_param.weight"),
            ("layers.3.attn.wq_a.weight", "model.layers.3.self_attn.wq_a.weight"),
            ("layers.2.attn.compressor.wgate.weight", "model.layers.2.self_attn.compressor.wgate.weight"),
            ("layers.2.attn.compressor.norm.weight", "model.layers.2.self_attn.compressor.norm.weight"),
            ("layers.2.attn.indexer.wk.weight", "model.layers.2.self_attn.indexer.wk.weight"),
            ("layers.5.attn.indexer.weights_proj.weight", "model.layers.5.self_attn.indexer.weights_proj.weight"),
            ("layers.3.ffn.gate.weight", "model.layers.3.mlp.gate.weight"),
            ("layers.3.ffn.gate.bias", "model.layers.3.mlp.gate.e_score_correction_bias"),
            ("layers.3.ffn.gate.bias_vl", "model.layers.3.mlp.gate.bias_vl"),
            ("vision.norm.weight", "model.vision.norm.weight"),
            ("aligner.w1.weight", "model.aligner.w1.weight"),
            ("image_start", "model.image_start"),
            ("image_end", "model.image_end"),
            ("image_newline", "model.image_newline"),
            ("layers.3.ffn.shared_experts.w1.weight", "model.layers.3.mlp.shared_experts.gate_proj.weight"),
            ("layers.3.ffn.shared_experts.w3.weight", "model.layers.3.mlp.shared_experts.up_proj.weight"),
            ("layers.3.ffn.shared_experts.w2.weight", "model.layers.3.mlp.shared_experts.down_proj.weight"),
            ("layers.3.hc_attn_fn", "model.layers.3.attn_hc.fn"),
            ("layers.3.hc_ffn_scale", "model.layers.3.ffn_hc.scale"),
            ("layers.1.engram.embed.weight", "model.layers.1.engram.embed.weight"),
            ("layers.1.engram.q_weight", "model.layers.1.engram.q_weight"),
        ],
    )
    def test_round_trip(self, hf_key, internal_key):
        assert _rename_hf_key(hf_key) == internal_key
        assert _internal_key_to_hf(internal_key) == hf_key

    def test_unknown_key_unchanged(self):
        assert _rename_hf_key("mystery.weight") == "mystery.weight"


class TestDequantization:
    def test_infer_block_size(self):
        assert infer_fp8_block_size((512, 5120), (16, 160)) == 32
        assert infer_fp8_block_size((8192, 4096), (256, 128)) == 32
        assert infer_fp8_block_size((1024, 4096), (8, 32)) == 128
        with pytest.raises(ValueError):
            infer_fp8_block_size((64, 64), (5, 5))

    def test_fp8_32_block_dequant_matches_manual(self):
        torch.manual_seed(0)
        weight = (torch.randn(64, 96) * 4).to(torch.float8_e4m3fn)
        scale = _e8m0(torch.randint(-4, 4, (2, 3)))
        out = dequantize_fp8_blocks(weight, scale, torch.float32)
        expected = weight.float().view(2, 32, 3, 32) * torch.pow(2.0, scale.view(torch.uint8).int() - 127).float().view(
            2, 1, 3, 1
        )
        assert torch.equal(out, expected.view(64, 96))

    def test_fp8_dequant_handles_ragged_last_block(self):
        weight = torch.ones(40, 40).to(torch.float8_e4m3fn)
        scale = _e8m0(torch.tensor([[1, 2], [3, 4]]))
        out = dequantize_fp8_blocks(weight, scale, torch.float32)
        assert out.shape == (40, 40)
        assert out[0, 0] == 2.0 and out[0, 39] == 4.0 and out[39, 0] == 8.0 and out[39, 39] == 16.0

    def test_engram_table_dequant(self):
        weight = torch.ones(4, 64).to(torch.float8_e4m3fn)
        scale = _e8m0(torch.tensor([[0, 1], [2, 3], [-1, -2], [0, 0]]))
        out = dequantize_engram_table(weight, scale, torch.bfloat16)
        assert out.dtype == torch.bfloat16 and out.shape == (4, 64)
        assert out[0, 0] == 1.0 and out[0, 63] == 2.0 and out[1, 0] == 4.0 and out[2, 40] == 0.25
        with pytest.raises(ValueError, match="does not match"):
            dequantize_engram_table(weight, scale[:, :1], torch.bfloat16)


class TestFromHf:
    def test_drops_out_of_scope_tensors_and_dequantizes(self):
        adapter = _make_adapter()
        hf = {
            "embed.weight": torch.zeros(4, 8),
            "mtp.0.attn.wq_a.weight": torch.zeros(2, 2),
            "vision.norm.weight": torch.zeros(2),
            "aligner.w1.weight": torch.zeros(2, 2),
            "image_start": torch.zeros(2),
            "layers.0.ffn.gate.bias_vl": torch.zeros(8),
            "layers.0.ffn.gate.bias": torch.zeros(8),
            "layers.0.attn.wkv.weight": torch.ones(64, 64).to(torch.float8_e4m3fn),
            "layers.0.attn.wkv.scale": _e8m0(torch.full((2, 2), 1)),
            "layers.1.engram.embed.weight": torch.ones(3, 32).to(torch.float8_e4m3fn),
            "layers.1.engram.embed.scale": _e8m0(torch.full((3, 1), 2)),
            "layers.1.engram.q_weight": torch.ones(4, 64),
        }
        out = adapter.from_hf(hf)
        assert set(out) == {
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate.e_score_correction_bias",
            "model.layers.0.mlp.gate.bias_vl",
            "model.layers.0.self_attn.wkv.weight",
            "model.layers.1.engram.embed.weight",
            "model.layers.1.engram.q_weight",
        }
        assert torch.equal(out["model.layers.0.self_attn.wkv.weight"], torch.full((64, 64), 2.0))
        assert torch.equal(out["model.layers.1.engram.embed.weight"], torch.full((3, 32), 4.0))

    def test_engram_disabled_drops_engram_keys(self):
        adapter = _make_adapter(engram_enabled=False)
        out = adapter.from_hf({"layers.1.engram.q_weight": torch.ones(4, 64), "norm.weight": torch.ones(64)})
        assert set(out) == {"model.norm.weight"}

    def test_fp4_experts_are_unpacked_and_stacked(self):
        adapter = _make_adapter()
        n_experts, inter, hidden = adapter.moe_config.n_routed_experts, 32, 64
        hf = {}
        for eid in range(n_experts):
            # nibble 0x2 == 1.0 in the E2M1 table, packed twice per byte -> value 1.0 everywhere.
            for name, out_dim, in_dim in (("w1", inter, hidden), ("w3", inter, hidden), ("w2", hidden, inter)):
                hf[f"layers.0.ffn.experts.{eid}.{name}.weight"] = torch.full(
                    (out_dim, in_dim // 2), 0x22, dtype=torch.int8
                )
                hf[f"layers.0.ffn.experts.{eid}.{name}.scale"] = _e8m0(torch.full((out_dim, in_dim // 32), eid % 3))
        out = adapter.from_hf(hf)
        gate_up = out["model.layers.0.mlp.experts.gate_and_up_projs"]
        down = out["model.layers.0.mlp.experts.down_projs"]
        assert gate_up.shape == (n_experts, hidden, 2 * inter)
        assert down.shape == (n_experts, inter, hidden)
        for eid in range(n_experts):
            assert torch.equal(gate_up[eid], torch.full((hidden, 2 * inter), float(2 ** (eid % 3))))
            assert torch.equal(down[eid], torch.full((inter, hidden), float(2 ** (eid % 3))))


class TestToHf:
    def test_renames_and_splits_experts(self):
        adapter = _make_adapter()
        n_experts, inter, hidden = adapter.moe_config.n_routed_experts, 32, 64
        internal = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.arange(
                n_experts * hidden * 2 * inter, dtype=torch.float32
            ).view(n_experts, hidden, 2 * inter),
            "model.layers.0.mlp.experts.down_projs": torch.zeros(n_experts, inter, hidden),
            "model.layers.0.self_attn.sinks_param.weight": torch.zeros(4),
            "lm_head.weight": torch.zeros(8, 4),
        }
        hf = adapter.to_hf(internal)
        assert "layers.0.attn.attn_sink" in hf and "head.weight" in hf
        assert hf["layers.0.ffn.experts.3.w1.weight"].shape == (inter, hidden)
        assert hf["layers.0.ffn.experts.3.w2.weight"].shape == (hidden, inter)
        expected_w3 = internal["model.layers.0.mlp.experts.gate_and_up_projs"][3, :, inter:].T
        assert torch.equal(hf["layers.0.ffn.experts.3.w3.weight"], expected_w3)

    def test_quantization_placeholders_follow_on_disk_layout(self, monkeypatch):
        adapter = _make_adapter(engram_num_embeddings=[5])
        monkeypatch.setattr(adapter, "_checkpoint_expert_quant_layout", lambda: _ExpertQuantLayout.FP4)
        n_experts, inter, hidden = adapter.moe_config.n_routed_experts, 32, 64
        internal = {
            "model.layers.0.self_attn.wq_b.weight": torch.zeros(96, 64),
            "model.layers.0.self_attn.indexer.wq_b.weight": torch.zeros(64, 32),
            "model.layers.0.self_attn.indexer.wk.weight": torch.zeros(32, 64),
            "model.layers.0.mlp.shared_experts.up_proj.weight": torch.zeros(32, 64),
            "model.layers.1.engram.embed.weight": torch.zeros(5, 64),
            "model.layers.1.engram.wkv.weight": torch.zeros(320, 128),
            "model.layers.0.mlp.gate.weight": torch.zeros(8, 64),
            "model.layers.0.mlp.experts.down_projs": torch.zeros(n_experts, inter, hidden),
            "model.norm.weight": torch.zeros(64),
        }
        hf = adapter.to_hf(internal, quantization=True)
        assert hf["layers.0.attn.wq_b.weight"].dtype == torch.float8_e4m3fn
        assert hf["layers.0.attn.wq_b.scale"].dtype == torch.float8_e8m0fnu
        assert hf["layers.0.attn.wq_b.scale"].shape == (3, 2)
        assert hf["layers.0.attn.indexer.wq_b.scale"].shape == (2, 1)
        assert hf["layers.0.ffn.shared_experts.w3.scale"].shape == (1, 2)
        assert hf["layers.1.engram.wkv.scale"].shape == (10, 4)
        assert hf["layers.1.engram.embed.weight"].dtype == torch.float8_e4m3fn
        assert hf["layers.1.engram.embed.scale"].shape == (5, 2)
        assert hf["layers.0.ffn.experts.0.w2.weight"].dtype == torch.int8
        assert hf["layers.0.ffn.experts.0.w2.weight"].shape == (hidden, inter // 2)
        assert hf["layers.0.ffn.experts.0.w2.scale"].shape == (hidden, inter // 32)
        # Unquantized on disk: no scale companions.
        for key in ("layers.0.attn.indexer.wk", "layers.0.ffn.gate", "norm"):
            assert f"{key}.scale" not in hf
            assert hf[f"{key}.weight"].dtype == torch.float32

    def test_from_hf_to_hf_key_round_trip(self):
        adapter = _make_adapter()
        hf = {
            "embed.weight": torch.zeros(4, 8),
            "layers.2.attn.compressor.norm.weight": torch.ones(64),
            "layers.2.attn.indexer.k_norm.weight": torch.ones(32),
            "layers.5.attn.indexer.wq_b.weight": torch.zeros(64, 32),
            "layers.0.hc_ffn_base": torch.zeros(24),
            "layers.1.engram.k_weight": torch.ones(4, 64),
        }
        internal = adapter.from_hf(dict(hf))
        assert set(adapter.to_hf(internal)) == set(hf)


@pytest.mark.parametrize("vision_layers", [0, 1])
def test_optional_vision_and_always_present_router_bias_use_consistent_protocols(vision_layers):
    """Discovery, import, export and forced FP32 selection share one scope."""
    adapter = _make_adapter(vision_config={"num_hidden_layers": vision_layers})
    source = {
        "vision.norm.weight": torch.tensor([1.00123], dtype=torch.float32),
        "aligner.w1.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "image_start": torch.ones(2, dtype=torch.bfloat16),
        "image_end": torch.ones(2, dtype=torch.bfloat16),
        "image_newline": torch.ones(2, dtype=torch.bfloat16),
        "layers.0.ffn.gate.bias_vl": torch.tensor([0.1234567]),
        "mtp.0.ffn.gate.bias_vl": torch.ones(1),
        "image_pad": torch.ones(2),
    }
    wanted = {"layers.0.ffn.gate.bias_vl"}
    if vision_layers:
        wanted.update({"vision.norm.weight", "aligner.w1.weight", "image_start", "image_end", "image_newline"})
    native = adapter.from_hf(source)
    assert set(adapter.get_hf_state_dict_keys(native)) == wanted
    exported = adapter.to_hf(native)
    assert exported.keys() == wanted
    for name in wanted:
        torch.testing.assert_close(exported[name], source[name], rtol=0, atol=0)
    forced = adapter.forced_hf_dtype_mapping(native)
    assert forced == {name: "float32" for name in wanted if source[name].dtype == torch.float32}


@pytest.mark.parametrize("scale_byte", [0, 255], ids=["smallest_exponent", "nan"])
def test_legacy_fp8_block_scales_preserve_e8m0_special_values(scale_byte: int) -> None:
    """The retained 128x128 decoder must preserve E8M0's boundary encodings."""
    weight = torch.ones((128, 128)).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), scale_byte, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    actual = dequantize_fp8_blocks(weight, scale, torch.float32)
    if scale_byte == 255:
        assert torch.isnan(actual).all()
    else:
        assert torch.equal(actual, torch.full((128, 128), 2.0**-127, dtype=torch.float32))
