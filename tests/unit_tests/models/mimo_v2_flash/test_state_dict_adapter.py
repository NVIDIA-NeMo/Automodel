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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter import (
    NON_QUANTIZED_KEY_PATTERNS,
    MiMoV2FlashStateDictAdapter,
    _should_quantize_key,
)
from nemo_automodel.components.moe.config import MoEConfig


@pytest.fixture
def hf_config():
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid_with_bias",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(hf_config, moe_config, backend_config):
    return MiMoV2FlashStateDictAdapter(
        config=hf_config, moe_config=moe_config, backend=backend_config, dtype=torch.float32
    )


class TestShouldQuantizeKey:
    @pytest.mark.parametrize(
        "key",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
        ],
    )
    def test_quantizable_weights(self, key):
        assert _should_quantize_key(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ],
    )
    def test_non_quantizable_weights(self, key):
        assert _should_quantize_key(key) is False

    def test_non_weight_key_skipped(self):
        # buffers / biases / scale tensors aren't quantized
        assert _should_quantize_key("model.layers.0.mlp.gate.e_score_correction_bias") is False
        assert _should_quantize_key("model.layers.0.self_attn.q_proj.bias") is False
        assert _should_quantize_key("model.layers.0.mlp.experts.0.up_proj.weight_scale_inv") is False


class TestMiMoV2FlashStateDictAdapterInit:
    def test_stores_fields(self, hf_config, moe_config, backend_config):
        adapter = MiMoV2FlashStateDictAdapter(hf_config, moe_config, backend_config, dtype=torch.bfloat16)
        assert adapter.config is hf_config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype is torch.bfloat16
        assert adapter._uses_model_prefix is True


class TestFromHf:
    def test_drops_scale_inv_keys_and_renames(self, adapter):
        hf_state = {
            "model.embed_tokens.weight": torch.randn(8, 4),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 2),
            "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv": torch.ones(1, 1),
        }
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd) as mock_merge:
            with patch(
                "nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter.dequantize_from_fp8",
                side_effect=lambda w, _, dtype, name: w.to(dtype),
            ) as mock_dequant:
                out = adapter.from_hf(hf_state)
        # scale_inv keys must be removed after dequant
        assert "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv" not in out
        assert "model.layers.0.mlp.experts.0.up_proj.weight" in out
        mock_dequant.assert_called_once()
        mock_merge.assert_called_once()

    def test_uses_model_prefix_detection_with_prefix(self, adapter):
        hf_state = {"model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 2)}
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd):
            adapter.from_hf(hf_state)
        assert adapter._uses_model_prefix is True

    def test_uses_model_prefix_detection_without_prefix(self, adapter):
        hf_state = {"layers.0.mlp.experts.0.up_proj.weight": torch.zeros(2, 2)}
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd):
            adapter.from_hf(hf_state)
        assert adapter._uses_model_prefix is False

    def test_forwards_device_mesh_to_merge_helper(self, adapter):
        hf_state = {"model.embed_tokens.weight": torch.randn(2, 2)}
        mesh = Mock()
        with patch.object(adapter, "_from_hf_w_merged_experts", return_value=hf_state) as mock_merge:
            adapter.from_hf(hf_state, device_mesh=mesh)
        assert mock_merge.call_args[0][1] is mesh


class TestConvertSingleTensorToHf:
    def test_non_expert_passthrough(self, adapter):
        tensor = torch.randn(4, 4)
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            # Non-quantized key (matches "embed_tokens.weight") → no FP8 cast
            out = adapter.convert_single_tensor_to_hf("model.embed_tokens.weight", tensor)
        assert len(out) == 1
        assert out[0][0] == "model.embed_tokens.weight"
        assert out[0][1] is tensor

    def test_quantizes_quantizable_key(self, adapter):
        tensor = torch.randn(8, 8)
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            out = adapter.convert_single_tensor_to_hf("model.layers.0.self_attn.q_proj.weight", tensor)
        # Returns (weight in fp8, weight_scale_inv) pair.
        keys = [k for k, _ in out]
        assert "model.layers.0.self_attn.q_proj.weight" in keys
        assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in keys
        weight_kv = [(k, v) for k, v in out if k == "model.layers.0.self_attn.q_proj.weight"][0]
        assert weight_kv[1].dtype == torch.float8_e4m3fn

    def test_expert_split_keys_get_quantized(self, adapter):
        tensor = torch.randn(4, 16, 64)
        split_pairs = [
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(32, 16)),
        ]
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=split_pairs):
            out = adapter.convert_single_tensor_to_hf("model.layers.0.mlp.experts.gate_and_up_projs", tensor)
        keys = [k for k, _ in out]
        # The split expert weight must be fp8'd and a scale_inv must be emitted.
        assert "model.layers.0.mlp.experts.0.up_proj.weight" in keys
        assert "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv" in keys

    def test_exclude_key_regex_filters_before_quantize(self, adapter):
        tensor = torch.randn(4, 4)
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            out = adapter.convert_single_tensor_to_hf("lm_head.weight", tensor, exclude_key_regex=r"lm_head.*")
        assert out == []


class TestKPadScaleInv:
    def test_pads_k_proj_scale_inv_when_too_few_rows(self, hf_config, moe_config, backend_config):
        """Per FP8 layout, k_proj scale_inv must have at least 8 rows when full_k_rows matches."""
        hf_config.num_key_value_heads = 2
        hf_config.head_dim = 16
        # full_k_rows = 2 * 16 = 32. Weight shape[0] must equal 32 to trigger the pad branch.
        adapter = MiMoV2FlashStateDictAdapter(hf_config, moe_config, backend_config, dtype=torch.float32)

        # Mock the underlying scale_inv helper to return a 3-row tensor (< 8 rows)
        fake_scale_inv = torch.ones(3, 4)
        with patch(
            "nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter.create_scale_inv_for_weight",
            return_value=fake_scale_inv,
        ):
            weight = torch.zeros(32, 64)
            out = adapter._create_scale_inv_for_hf_key("model.layers.0.self_attn.k_proj.weight", weight)
        # Padded to 8 rows
        assert out.shape == (8, 4)
        # First 3 rows preserved, remaining 5 rows are ones (the pad)
        torch.testing.assert_close(out[:3], fake_scale_inv)
        torch.testing.assert_close(out[3:], torch.ones(5, 4))

    def test_does_not_pad_when_already_full(self, hf_config, moe_config, backend_config):
        hf_config.num_key_value_heads = 2
        hf_config.head_dim = 16
        adapter = MiMoV2FlashStateDictAdapter(hf_config, moe_config, backend_config, dtype=torch.float32)

        fake_scale_inv = torch.ones(8, 4)
        with patch(
            "nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter.create_scale_inv_for_weight",
            return_value=fake_scale_inv,
        ):
            weight = torch.zeros(32, 64)
            out = adapter._create_scale_inv_for_hf_key("model.layers.0.self_attn.k_proj.weight", weight)
        assert out.shape == (8, 4)

    def test_no_pad_for_non_k_proj(self, hf_config, moe_config, backend_config):
        adapter = MiMoV2FlashStateDictAdapter(hf_config, moe_config, backend_config, dtype=torch.float32)
        fake_scale_inv = torch.ones(3, 4)
        with patch(
            "nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter.create_scale_inv_for_weight",
            return_value=fake_scale_inv,
        ):
            weight = torch.zeros(16, 64)
            out = adapter._create_scale_inv_for_hf_key("model.layers.0.self_attn.q_proj.weight", weight)
        # q_proj never triggers the k-proj-specific pad path
        assert out.shape == (3, 4)


class TestDequantize:
    def test_dequantizes_pairs_and_removes_scale_keys(self, adapter):
        weight = torch.zeros(4, 4, dtype=torch.float8_e4m3fn)
        state = {
            "model.layers.0.self_attn.q_proj.weight": weight,
            "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(1, 1),
            "model.embed_tokens.weight": torch.randn(4, 4),
        }
        with patch(
            "nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter.dequantize_from_fp8",
            side_effect=lambda w, _, dtype, name: torch.zeros(4, 4, dtype=dtype),
        ):
            out = adapter._dequantize(state)
        assert "model.layers.0.self_attn.q_proj.weight" in out
        assert "model.layers.0.self_attn.q_proj.weight_scale_inv" not in out
        assert "model.embed_tokens.weight" in out
        # Dequantized to the adapter dtype
        assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float32

    def test_skips_when_no_scale_inv(self, adapter):
        weight = torch.randn(2, 2)
        state = {"model.layers.0.self_attn.q_proj.weight": weight}
        out = adapter._dequantize(state)
        # No scale_inv → no dequant; tensor passes through untouched.
        assert out["model.layers.0.self_attn.q_proj.weight"] is weight


class TestNonQuantizedKeyPatterns:
    def test_all_expected_patterns_listed(self):
        # Sanity check: explicit allowlist for HF round-trip layout.
        expected = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "norm.weight",
            "lm_head.weight",
            "embed_tokens.weight",
            "mlp.gate.weight",
            "self_attn.o_proj.weight",
        }
        assert set(NON_QUANTIZED_KEY_PATTERNS) == expected
