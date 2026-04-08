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

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.state_dict_adapter import (
    Mistral4MultimodalStateDictAdapter,
    Mistral4StateDictAdapter,
    _convert_aggregated_experts,
    _dequantize_state_dict,
    _inject_missing_gate_bias,
    _should_quantize_key,
)
from nemo_automodel.components.moe.config import MoEConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="softmax_with_bias",
        route_scale=1.0,
        aux_loss_coeff=0,
        norm_topk_prob=True,
    )


@pytest.fixture
def backend():
    return BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False)


@pytest.fixture
def text_adapter(moe_config, backend):
    config = Mock()
    config.hidden_size = 64
    config.torch_dtype = "bfloat16"
    return Mistral4StateDictAdapter(config=config, moe_config=moe_config, backend=backend, dtype=torch.bfloat16)


@pytest.fixture
def mm_adapter(moe_config, backend):
    config = Mock()
    config.text_config = Mock()
    config.text_config.hidden_size = 64
    config.text_config.torch_dtype = "bfloat16"
    return Mistral4MultimodalStateDictAdapter(
        config=config, moe_config=moe_config, backend=backend, dtype=torch.bfloat16
    )


# ---------------------------------------------------------------------------
# _should_quantize_key
# ---------------------------------------------------------------------------


class TestShouldQuantizeKey:
    def test_standard_weight_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.self_attn.q_a_proj.weight") is True

    def test_expert_gate_up_proj_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.mlp.experts.gate_up_proj") is True

    def test_expert_down_proj_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.mlp.experts.down_proj") is True

    def test_layernorm_not_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.input_layernorm.weight") is False

    def test_embed_tokens_not_quantized(self):
        assert _should_quantize_key("language_model.model.embed_tokens.weight") is False

    def test_lm_head_not_quantized(self):
        assert _should_quantize_key("language_model.lm_head.weight") is False

    def test_gate_weight_not_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.mlp.gate.weight") is False

    def test_norm_weight_not_quantized(self):
        assert _should_quantize_key("language_model.model.norm.weight") is False

    def test_vision_tower_not_quantized(self):
        assert _should_quantize_key("vision_tower.transformer.layers.0.attention.q_proj.weight") is False

    def test_projector_not_quantized(self):
        assert _should_quantize_key("multi_modal_projector.linear_1.weight") is False

    def test_non_weight_key_not_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.mlp.gate.e_score_correction_bias") is False

    def test_scale_inv_key_not_quantized(self):
        assert _should_quantize_key("language_model.model.layers.0.self_attn.q_a_proj.weight_scale_inv") is False


# ---------------------------------------------------------------------------
# _dequantize_state_dict
# ---------------------------------------------------------------------------


class TestDequantizeStateDict:
    def test_per_tensor_scalar_scale(self):
        """Per-tensor dequant: weight * scale_inv (scalar)."""
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor(0.5)
        sd = {"x.weight": weight, "x.weight_scale_inv": scale_inv}
        result = _dequantize_state_dict(sd, torch.float32)
        assert "x.weight" in result
        assert "x.weight_scale_inv" not in result
        expected = weight.float() * 0.5
        torch.testing.assert_close(result["x.weight"], expected, rtol=0.1, atol=0.1)

    def test_per_expert_scale(self):
        """Per-expert dequant: 3D weight with 1D scale [n_experts]."""
        n_experts = 4
        weight = torch.randn(n_experts, 8, 8).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sd = {"mlp.experts.gate_up_proj": weight, "mlp.experts.gate_up_proj_scale_inv": scale_inv}
        result = _dequantize_state_dict(sd, torch.float32)
        assert "mlp.experts.gate_up_proj" in result
        assert "mlp.experts.gate_up_proj_scale_inv" not in result
        assert result["mlp.experts.gate_up_proj"].dtype == torch.float32

    def test_activation_scale_removed(self):
        """Activation scale keys should be removed."""
        sd = {
            "x.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
            "x.weight_scale_inv": torch.tensor(0.5),
            "x.activation_scale": torch.tensor(1.0),
        }
        result = _dequantize_state_dict(sd, torch.float32)
        assert "x.activation_scale" not in result

    def test_non_quantized_keys_preserved(self):
        """Non-quantized keys should pass through unchanged."""
        norm_weight = torch.randn(64)
        sd = {"norm.weight": norm_weight}
        result = _dequantize_state_dict(sd, torch.float32)
        assert torch.equal(result["norm.weight"], norm_weight)


# ---------------------------------------------------------------------------
# _convert_aggregated_experts
# ---------------------------------------------------------------------------


class TestConvertAggregatedExperts:
    def test_gate_up_proj_transpose(self):
        """gate_up_proj [n, a, b] -> gate_and_up_projs [n, b, a]."""
        n, a, b = 4, 8, 16
        weight = torch.randn(n, a, b)
        sd = {"layers.0.mlp.experts.gate_up_proj": weight}
        result = _convert_aggregated_experts(sd)
        assert "layers.0.mlp.experts.gate_and_up_projs" in result
        assert "layers.0.mlp.experts.gate_up_proj" not in result
        assert result["layers.0.mlp.experts.gate_and_up_projs"].shape == (n, b, a)

    def test_down_proj_transpose(self):
        """down_proj [n, a, b] -> down_projs [n, b, a]."""
        n, a, b = 4, 16, 8
        weight = torch.randn(n, a, b)
        sd = {"layers.0.mlp.experts.down_proj": weight}
        result = _convert_aggregated_experts(sd)
        assert "layers.0.mlp.experts.down_projs" in result
        assert "layers.0.mlp.experts.down_proj" not in result
        assert result["layers.0.mlp.experts.down_projs"].shape == (n, b, a)

    def test_scale_keys_not_converted(self):
        """_scale_inv and _activation_scale keys should not be converted."""
        sd = {
            "layers.0.mlp.experts.gate_up_proj_scale_inv": torch.tensor([0.1, 0.2]),
            "layers.0.mlp.experts.down_proj_activation_scale": torch.tensor([0.3, 0.4]),
        }
        result = _convert_aggregated_experts(sd)
        assert "layers.0.mlp.experts.gate_up_proj_scale_inv" in result
        assert "layers.0.mlp.experts.down_proj_activation_scale" in result

    def test_non_expert_keys_preserved(self):
        """Non-expert keys should pass through unchanged."""
        w = torch.randn(64)
        sd = {"layers.0.input_layernorm.weight": w}
        result = _convert_aggregated_experts(sd)
        assert torch.equal(result["layers.0.input_layernorm.weight"], w)


# ---------------------------------------------------------------------------
# _inject_missing_gate_bias
# ---------------------------------------------------------------------------


class TestInjectMissingGateBias:
    def test_injects_when_missing(self):
        """Injects zero bias when gate.weight exists but bias does not."""
        sd = {
            "model.layers.0.mlp.gate.weight": torch.randn(4, 64),
            "model.layers.1.mlp.gate.weight": torch.randn(4, 64),
        }
        result = _inject_missing_gate_bias(sd, n_routed_experts=4)
        assert "model.layers.0.mlp.gate.e_score_correction_bias" in result
        assert "model.layers.1.mlp.gate.e_score_correction_bias" in result
        assert result["model.layers.0.mlp.gate.e_score_correction_bias"].shape == (4,)
        torch.testing.assert_close(
            result["model.layers.0.mlp.gate.e_score_correction_bias"],
            torch.zeros(4, dtype=torch.float32),
        )

    def test_no_op_when_bias_exists(self):
        """Does not overwrite existing bias."""
        existing_bias = torch.ones(4)
        sd = {
            "model.layers.0.mlp.gate.weight": torch.randn(4, 64),
            "model.layers.0.mlp.gate.e_score_correction_bias": existing_bias,
        }
        result = _inject_missing_gate_bias(sd, n_routed_experts=4)
        assert result["model.layers.0.mlp.gate.e_score_correction_bias"] is existing_bias

    def test_no_op_when_no_gate(self):
        """Does nothing when there are no gate.weight keys."""
        sd = {"model.layers.0.input_layernorm.weight": torch.randn(64)}
        result = _inject_missing_gate_bias(sd, n_routed_experts=4)
        assert "e_score_correction_bias" not in str(result.keys())


# ---------------------------------------------------------------------------
# _dequantize_state_dict — 3D scale squeeze
# ---------------------------------------------------------------------------


class TestDequantizeMultiDimMesh:
    def test_mesh_idx_passed_to_get_local_rank_and_size(self):
        """Regression: get_local_rank / size must receive the mesh_idx of the Shard(0) placement.

        Simulates rank 1 of 2 on a 2D mesh (EP_SHARD=2, EP=16).  Shard(0) sits at
        placement index 1 (the EP dim), so the fix should call get_local_rank(1) and
        size(1) — NOT the no-arg versions that crash on multi-dim meshes.
        """
        from unittest.mock import MagicMock, PropertyMock, patch

        from torch.distributed._tensor import Shard

        n_local = 4  # this rank's experts after sharding
        weight_local = torch.randn(n_local, 8, 8).to(torch.float8_e4m3fn)
        scale_all = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Mock DTensor: placements=(Replicate(), Shard(0)) — Shard(0) at mesh_idx=1
        mock_weight = MagicMock()
        mock_weight.to_local.return_value = weight_local
        type(mock_weight).placements = PropertyMock(return_value=(MagicMock(), Shard(0)))
        type(mock_weight).dtype = PropertyMock(return_value=torch.float8_e4m3fn)

        mock_mesh = MagicMock()
        mock_mesh.ndim = 2
        # rank 1 of 2 in the EP dimension → experts 4..7
        mock_mesh.get_local_rank.return_value = 1
        mock_mesh.size.return_value = 2
        type(mock_weight).device_mesh = PropertyMock(return_value=mock_mesh)

        sd = {"mlp.experts.gate_up_proj": mock_weight, "mlp.experts.gate_up_proj_scale_inv": scale_all}

        with (
            patch(
                "nemo_automodel.components.models.mistral4.state_dict_adapter.is_dtensor",
                side_effect=lambda t: t is mock_weight,
            ),
            patch("torch.distributed._tensor.DTensor.from_local", side_effect=lambda t, *a, **kw: t),
        ):
            result = _dequantize_state_dict(sd, torch.float32)

        # Shard(0) is at placement index 1, so mesh_idx=1 must be passed
        mock_mesh.get_local_rank.assert_called_once_with(1)
        mock_mesh.size.assert_called_once_with(1)

        # Rank 1 of 2 over 8 experts → chunk_size=4, start=4 → scale[4:8]
        expected_scale = scale_all[4:8].float().view(-1, 1, 1)
        expected = (weight_local.float() * expected_scale).to(torch.float32)
        torch.testing.assert_close(result["mlp.experts.gate_up_proj"], expected)


class TestDequantize3DScale:
    def test_per_expert_scale_3d(self):
        """scale_inv [n_experts, 1, 1] is squeezed and dequantized correctly."""
        n_experts = 4
        weight = torch.randn(n_experts, 8, 8).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([0.1, 0.2, 0.3, 0.4]).view(4, 1, 1)  # 3D
        sd = {
            "mlp.experts.gate_up_proj": weight,
            "mlp.experts.gate_up_proj_scale_inv": scale_inv,
        }
        result = _dequantize_state_dict(sd, torch.float32)
        assert "mlp.experts.gate_up_proj" in result
        assert "mlp.experts.gate_up_proj_scale_inv" not in result
        assert result["mlp.experts.gate_up_proj"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Mistral4StateDictAdapter (text-only)
# ---------------------------------------------------------------------------


class TestMistral4StateDictAdapter:
    def test_strip_prefix(self, text_adapter):
        sd = {"language_model.model.layers.0.weight": torch.randn(4)}
        result = text_adapter._strip_prefix(sd)
        assert "model.layers.0.weight" in result
        assert "language_model.model.layers.0.weight" not in result

    def test_strip_prefix_no_prefix(self, text_adapter):
        sd = {"model.layers.0.weight": torch.randn(4)}
        result = text_adapter._strip_prefix(sd)
        assert "model.layers.0.weight" in result

    def test_from_hf_pipeline(self, text_adapter):
        """Full pipeline: strip prefix + dequant + expert conversion."""
        sd = {
            "language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
            "language_model.model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 64).to(torch.float8_e4m3fn),
            "language_model.model.layers.0.mlp.experts.gate_up_proj_scale_inv": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "language_model.model.layers.0.mlp.experts.gate_up_proj_activation_scale": torch.tensor(
                [1.0, 1.0, 1.0, 1.0]
            ),
            "language_model.model.layers.0.mlp.experts.down_proj": torch.randn(4, 64, 32).to(torch.float8_e4m3fn),
            "language_model.model.layers.0.mlp.experts.down_proj_scale_inv": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "language_model.model.layers.0.mlp.experts.down_proj_activation_scale": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        }
        result = text_adapter.from_hf(sd)
        # Prefix stripped
        assert any(k.startswith("model.layers.") for k in result)
        # Experts converted
        assert "model.layers.0.mlp.experts.gate_and_up_projs" in result
        assert "model.layers.0.mlp.experts.down_projs" in result
        # Scale keys removed
        assert not any("_scale_inv" in k for k in result)
        assert not any("_activation_scale" in k for k in result)

    def test_from_hf_injects_gate_bias(self, text_adapter):
        """from_hf injects e_score_correction_bias when missing from checkpoint."""
        sd = {
            "language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
            "language_model.model.layers.0.mlp.gate.weight": torch.randn(4, 64),
        }
        result = text_adapter.from_hf(sd)
        assert "model.layers.0.mlp.gate.e_score_correction_bias" in result
        assert result["model.layers.0.mlp.gate.e_score_correction_bias"].shape == (4,)

    def test_to_hf_expert_conversion(self, text_adapter):
        """to_hf reverses expert key names and adds prefix."""
        tensor = torch.randn(4, 64, 64)
        result = text_adapter.convert_single_tensor_to_hf("model.layers.0.mlp.experts.gate_and_up_projs", tensor)
        assert len(result) == 1
        key, val = result[0]
        assert key == "language_model.model.layers.0.mlp.experts.gate_up_proj"
        assert val.shape == (4, 64, 64)  # transposed back

    def test_to_hf_regular_key(self, text_adapter):
        """Regular keys just get prefix added."""
        tensor = torch.randn(64)
        result = text_adapter.convert_single_tensor_to_hf("model.layers.0.input_layernorm.weight", tensor)
        assert len(result) == 1
        assert result[0][0] == "language_model.model.layers.0.input_layernorm.weight"

    def test_to_hf_exclude_regex(self, text_adapter):
        """exclude_key_regex filters out matching keys."""
        tensor = torch.randn(64)
        result = text_adapter.convert_single_tensor_to_hf(
            "model.layers.0.self_attn.q_a_proj._extra_state",
            tensor,
            exclude_key_regex=r".*_extra_state.*",
        )
        assert len(result) == 0

    def test_to_hf_quantization_creates_fp8(self, text_adapter):
        """quantization=True creates FP8 + scale_inv + activation_scale entries."""
        tensor = torch.randn(64, 64)
        result = text_adapter.convert_single_tensor_to_hf(
            "model.layers.0.self_attn.q_a_proj.weight",
            tensor,
            quantization=True,
        )
        keys = [k for k, _ in result]
        assert "language_model.model.layers.0.self_attn.q_a_proj.weight" in keys
        assert "language_model.model.layers.0.self_attn.q_a_proj.weight_scale_inv" in keys
        assert "language_model.model.layers.0.self_attn.q_a_proj.activation_scale" in keys
        # FP8 dtype
        for k, v in result:
            if k.endswith(".weight") and "scale" not in k:
                assert v.dtype == torch.float8_e4m3fn

    def test_to_hf_quantization_skips_layernorm(self, text_adapter):
        """quantization=True skips non-quantized keys."""
        tensor = torch.randn(64)
        result = text_adapter.convert_single_tensor_to_hf(
            "model.layers.0.input_layernorm.weight",
            tensor,
            quantization=True,
        )
        assert len(result) == 1
        assert result[0][1].dtype != torch.float8_e4m3fn

    def test_to_hf_quantization_expert_3d(self, text_adapter):
        """quantization=True for expert keys creates [n_experts,1,1] scale_inv and [n_experts] act_scale."""
        tensor = torch.randn(4, 64, 64)
        result = text_adapter.convert_single_tensor_to_hf(
            "model.layers.0.mlp.experts.gate_and_up_projs",
            tensor,
            quantization=True,
        )
        keys = {k: v for k, v in result}
        scale_key = "language_model.model.layers.0.mlp.experts.gate_up_proj_scale_inv"
        act_key = "language_model.model.layers.0.mlp.experts.gate_up_proj_activation_scale"
        assert scale_key in keys
        assert keys[scale_key].shape == (4, 1, 1)
        assert act_key in keys
        assert keys[act_key].shape == (4,)

    def test_to_hf_quantization_drops_gate_bias(self, text_adapter):
        """quantization=True drops e_score_correction_bias keys."""
        tensor = torch.zeros(4)
        result = text_adapter.convert_single_tensor_to_hf(
            "model.layers.0.mlp.gate.e_score_correction_bias",
            tensor,
            quantization=True,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Mistral4MultimodalStateDictAdapter
# ---------------------------------------------------------------------------


class TestMistral4MultimodalStateDictAdapter:
    def test_remap_language_model_keys(self, mm_adapter):
        sd = {"language_model.model.layers.0.weight": torch.randn(4)}
        result = mm_adapter._remap_keys_from_hf(sd)
        assert "model.language_model.model.layers.0.weight" in result

    def test_remap_lm_head(self, mm_adapter):
        sd = {"language_model.lm_head.weight": torch.randn(4)}
        result = mm_adapter._remap_keys_from_hf(sd)
        assert "model.language_model.lm_head.weight" in result

    def test_remap_vision_tower(self, mm_adapter):
        sd = {"vision_tower.patch_conv.weight": torch.randn(4)}
        result = mm_adapter._remap_keys_from_hf(sd)
        assert "model.vision_tower.patch_conv.weight" in result

    def test_remap_projector(self, mm_adapter):
        sd = {"multi_modal_projector.linear_1.weight": torch.randn(4)}
        result = mm_adapter._remap_keys_from_hf(sd)
        assert "model.multi_modal_projector.linear_1.weight" in result

    def test_remap_to_hf_roundtrip(self, mm_adapter):
        """from_hf key mapping is reversible via to_hf key mapping."""
        original_keys = [
            "language_model.model.layers.0.self_attn.q_a_proj.weight",
            "language_model.lm_head.weight",
            "vision_tower.ln_pre.weight",
            "multi_modal_projector.linear_1.weight",
        ]
        for orig_key in original_keys:
            sd = {orig_key: torch.randn(4)}
            remapped = mm_adapter._remap_keys_from_hf(sd)
            native_key = list(remapped.keys())[0]
            restored = mm_adapter._remap_keys_to_hf(native_key)
            assert restored == orig_key, f"{orig_key} -> {native_key} -> {restored}"

    def test_from_hf_full_pipeline(self, mm_adapter):
        """Full pipeline: remap + dequant + expert conversion."""
        sd = {
            "language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
            "language_model.model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 64).to(torch.float8_e4m3fn),
            "language_model.model.layers.0.mlp.experts.gate_up_proj_scale_inv": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "language_model.lm_head.weight": torch.randn(128, 64),
            "vision_tower.patch_conv.weight": torch.randn(64, 3, 14, 14),
        }
        result = mm_adapter.from_hf(sd)
        assert "model.language_model.model.layers.0.input_layernorm.weight" in result
        assert "model.language_model.model.layers.0.mlp.experts.gate_and_up_projs" in result
        assert "model.language_model.lm_head.weight" in result
        assert "model.vision_tower.patch_conv.weight" in result

    def test_from_hf_injects_gate_bias(self, mm_adapter):
        """from_hf injects e_score_correction_bias when missing from checkpoint."""
        sd = {
            "language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
            "language_model.model.layers.0.mlp.gate.weight": torch.randn(4, 64),
        }
        result = mm_adapter.from_hf(sd)
        bias_key = "model.language_model.model.layers.0.mlp.gate.e_score_correction_bias"
        assert bias_key in result
        assert result[bias_key].shape == (4,)

    def test_to_hf_quantization_skips_vision(self, mm_adapter):
        """quantization=True skips vision tower and projector."""
        sd = {
            "model.vision_tower.patch_conv.weight": torch.randn(64, 3, 14, 14),
            "model.multi_modal_projector.linear_1.weight": torch.randn(64, 64),
        }
        hf_sd = mm_adapter.to_hf(sd, quantization=True)
        for key, val in hf_sd.items():
            assert val.dtype != torch.float8_e4m3fn, f"{key} should not be quantized"

    def test_to_hf_full(self, mm_adapter):
        """to_hf converts all keys back to HF format."""
        sd = {
            "model.language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
            "model.language_model.model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 64),
            "model.language_model.model.layers.0.mlp.experts.down_projs": torch.randn(4, 32, 64),
            "model.language_model.lm_head.weight": torch.randn(128, 64),
            "model.vision_tower.patch_conv.weight": torch.randn(64, 3, 14, 14),
            "model.multi_modal_projector.linear_1.weight": torch.randn(64, 64),
        }
        hf_sd = mm_adapter.to_hf(sd)
        assert "language_model.model.layers.0.input_layernorm.weight" in hf_sd
        assert "language_model.model.layers.0.mlp.experts.gate_up_proj" in hf_sd
        assert "language_model.model.layers.0.mlp.experts.down_proj" in hf_sd
        assert "language_model.lm_head.weight" in hf_sd
        assert "vision_tower.patch_conv.weight" in hf_sd
        assert "multi_modal_projector.linear_1.weight" in hf_sd

    def test_to_hf_exclude_regex(self, mm_adapter):
        """to_hf with exclude_key_regex filters matching keys."""
        sd = {
            "model.language_model.model.layers.0.self_attn.q_a_proj._extra_state": torch.randn(4),
            "model.language_model.model.layers.0.input_layernorm.weight": torch.randn(64),
        }
        hf_sd = mm_adapter.to_hf(sd, exclude_key_regex=r".*_extra_state.*")
        assert len(hf_sd) == 1
        assert "language_model.model.layers.0.input_layernorm.weight" in hf_sd

    def test_to_hf_quantization_creates_fp8_for_text(self, mm_adapter):
        """quantization=True creates FP8 + scale_inv for text model weights."""
        sd = {
            "model.language_model.model.layers.0.self_attn.q_a_proj.weight": torch.randn(64, 64),
        }
        hf_sd = mm_adapter.to_hf(sd, quantization=True)
        keys = list(hf_sd.keys())
        assert "language_model.model.layers.0.self_attn.q_a_proj.weight" in keys
        assert "language_model.model.layers.0.self_attn.q_a_proj.weight_scale_inv" in keys
        assert "language_model.model.layers.0.self_attn.q_a_proj.activation_scale" in keys
        assert hf_sd["language_model.model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.float8_e4m3fn

    def test_to_hf_quantization_expert_3d(self, mm_adapter):
        """quantization=True for expert keys creates [n_experts,1,1] scale_inv and [n_experts] act_scale."""
        sd = {
            "model.language_model.model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 64),
        }
        hf_sd = mm_adapter.to_hf(sd, quantization=True)
        scale_key = "language_model.model.layers.0.mlp.experts.gate_up_proj_scale_inv"
        act_key = "language_model.model.layers.0.mlp.experts.gate_up_proj_activation_scale"
        assert scale_key in hf_sd
        assert hf_sd[scale_key].shape == (4, 1, 1)
        assert act_key in hf_sd
        assert hf_sd[act_key].shape == (4,)

    def test_to_hf_quantization_drops_gate_bias(self, mm_adapter):
        """quantization=True drops e_score_correction_bias keys."""
        sd = {
            "model.language_model.model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros(4),
        }
        hf_sd = mm_adapter.to_hf(sd, quantization=True)
        assert len(hf_sd) == 0

    def test_to_hf_quantization_non_weight_suffix(self, mm_adapter):
        """quantization with non-.weight key uses key + _activation_scale."""
        sd = {
            "model.language_model.model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 64),
        }
        hf_sd = mm_adapter.to_hf(sd, quantization=True)
        # gate_up_proj doesn't end with .weight, so activation_scale uses _activation_scale suffix
        act_key = "language_model.model.layers.0.mlp.experts.gate_up_proj_activation_scale"
        assert act_key in hf_sd

    def test_remap_unknown_key_passthrough(self, mm_adapter):
        """Keys not matching any prefix pattern pass through unchanged."""
        sd = {"some_unknown_key.weight": torch.randn(4)}
        result = mm_adapter._remap_keys_from_hf(sd)
        assert "some_unknown_key.weight" in result

    def test_remap_to_hf_unknown_key(self, mm_adapter):
        """_remap_keys_to_hf passes unknown keys through unchanged."""
        result = mm_adapter._remap_keys_to_hf("some_unknown_key.weight")
        assert result == "some_unknown_key.weight"

    def test_remap_to_hf_language_model_prefix(self, mm_adapter):
        """_remap_keys_to_hf handles model.language_model. prefix (non-model/non-lm_head)."""
        result = mm_adapter._remap_keys_to_hf("model.language_model.some_attr.weight")
        assert result == "language_model.some_attr.weight"


# ---------------------------------------------------------------------------
# Mistral4StateDictAdapter.to_hf (full dict)
# ---------------------------------------------------------------------------


class TestMistral4StateDictAdapterToHf:
    def test_to_hf_full_dict(self, text_adapter):
        """to_hf converts a full dict of native keys to HF format."""
        sd = {
            "model.layers.0.input_layernorm.weight": torch.randn(64),
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 64),
            "model.layers.0.mlp.experts.down_projs": torch.randn(4, 32, 64),
            "lm_head.weight": torch.randn(128, 64),
        }
        hf_sd = text_adapter.to_hf(sd)
        assert "language_model.model.layers.0.input_layernorm.weight" in hf_sd
        assert "language_model.model.layers.0.mlp.experts.gate_up_proj" in hf_sd
        assert "language_model.model.layers.0.mlp.experts.down_proj" in hf_sd
        assert "language_model.lm_head.weight" in hf_sd

    def test_to_hf_with_exclude_and_quantization(self, text_adapter):
        """to_hf with both exclude_regex and quantization."""
        sd = {
            "model.layers.0.self_attn.q_a_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.q_a_proj._extra_state": torch.randn(4),
        }
        hf_sd = text_adapter.to_hf(sd, exclude_key_regex=r".*_extra_state.*", quantization=True)
        assert not any("_extra_state" in k for k in hf_sd)
        assert "language_model.model.layers.0.self_attn.q_a_proj.weight" in hf_sd
        assert hf_sd["language_model.model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.float8_e4m3fn

    def test_to_hf_down_projs_conversion(self, text_adapter):
        """to_hf converts down_projs back to down_proj with transpose."""
        tensor = torch.randn(4, 32, 64)
        result = text_adapter.convert_single_tensor_to_hf("model.layers.0.mlp.experts.down_projs", tensor)
        assert len(result) == 1
        key, val = result[0]
        assert key == "language_model.model.layers.0.mlp.experts.down_proj"
        assert val.shape == (4, 64, 32)  # transposed
