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

"""Unit tests for the Mistral3 FP8 VLM state-dict adapter."""

import pytest
import torch

from nemo_automodel.components.models.mistral3_vlm.state_dict_adapter import (
    Mistral3FP8StateDictAdapter,
    _NON_QUANTIZED_SUFFIXES,
    _dequantize_from_fp8,
    _is_fp8_weight_key,
)


# --------------------------------------------------------------------------- #
# _is_fp8_weight_key                                                          #
# --------------------------------------------------------------------------- #
class TestIsFp8WeightKey:
    """Gates whether a key names an FP8-stored Linear weight."""

    @pytest.mark.parametrize(
        "key",
        [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.42.mlp.down_proj.weight",
        ],
    )
    def test_layer_linear_weights_are_fp8(self, key):
        assert _is_fp8_weight_key(key) is True

    @pytest.mark.parametrize("suffix", _NON_QUANTIZED_SUFFIXES)
    def test_non_quantized_suffixes_excluded(self, suffix):
        # Build a plausible full key ending in the suffix.
        assert _is_fp8_weight_key(f"some.parent.{suffix}") is False

    def test_non_weight_keys_excluded(self):
        assert _is_fp8_weight_key("model.layers.0.self_attn.q_proj.bias") is False
        assert _is_fp8_weight_key("model.language_model.layers.0.input_layernorm") is False

    def test_not_fp8_prefixes_excluded_exact(self):
        assert (
            _is_fp8_weight_key(
                "model.vision_tower",
                not_fp8_prefixes=("model.vision_tower",),
            )
            is False
        )

    def test_not_fp8_prefixes_excluded_descendant(self):
        # Prefix-with-dot match — should exclude any descendant.
        assert (
            _is_fp8_weight_key(
                "model.vision_tower.transformer.layers.0.self_attn.q_proj.weight",
                not_fp8_prefixes=("model.vision_tower",),
            )
            is False
        )
        assert (
            _is_fp8_weight_key(
                "model.multi_modal_projector.linear_1.weight",
                not_fp8_prefixes=("model.vision_tower", "model.multi_modal_projector"),
            )
            is False
        )

    def test_not_fp8_prefix_is_not_substring_match(self):
        # "model.vision_tower_other" should NOT be excluded by "model.vision_tower" prefix.
        assert (
            _is_fp8_weight_key(
                "model.vision_tower_other.weight",
                not_fp8_prefixes=("model.vision_tower",),
            )
            is True
        )


# --------------------------------------------------------------------------- #
# _dequantize_from_fp8                                                        #
# --------------------------------------------------------------------------- #
class TestDequantizeFromFp8:
    """w_bf16 = w_fp8.to(bf16) * scale_inv.to(bf16)."""

    def test_per_tensor_scale_multiply(self):
        # FP8 e4m3 has limited precision; pick exact-representable values.
        w_fp8 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
        scale = torch.tensor(0.5, dtype=torch.bfloat16)
        out = _dequantize_from_fp8(w_fp8, scale, target_dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16
        expected = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.bfloat16)
        assert torch.equal(out, expected)

    def test_target_dtype_float32(self):
        w_fp8 = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        scale = torch.tensor(2.0, dtype=torch.bfloat16)
        out = _dequantize_from_fp8(w_fp8, scale, target_dtype=torch.float32)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.tensor([2.0, 4.0]))


# --------------------------------------------------------------------------- #
# Mistral3FP8StateDictAdapter — factories and identity rewrites               #
# --------------------------------------------------------------------------- #
class TestForVlmFullFactory:
    """The single shipped factory wires layout name and not_fp8_prefixes."""

    def test_layout_name(self):
        a = Mistral3FP8StateDictAdapter.for_vlm_full()
        assert a._layout_name == "vlm_full"

    def test_not_fp8_prefixes(self):
        a = Mistral3FP8StateDictAdapter.for_vlm_full()
        assert a._not_fp8_prefixes == (
            "model.vision_tower",
            "model.multi_modal_projector",
        )

    def test_keys_round_trip_identity(self):
        # Both rewrites are _identity by default — keys should pass through.
        a = Mistral3FP8StateDictAdapter.for_vlm_full()
        for k in (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.patch_conv.weight",
            "lm_head.weight",
        ):
            assert a._native_to_hf(k) == k
            assert a._hf_to_native(k) == k


# --------------------------------------------------------------------------- #
# from_hf                                                                     #
# --------------------------------------------------------------------------- #
class TestFromHf:
    """from_hf dequantizes FP8 weights against scale_inv siblings and drops scale keys."""

    def _adapter(self):
        return Mistral3FP8StateDictAdapter.for_vlm_full()

    def test_fp8_weight_dequantizes_with_scale(self):
        a = self._adapter()
        w_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        sd = {
            w_key: torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn),
            w_key + "_scale_inv": torch.tensor(0.5, dtype=torch.bfloat16),
        }
        out = a.from_hf(sd)
        # scale_inv key dropped
        assert w_key + "_scale_inv" not in out
        # weight dequanted to bf16
        assert out[w_key].dtype == torch.bfloat16
        assert torch.equal(
            out[w_key], torch.tensor([[0.5, 1.0]], dtype=torch.bfloat16)
        )

    def test_bf16_weight_passes_through(self):
        a = self._adapter()
        v = torch.tensor([[1.5, -2.0]], dtype=torch.bfloat16)
        sd = {"model.vision_tower.ln_pre.weight": v}
        out = a.from_hf(sd)
        assert torch.equal(out["model.vision_tower.ln_pre.weight"], v)

    def test_activation_scale_keys_dropped(self):
        a = self._adapter()
        sd = {
            "model.language_model.layers.0.self_attn.q_proj.activation_scale": torch.tensor(1.0),
            "lm_head.weight": torch.tensor([1.0]),
        }
        out = a.from_hf(sd)
        assert "model.language_model.layers.0.self_attn.q_proj.activation_scale" not in out
        assert "lm_head.weight" in out

    def test_fp8_weight_without_scale_passes_through_untouched(self):
        # Defensive: if no scale_inv sibling, don't dequant — pass through.
        a = self._adapter()
        w_key = "model.language_model.layers.0.mlp.up_proj.weight"
        v = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
        out = a.from_hf({w_key: v})
        assert out[w_key].dtype == torch.float8_e4m3fn


# --------------------------------------------------------------------------- #
# to_hf                                                                       #
# --------------------------------------------------------------------------- #
class TestToHf:
    """to_hf casts FP8 weights and emits scale_inv placeholders when quantization=True."""

    def _adapter(self):
        return Mistral3FP8StateDictAdapter.for_vlm_full()

    def test_quantization_off_passes_through(self):
        a = self._adapter()
        w_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        v = torch.zeros(2, 2, dtype=torch.bfloat16)
        out = a.to_hf({w_key: v}, quantization=False)
        assert out == {w_key: v}

    def test_quantization_emits_scale_inv_for_fp8_keys(self):
        a = self._adapter()
        w_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        out = a.to_hf({w_key: torch.zeros(2, 2, dtype=torch.bfloat16)}, quantization=True)
        assert w_key in out
        assert out[w_key].dtype == torch.float8_e4m3fn
        assert w_key + "_scale_inv" in out
        assert out[w_key + "_scale_inv"].dtype == torch.bfloat16
        # Scalar (0-d) placeholder
        assert out[w_key + "_scale_inv"].shape == ()

    def test_quantization_skips_placeholder_for_non_fp8_keys(self):
        a = self._adapter()
        v = torch.zeros(4, dtype=torch.bfloat16)
        # vision_tower + multi_modal_projector + lm_head + embed_tokens excluded.
        for key in (
            "model.vision_tower.transformer.layers.0.attention.q_proj.weight",
            "model.multi_modal_projector.linear_1.weight",
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
        ):
            out = a.to_hf({key: v}, quantization=True)
            assert key in out
            assert key + "_scale_inv" not in out
            # Original dtype preserved
            assert out[key].dtype == torch.bfloat16

    def test_exclude_key_regex(self):
        a = self._adapter()
        out = a.to_hf(
            {
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
                "lm_head.weight": torch.zeros(1),
            },
            exclude_key_regex=r"^lm_head\.",
        )
        assert "lm_head.weight" not in out
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out


# --------------------------------------------------------------------------- #
# convert_single_tensor_to_hf                                                  #
# --------------------------------------------------------------------------- #
class TestConvertSingleTensorToHf:
    """Per-tensor save path used by Checkpointer.save_model."""

    def _adapter(self):
        return Mistral3FP8StateDictAdapter.for_vlm_full()

    def test_no_quantization(self):
        a = self._adapter()
        t = torch.zeros(1)
        pairs = a.convert_single_tensor_to_hf(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            t,
        )
        assert len(pairs) == 1
        assert pairs[0][0] == "model.language_model.layers.0.self_attn.q_proj.weight"
        assert pairs[0][1] is t

    def test_quantization_fp8_emits_two_pairs(self):
        a = self._adapter()
        t = torch.zeros(1)
        fqn = "model.language_model.layers.0.self_attn.q_proj.weight"
        pairs = a.convert_single_tensor_to_hf(fqn, t, quantization=True)
        assert len(pairs) == 2
        assert pairs[0] == (fqn, t)
        assert pairs[1][0] == fqn + "_scale_inv"
        assert pairs[1][1].dtype == torch.bfloat16
        assert pairs[1][1].shape == ()

    def test_quantization_non_fp8_emits_one_pair(self):
        a = self._adapter()
        t = torch.zeros(1)
        for fqn in (
            "model.vision_tower.transformer.layers.0.attention.q_proj.weight",
            "model.multi_modal_projector.linear_1.weight",
            "lm_head.weight",
        ):
            pairs = a.convert_single_tensor_to_hf(fqn, t, quantization=True)
            assert len(pairs) == 1
            assert pairs[0] == (fqn, t)
