# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for FP8 model + PEFT dequantization logic."""

from unittest.mock import MagicMock

import pytest

from nemo_automodel._transformers.auto_model import _maybe_dequantize_fp8_for_peft


# ---------------------------------------------------------------------------
# Tests: FP8 + PEFT auto-dequantize
# ---------------------------------------------------------------------------


class TestFP8PeftDequantize:
    """Tests for auto-dequantization of FP8 models when PEFT is requested."""

    def test_fp8_model_with_peft_sets_dequantize_true(self):
        """When model has fp8 quantization_config and peft_config is set, dequantize=True."""
        quant_cfg = {
            "quant_method": "fp8",
            "dequantize": False,
            "activation_scheme": "static",
        }
        peft_config = MagicMock()

        result = _maybe_dequantize_fp8_for_peft(quant_cfg, peft_config, "some-model")

        assert result is True
        assert quant_cfg["dequantize"] is True

    def test_fp8_model_without_peft_does_not_dequantize(self):
        """When peft_config is None, FP8 model should NOT be dequantized."""
        quant_cfg = {
            "quant_method": "fp8",
            "dequantize": False,
        }

        result = _maybe_dequantize_fp8_for_peft(quant_cfg, peft_config=None, pretrained_path="some-model")

        assert result is False
        assert quant_cfg["dequantize"] is False

    def test_non_fp8_model_with_peft_does_not_dequantize(self):
        """When model uses non-FP8 quantization (e.g. GPTQ), should NOT set dequantize."""
        quant_cfg = {
            "quant_method": "gptq",
            "bits": 4,
        }
        peft_config = MagicMock()

        result = _maybe_dequantize_fp8_for_peft(quant_cfg, peft_config, "some-model")

        assert result is False
        assert "dequantize" not in quant_cfg

    def test_no_quantization_config_with_peft(self):
        """When quantization_config is None, should be a no-op."""
        peft_config = MagicMock()

        result = _maybe_dequantize_fp8_for_peft(None, peft_config, "some-model")

        assert result is False

    def test_non_string_pretrained_path_does_not_dequantize(self):
        """When pretrained_path is not a string (e.g. a config object), should not dequantize."""
        quant_cfg = {
            "quant_method": "fp8",
            "dequantize": False,
        }
        peft_config = MagicMock()

        result = _maybe_dequantize_fp8_for_peft(quant_cfg, peft_config, pretrained_path=MagicMock())

        assert result is False
        assert quant_cfg["dequantize"] is False

    def test_fp8_dequantize_preserves_other_quant_fields(self):
        """Dequantize should only add/modify 'dequantize', not touch other fields."""
        quant_cfg = {
            "quant_method": "fp8",
            "dequantize": False,
            "activation_scheme": "static",
            "modules_to_not_convert": ["lm_head", "vision_tower"],
            "weight_block_size": None,
        }
        peft_config = MagicMock()

        _maybe_dequantize_fp8_for_peft(quant_cfg, peft_config, "some-model")

        assert quant_cfg["dequantize"] is True
        assert quant_cfg["activation_scheme"] == "static"
        assert quant_cfg["modules_to_not_convert"] == ["lm_head", "vision_tower"]
        assert quant_cfg["weight_block_size"] is None
        assert quant_cfg["quant_method"] == "fp8"


# ---------------------------------------------------------------------------
# Tests: is_meta_device with native quantization config
# ---------------------------------------------------------------------------


class TestMetaDeviceWithNativeQuantConfig:
    """Tests for is_meta_device logic accounting for native HF quantization config."""

    @staticmethod
    def _compute_is_meta_device(model_wrapper, world_size, is_hf_model, quantization_config, hf_native_quant_cfg):
        """Replicate the is_meta_device logic from _build_model."""
        from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
        from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager
        from nemo_automodel.components.distributed.ddp import DDPManager

        return all(
            [
                not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager)),
                world_size > 1 or not is_hf_model,
                quantization_config is None and hf_native_quant_cfg is None,
            ]
        )

    def test_meta_device_disabled_when_hf_native_quant_config_present(self):
        """Meta device init should be disabled when model has native quantization config."""
        quant_cfg = {"quant_method": "fp8"}
        result = self._compute_is_meta_device(
            model_wrapper=None,
            world_size=2,
            is_hf_model=True,
            quantization_config=None,
            hf_native_quant_cfg=quant_cfg,
        )
        assert result is False

    def test_meta_device_disabled_when_user_quantization_config_present(self):
        """Meta device init should be disabled when user provides BNB quantization_config."""
        bnb_config = MagicMock()  # BitsAndBytesConfig
        result = self._compute_is_meta_device(
            model_wrapper=None,
            world_size=2,
            is_hf_model=True,
            quantization_config=bnb_config,
            hf_native_quant_cfg=None,
        )
        assert result is False

    def test_meta_device_enabled_when_no_quantization(self):
        """Meta device init should be enabled when no quantization is used (multi-GPU)."""
        result = self._compute_is_meta_device(
            model_wrapper=None,
            world_size=2,
            is_hf_model=True,
            quantization_config=None,
            hf_native_quant_cfg=None,
        )
        assert result is True

    def test_meta_device_disabled_single_gpu_hf_model(self):
        """Meta device init should be disabled for single-GPU HF model (no quantization)."""
        result = self._compute_is_meta_device(
            model_wrapper=None,
            world_size=1,
            is_hf_model=True,
            quantization_config=None,
            hf_native_quant_cfg=None,
        )
        assert result is False
