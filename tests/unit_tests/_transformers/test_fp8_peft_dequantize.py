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

"""Tests for FP8 model + PEFT dequantization logic in _build_model."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_automodel._transformers.model_init import get_hf_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hf_config(quantization_config=None):
    """Create a mock HF config with optional quantization_config."""
    cfg = SimpleNamespace()
    if quantization_config is not None:
        cfg.quantization_config = quantization_config
    return cfg


def _run_fp8_dequantize_logic(hf_config, peft_config, pretrained_path="some-model"):
    """
    Replicate the FP8 dequantization logic from _build_model in isolation.

    Returns the (possibly mutated) hf_config and kwargs dict.
    """
    kwargs = {}
    attn_implementation = "eager"

    # Replicate the logic from _build_model
    _hf_config = hf_config
    _hf_native_quant_cfg = getattr(_hf_config, "quantization_config", None)
    if peft_config is not None and isinstance(pretrained_path, str):
        if isinstance(_hf_native_quant_cfg, dict) and _hf_native_quant_cfg.get("quant_method") == "fp8":
            _hf_native_quant_cfg["dequantize"] = True
            kwargs["config"] = _hf_config

    return _hf_config, _hf_native_quant_cfg, kwargs


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
        hf_config = _make_hf_config(quant_cfg)
        peft_config = MagicMock()

        _, quant_result, kwargs = _run_fp8_dequantize_logic(hf_config, peft_config)

        assert quant_result["dequantize"] is True
        assert "config" in kwargs
        assert kwargs["config"] is hf_config

    def test_fp8_model_without_peft_does_not_dequantize(self):
        """When peft_config is None, FP8 model should NOT be dequantized."""
        quant_cfg = {
            "quant_method": "fp8",
            "dequantize": False,
        }
        hf_config = _make_hf_config(quant_cfg)

        _, quant_result, kwargs = _run_fp8_dequantize_logic(hf_config, peft_config=None)

        assert quant_result["dequantize"] is False
        assert "config" not in kwargs

    def test_non_fp8_model_with_peft_does_not_dequantize(self):
        """When model uses non-FP8 quantization (e.g. GPTQ), should NOT set dequantize."""
        quant_cfg = {
            "quant_method": "gptq",
            "bits": 4,
        }
        hf_config = _make_hf_config(quant_cfg)
        peft_config = MagicMock()

        _, quant_result, kwargs = _run_fp8_dequantize_logic(hf_config, peft_config)

        assert "dequantize" not in quant_result
        assert "config" not in kwargs

    def test_model_without_quantization_config_with_peft(self):
        """When model has no quantization_config at all, should be a no-op."""
        hf_config = _make_hf_config()  # no quantization_config
        peft_config = MagicMock()

        _, quant_result, kwargs = _run_fp8_dequantize_logic(hf_config, peft_config)

        assert quant_result is None
        assert "config" not in kwargs

    def test_qlora_model_with_peft_does_not_trigger_fp8_dequantize(self):
        """QLoRA models (BNB quantized) should NOT trigger FP8 dequantization.

        QLoRA models don't have quant_method='fp8' in their HF config.
        The BNB quantization_config is a separate user-provided parameter.
        """
        # QLoRA model has no quantization_config in HF config
        hf_config = _make_hf_config()
        peft_config = MagicMock()

        _, quant_result, kwargs = _run_fp8_dequantize_logic(hf_config, peft_config)

        assert quant_result is None
        assert "config" not in kwargs

    def test_fp8_dequantize_preserves_other_quant_fields(self):
        """Dequantize should only add/modify 'dequantize', not touch other fields."""
        quant_cfg = {
            "quant_method": "fp8",
            "dequantize": False,
            "activation_scheme": "static",
            "modules_to_not_convert": ["lm_head", "vision_tower"],
            "weight_block_size": None,
        }
        hf_config = _make_hf_config(quant_cfg)
        peft_config = MagicMock()

        _, quant_result, _ = _run_fp8_dequantize_logic(hf_config, peft_config)

        assert quant_result["dequantize"] is True
        assert quant_result["activation_scheme"] == "static"
        assert quant_result["modules_to_not_convert"] == ["lm_head", "vision_tower"]
        assert quant_result["weight_block_size"] is None
        assert quant_result["quant_method"] == "fp8"


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
