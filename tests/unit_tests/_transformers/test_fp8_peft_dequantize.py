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

from unittest.mock import MagicMock, patch

from nemo_automodel._transformers.auto_model import _BaseNeMoAutoModelClass, _maybe_dequantize_fp8_for_peft

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
        from nemo_automodel.components.distributed.ddp import DDPManager
        from nemo_automodel.components.distributed.megatron_fsdp import MegatronFSDPManager

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


# ---------------------------------------------------------------------------
# Tests: kwargs["config"] injection gated on is_hf_model (issue #2164)
# ---------------------------------------------------------------------------


class TestKwargsConfigInjectionGate:
    """Tests for the is_hf_model gate on kwargs["config"] = _hf_config injection.

    Custom models receive _hf_config positionally in model_init.py:783 via
    model_cls(hf_config, *model_args, **kwargs); injecting config into kwargs
    causes a TypeError ("got multiple values for argument 'config'"). The gate
    suppresses the injection for the custom-model path while preserving the
    in-place dequantize=True mutation needed by the HF path.
    """

    @staticmethod
    def _make_build_kwargs(is_hf_model):
        """Minimal kwargs for running _build_model through the FP8+PEFT gate."""
        mesh = MagicMock()
        mesh.tp_size = 1
        mesh.cp_size = 1
        return dict(
            is_hf_model=is_hf_model,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            sdpa_method=None,
            torch_dtype="auto",
            attn_implementation="eager",
            quantization_config=None,
            force_hf=False,
            model_wrapper=None,
            autopipeline=None,
            parallelize_fn=None,
            qat_quantizer=None,
            mesh=mesh,
            loss_fn=None,
            peft_config=MagicMock(),
            fp8_config=None,
            compile_config=None,
            load_base_model=True,
        )

    @staticmethod
    def _run_build_model_with_native_fp8(is_hf_model):
        quant_cfg = {"quant_method": "fp8", "dequantize": False}
        hf_config = MagicMock()
        hf_config.quantization_config = quant_cfg
        sentinel_model = MagicMock()

        with (
            patch("nemo_automodel._transformers.auto_model._apply_preload_overrides", return_value=("eager", False)),
            patch("nemo_automodel._transformers.auto_model.get_hf_config", return_value=hf_config),
            patch("nemo_automodel._transformers.auto_model._init_model") as mock_init,
            patch("nemo_automodel._transformers.auto_model.get_world_size_safe", return_value=1),
            patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"),
            patch(
                "nemo_automodel._transformers.capabilities.attach_capabilities_and_validate",
                return_value=sentinel_model,
            ),
            patch("nemo_automodel._transformers.auto_model.apply_model_infrastructure", return_value=sentinel_model),
            patch("torch.cuda.current_device", return_value=0),
        ):
            mock_init.return_value = (not is_hf_model, sentinel_model)
            result = _BaseNeMoAutoModelClass._build_model(
                "some-model",
                **TestKwargsConfigInjectionGate._make_build_kwargs(is_hf_model),
            )

        return quant_cfg, hf_config, result, sentinel_model, mock_init

    def test_build_model_hf_fp8_peft_injects_config_kwarg(self):
        """_build_model should pass mutated config through kwargs for HF from_pretrained."""
        quant_cfg, hf_config, result, sentinel_model, mock_init = self._run_build_model_with_native_fp8(
            is_hf_model=True
        )

        assert result is sentinel_model
        assert mock_init.call_args.kwargs["config"] is hf_config
        assert quant_cfg["dequantize"] is True

    def test_build_model_custom_fp8_peft_does_not_inject_config_kwarg(self):
        """_build_model should not pass duplicate config kwargs for custom model init."""
        quant_cfg, _hf_config, result, sentinel_model, mock_init = self._run_build_model_with_native_fp8(
            is_hf_model=False
        )

        assert result is sentinel_model
        assert "config" not in mock_init.call_args.kwargs
        assert quant_cfg["dequantize"] is True

    @staticmethod
    def _apply_gate(hf_native_quant_cfg, peft_config, pretrained_path, is_hf_model, hf_config_obj):
        """Replicate the gated kwargs["config"] injection from _build_model."""
        kwargs: dict = {}
        if _maybe_dequantize_fp8_for_peft(hf_native_quant_cfg, peft_config, pretrained_path):
            if is_hf_model:
                kwargs["config"] = hf_config_obj
        return kwargs

    def test_hf_model_fp8_peft_injects_config_kwarg(self):
        """HF path needs config in kwargs so HF.from_pretrained sees the dequantize mutation."""
        quant_cfg = {"quant_method": "fp8", "dequantize": False}
        hf_config = MagicMock()
        hf_config.quantization_config = quant_cfg

        kwargs = self._apply_gate(quant_cfg, MagicMock(), "some-model", is_hf_model=True, hf_config_obj=hf_config)

        assert "config" in kwargs
        assert kwargs["config"] is hf_config
        assert quant_cfg["dequantize"] is True

    def test_custom_model_fp8_peft_does_not_inject_config_kwarg(self):
        """Custom-model path receives hf_config positionally; injecting config would TypeError (#2164)."""
        quant_cfg = {"quant_method": "fp8", "dequantize": False}
        hf_config = MagicMock()
        hf_config.quantization_config = quant_cfg

        kwargs = self._apply_gate(quant_cfg, MagicMock(), "some-model", is_hf_model=False, hf_config_obj=hf_config)

        assert "config" not in kwargs
        # Dequantize mutation must still be applied so the custom path sees it via the
        # positional hf_config argument.
        assert quant_cfg["dequantize"] is True

    def test_no_peft_does_not_inject_regardless_of_is_hf_model(self):
        """When PEFT is not configured, no injection happens on either path."""
        quant_cfg = {"quant_method": "fp8", "dequantize": False}
        hf_config = MagicMock()

        kwargs_hf = self._apply_gate(quant_cfg, None, "some-model", is_hf_model=True, hf_config_obj=hf_config)
        kwargs_custom = self._apply_gate(quant_cfg, None, "some-model", is_hf_model=False, hf_config_obj=hf_config)

        assert "config" not in kwargs_hf
        assert "config" not in kwargs_custom
        assert quant_cfg["dequantize"] is False

    def test_non_fp8_quant_does_not_inject(self):
        """Non-FP8 quant configs (e.g. GPTQ) are not the FP8+PEFT case; no injection."""
        quant_cfg = {"quant_method": "gptq", "bits": 4}
        hf_config = MagicMock()

        kwargs = self._apply_gate(quant_cfg, MagicMock(), "some-model", is_hf_model=True, hf_config_obj=hf_config)

        assert "config" not in kwargs
