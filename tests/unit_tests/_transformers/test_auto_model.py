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

import logging
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel._transformers.auto_model import (
    _consume_config_overrides,
    _get_next_fallback_attn,
    _init_model,
    _patch_attention,
)
from nemo_automodel._transformers.infrastructure import _apply_peft_and_lower_precision
from nemo_automodel._transformers.model_init import _filter_kwargs_for_init, _get_mixin_wrapped_class
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin


class TestPatchAttention:
    """Test cases for _patch_attention function."""

    def test__patch_attention_basic(self):
        """Test basic _patch_attention functionality."""
        # Create a real object with a forward method to test the actual wrapping
        class DummyModule:
            def forward(self, x):
                """Dummy forward method."""
                return x * 2

        obj = DummyModule()
        original_forward = obj.forward

        with patch("nemo_automodel._transformers.kernel_patches.sdpa_kernel") as mock_sdpa_kernel:
            result = _patch_attention(obj)

            assert result is obj
            # Verify that the forward method was replaced
            assert obj.forward != original_forward
            # Verify the wrapper has the expected docstring prefix
            assert obj.forward.__doc__.startswith("SDPA kernel patch")

            # Call forward and verify sdpa_kernel was used as context manager
            output = obj.forward(5)
            assert output == 10  # Original forward logic still works
            mock_sdpa_kernel.assert_called_once()

    def test__patch_attention_with_custom_sdpa_method(self):
        """Test _patch_attention with custom SDPA method."""
        from torch.nn.attention import SDPBackend

        class DummyModule:
            def forward(self, x):
                """Dummy forward method."""
                return x + 1

        obj = DummyModule()
        custom_sdpa_method = [SDPBackend.FLASH_ATTENTION]

        with patch("nemo_automodel._transformers.kernel_patches.sdpa_kernel") as mock_sdpa_kernel:
            result = _patch_attention(obj, custom_sdpa_method)

            assert result is obj
            # Verify the wrapper has the expected docstring prefix
            assert obj.forward.__doc__.startswith("SDPA kernel patch")

            # Call forward and verify sdpa_kernel was called with the custom method
            output = obj.forward(5)
            assert output == 6  # Original forward logic still works
            mock_sdpa_kernel.assert_called_once_with(custom_sdpa_method)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_assert_same_signature_matching(self):
        """Test _assert_same_signature with matching signatures."""
        from nemo_automodel._transformers.kernel_patches import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, c=None):
            pass

        # Should not raise an exception
        _assert_same_signature(func1, func2)

    def test_assert_same_signature_different(self):
        """Test _assert_same_signature with different signatures."""
        from nemo_automodel._transformers.kernel_patches import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, d=None):
            pass

        # Should raise an AssertionError
        with pytest.raises(AssertionError):
            _assert_same_signature(func1, func2)

    def test_get_next_fallback_attn_valid_priorities(self):
        """Test _get_next_fallback_attn with valid attention implementations."""
        # Test fallback from highest to lowest priority
        assert _get_next_fallback_attn("flash_attention_3") == "flash_attention_2"
        assert _get_next_fallback_attn("flash_attention_2") == "sdpa"
        assert _get_next_fallback_attn("sdpa") == "eager"

        # Test that eager falls back to itself (lowest priority)
        assert _get_next_fallback_attn("eager") == "eager"

    def test_get_next_fallback_attn_invalid_implementations(self):
        """Test _get_next_fallback_attn with invalid/unknown attention implementations."""
        # Test various invalid implementations all fall back to eager
        assert _get_next_fallback_attn("flash_attention_1") == "eager"
        assert _get_next_fallback_attn("unknown_attention") == "eager"
        assert _get_next_fallback_attn("custom_attention") == "eager"
        assert _get_next_fallback_attn("") == "eager"
        assert _get_next_fallback_attn("none") == "eager"
        assert _get_next_fallback_attn("legacy_attention") == "eager"

    @pytest.mark.parametrize("attn_impl,expected", [
        ("flash_attention_3", "flash_attention_2"),
        ("flash_attention_2", "sdpa"),
        ("sdpa", "eager"),
        ("eager", "eager"),
        ("invalid", "eager"),
        ("custom_impl", "eager"),
        ("", "eager"),
    ])
    def test_get_next_fallback_attn_parametrized(self, attn_impl, expected):
        """Parametrized test for _get_next_fallback_attn covering all scenarios."""
        assert _get_next_fallback_attn(attn_impl) == expected

    def test_get_next_fallback_attn_edge_cases(self):
        """Test _get_next_fallback_attn with edge cases and special inputs."""
        # Test with None (should be treated as unknown)
        assert _get_next_fallback_attn(None) == "eager"

        # Test case sensitivity (should be treated as unknown since not exact match)
        assert _get_next_fallback_attn("EAGER") == "eager"
        assert _get_next_fallback_attn("Flash_Attention_2") == "eager"
        assert _get_next_fallback_attn("SDPA") == "eager"

        # Test with whitespace (should be treated as unknown)
        assert _get_next_fallback_attn(" eager ") == "eager"
        assert _get_next_fallback_attn("sdpa ") == "eager"

        # Test with numeric strings
        assert _get_next_fallback_attn("123") == "eager"
        assert _get_next_fallback_attn("0") == "eager"


class DummyModel(torch.nn.Module):
    """A tiny nn.Module that behaves enough like a HF/BERT style model."""

    def __init__(self):
        super().__init__()
        self.config = {}  # _patch_liger_kernel calls  model.config.update(...)
        self.called = False  # turned on by fake liger kernel

    def mark(self):
        self.called = True


def prepare_env(monkeypatch, target_mod, *, has_liger=True, apply_ok=True):
    """
    Patch every external symbol that _patch_liger_kernel touches.

    Parameters
    ----------
    has_liger : bool
        Value for HAS_LIGER_KERNEL global.
    apply_ok : bool
        Force liger_kernel_trf._apply_liger_kernel_to_instance to succeed/fail.
    """
    monkeypatch.setattr(target_mod, "HAS_LIGER_KERNEL", has_liger, raising=False)

    apply_mock = MagicMock()

    if apply_ok:
        # mark model when called so we can assert later
        apply_mock.side_effect = lambda *, model: model.mark()
    else:
        apply_mock.side_effect = RuntimeError("boom")

    liger_stub = types.SimpleNamespace(_apply_liger_kernel_to_instance=apply_mock)
    monkeypatch.setattr(target_mod, "liger_kernel_trf", liger_stub, raising=False)

    patch_attn_mock = MagicMock(side_effect=lambda *args, **kwargs: args[0])
    monkeypatch.setattr(target_mod, "_patch_attention", patch_attn_mock, raising=True)

    return apply_mock, patch_attn_mock


def test_patch_liger_kernel_success(monkeypatch):
    """Test _patch_liger_kernel successfully applies liger kernel when available."""
    import nemo_automodel._transformers.kernel_patches as tgt

    apply_mock, attn_mock = prepare_env(monkeypatch, tgt, has_liger=True, apply_ok=True)

    model = DummyModel()
    patched = tgt._patch_liger_kernel(model)

    # Returns same instance
    assert patched is model

    # Liger kernel was applied
    apply_mock.assert_called_once()
    assert model.called is True

    # SDPA not called inside _patch_liger_kernel (it's called separately)
    attn_mock.assert_not_called()



def test_liger_not_available(monkeypatch):
    """
    Asked for Liger but HAS_LIGER_KERNEL is False.
    Expect: return untouched model, _patch_attention still invoked,
            no exceptions thrown.
    """
    import nemo_automodel._transformers.kernel_patches as tgt

    apply_mock, attn_mock = prepare_env(
        monkeypatch,
        tgt,
        has_liger=False,  # unavailable
        apply_ok=True,
    )

    model = DummyModel()
    out = tgt._patch_liger_kernel(model)

    # untouched instance returned
    assert out is model
    assert model.called is False
    # _apply never called, because we short-circuit when HAS_LIGER_KERNEL==False
    apply_mock.assert_not_called()
    attn_mock.assert_not_called()


def test_liger_apply_failure_raises(monkeypatch):
    """
    If _apply_liger_kernel_to_instance throws, _patch_liger_kernel must
    clean up and raise RuntimeError.
    """
    import nemo_automodel._transformers.kernel_patches as tgt

    prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=False,  # force failure
    )

    with pytest.raises(RuntimeError, match="Failed to patch model"):
        tgt._patch_liger_kernel(DummyModel())


def test_patch_liger_kernel_skips_non_nn_module(monkeypatch, caplog):
    """
    When model is not an nn.Module (e.g., a lightweight mock), _patch_liger_kernel
    should skip patching and return the model unchanged with a warning.
    """
    import nemo_automodel._transformers.kernel_patches as tgt

    apply_mock, attn_mock = prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=True,
    )

    # Create a non-nn.Module mock object
    mock_model = MagicMock(spec=[])  # Empty spec means no nn.Module methods
    mock_model.config = {}

    with caplog.at_level(logging.WARNING):
        result = tgt._patch_liger_kernel(mock_model)

    # Should return the same mock unchanged
    assert result is mock_model
    # Liger kernel should NOT be applied
    apply_mock.assert_not_called()
    # Warning should be logged
    assert "Skipping Liger Kernel patch for non-nn.Module model" in caplog.text


# =============================================================================
# Tests for _get_mixin_wrapped_class
# =============================================================================

class TestGetMixinWrappedClass:
    """Test cases for _get_mixin_wrapped_class function."""

    def test_returns_original_if_already_has_mixin(self):
        """When model class already inherits from HFCheckpointingMixin, return it unchanged."""
        class ModelWithMixin(HFCheckpointingMixin, torch.nn.Module):
            pass

        result = _get_mixin_wrapped_class(ModelWithMixin)
        assert result is ModelWithMixin

    def test_creates_wrapper_for_hf_class_with_correct_attributes(self):
        """For HF model classes, create a wrapper inheriting from both and preserving attributes."""
        class PlainModel(torch.nn.Module):
            pass

        result = _get_mixin_wrapped_class(PlainModel)

        # Should be a new class inheriting from both
        assert result is not PlainModel
        assert issubclass(result, HFCheckpointingMixin)
        assert issubclass(result, PlainModel)
        # Should preserve original class attributes
        assert result.__module__ == PlainModel.__module__
        assert result.__qualname__ == PlainModel.__qualname__
        assert result.__name__ == PlainModel.__name__


# =============================================================================
# Tests for _apply_peft_and_lower_precision
# =============================================================================

class TestApplyPeftAndLowerPrecision:
    """Test cases for _apply_peft_and_lower_precision function."""

    def test_apply_peft_disables_triton_with_tp(self, caplog):
        """When tp_size > 1, sets peft_config.use_triton = False."""
        mock_model = MagicMock()
        mock_peft_config = MagicMock()
        mock_peft_config.use_triton = True

        with (
            patch("nemo_automodel._transformers.infrastructure.apply_lora_to_linear_modules") as mock_apply_lora,
            caplog.at_level(logging.INFO),
        ):
            result = _apply_peft_and_lower_precision(
                mock_model,
                tp_size=2,  # TP > 1
                autopipeline=None,
                peft_config=mock_peft_config,
                quantization_config=None,
                fp8_config=None,
                qat_quantizer=None,
            )

            assert mock_peft_config.use_triton is False
            assert "Disabling Triton with TP" in caplog.text
            mock_apply_lora.assert_called_once()

    def test_apply_peft_disables_triton_with_autopipeline(self, caplog):
        """When autopipeline is not None, disables Triton."""
        mock_model = MagicMock()
        mock_peft_config = MagicMock()
        mock_peft_config.use_triton = True
        mock_autopipeline = MagicMock()

        with (
            patch("nemo_automodel._transformers.infrastructure.apply_lora_to_linear_modules") as mock_apply_lora,
            caplog.at_level(logging.INFO),
        ):
            result = _apply_peft_and_lower_precision(
                mock_model,
                tp_size=1,
                autopipeline=mock_autopipeline,  # PP enabled
                peft_config=mock_peft_config,
                quantization_config=None,
                fp8_config=None,
                qat_quantizer=None,
            )

            assert mock_peft_config.use_triton is False
            assert "Disabling Triton with Pipeline Parallelism" in caplog.text

    def test_apply_fp8_when_configured(self):
        """When fp8_config provided, calls apply_fp8_to_model."""
        mock_model = MagicMock()
        mock_fp8_config = MagicMock()

        with patch("nemo_automodel._transformers.infrastructure.apply_fp8_to_model") as mock_apply_fp8:
            mock_apply_fp8.return_value = mock_model

            result = _apply_peft_and_lower_precision(
                mock_model,
                tp_size=1,
                autopipeline=None,
                peft_config=None,
                quantization_config=None,
                fp8_config=mock_fp8_config,
                qat_quantizer=None,
            )

            mock_apply_fp8.assert_called_once_with(mock_model, config=mock_fp8_config)

    def test_apply_qat_when_configured(self):
        """When qat_quantizer provided, calls prepare_qat_model."""
        mock_model = MagicMock()
        # Ensure model parameters return bfloat16
        mock_param = MagicMock()
        mock_param.dtype = torch.bfloat16
        mock_model.parameters.return_value = [mock_param]

        mock_qat_quantizer = MagicMock()

        # prepare_qat_model is imported inside the function, so we need to patch it in its source module
        with patch("nemo_automodel.components.quantization.qat.prepare_qat_model") as mock_prepare_qat:
            mock_prepare_qat.return_value = (mock_model, "qat_mode")

            result = _apply_peft_and_lower_precision(
                mock_model,
                tp_size=1,
                autopipeline=None,
                peft_config=None,
                quantization_config=None,
                fp8_config=None,
                qat_quantizer=mock_qat_quantizer,
            )

            mock_prepare_qat.assert_called_once_with(mock_model, mock_qat_quantizer)
            assert hasattr(result, "_qat_mode")




# =============================================================================
# Tests for _consume_config_overrides and _filter_kwargs_for_init
# =============================================================================

class TestConsumeConfigOverrides:
    """Test cases for _consume_config_overrides function."""

    def test_consume_config_overrides_moves_config_keys_to_config(self):
        """Config-related kwargs are moved from kwargs dict to config object."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"output_hidden_states": True, "use_cache": True}

        kwargs = {"output_hidden_states": True, "some_other_arg": 42}

        _consume_config_overrides(mock_config, kwargs)

        # output_hidden_states should be moved to config
        assert "output_hidden_states" not in kwargs
        assert "some_other_arg" in kwargs
        mock_config.output_hidden_states = True  # Should be set on config

    def test_consume_config_overrides_preserves_init_param_names(self):
        """Keys that are in init_param_names are kept in kwargs."""
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"output_hidden_states": True}

        kwargs = {"output_hidden_states": True, "explicit_param": 42}

        _consume_config_overrides(mock_config, kwargs, init_param_names={"explicit_param", "output_hidden_states"})

        # Both should be kept since they're in init_param_names
        assert "output_hidden_states" in kwargs
        assert "explicit_param" in kwargs


class TestFilterKwargsForInit:
    """Test cases for _filter_kwargs_for_init function."""

    def test_filter_kwargs_for_init_removes_unknown_kwargs(self):
        """Filters out kwargs not in model __init__ signature."""
        class ModelWithSpecificInit:
            def __init__(self, config, a, b):
                pass

        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = _filter_kwargs_for_init(ModelWithSpecificInit, kwargs)

        assert "a" in result
        assert "b" in result
        assert "c" not in result
        assert "d" not in result

    def test_filter_kwargs_for_init_keeps_all_with_var_keyword(self):
        """If __init__ has **kwargs, returns all kwargs unchanged."""
        class ModelWithVarKwargs:
            def __init__(self, config, **kwargs):
                pass

        kwargs = {"a": 1, "b": 2, "c": 3}
        result = _filter_kwargs_for_init(ModelWithVarKwargs, kwargs)

        assert result == kwargs


# =============================================================================
# Tests for NEED_SETUP_CACHE_CLASSES_MAPPING backward compatibility shim
# =============================================================================


class TestNeedSetupCacheClassesMapping:
    """Test cases for the NEED_SETUP_CACHE_CLASSES_MAPPING backward compat shim."""

    def test_shim_does_not_overwrite_existing_attribute(self):
        """If NEED_SETUP_CACHE_CLASSES_MAPPING already exists, shim doesn't overwrite."""
        import importlib

        import transformers.generation.utils as gen_utils

        sentinel = {"test": "sentinel_value"}
        gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = sentinel

        # Re-import to trigger the shim code
        import nemo_automodel._transformers.auto_model as mod
        importlib.reload(mod)

        # The sentinel should still be there (shim didn't overwrite)
        assert gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING is sentinel

        # Clean up
        del gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING

    def test_shim_creates_attribute_when_missing(self):
        """If NEED_SETUP_CACHE_CLASSES_MAPPING is missing, shim creates it."""
        import importlib

        import transformers.generation.utils as gen_utils

        # Remove the attribute if it exists
        if hasattr(gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
            delattr(gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING")

        # Re-import to trigger the shim
        import nemo_automodel._transformers.auto_model as mod
        importlib.reload(mod)

        assert hasattr(gen_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING")
        mapping = gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING
        assert "static" in mapping

        # Clean up
        del gen_utils.NEED_SETUP_CACHE_CLASSES_MAPPING


# =============================================================================
# Tests for _model_mapping KeyError fallback in _init_model
# =============================================================================


class TestModelMappingKeyErrorFallback:
    """Test cases for _model_mapping KeyError fallback in _init_model."""

    def _make_cls(self, model_mapping_dict):
        """Create a mock cls with _model_mapping, parent class methods, etc."""
        cls = MagicMock()
        cls._model_mapping = model_mapping_dict
        return cls

    def test_force_hf_known_config_type(self):
        """force_hf path: _model_mapping lookup succeeds, class gets wrapped with mixin."""

        class FakeConfig:
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = FakeConfig()
        fake_model = FakeModel()

        cls = self._make_cls({FakeConfig: FakeModel})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,  # Pass config object directly (not str) to skip pretrained path
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=True,
            )

        assert is_custom is False
        mock_wrap.assert_called_once_with(FakeModel)

    def test_force_hf_unknown_config_falls_back_to_type_model(self):
        """force_hf path: KeyError in _model_mapping falls back to type(model)."""

        class UnknownConfig:
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = UnknownConfig()
        fake_model = FakeModel()

        # _model_mapping does NOT have UnknownConfig
        cls = self._make_cls({})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=True,
            )

        assert is_custom is False
        # Fallback: type(model) = FakeModel
        mock_wrap.assert_called_once_with(FakeModel)

    def test_fallback_path_known_config_type(self):
        """Fallback (non-force_hf, no custom model) path: _model_mapping succeeds."""

        class FakeConfig:
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = FakeConfig()
        fake_model = FakeModel()

        cls = self._make_cls({FakeConfig: FakeModel})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init.get_architectures", return_value=[]),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=False,
            )

        assert is_custom is False
        mock_wrap.assert_called_once_with(FakeModel)

    def test_fallback_path_unknown_config_falls_back_to_type_model(self):
        """Fallback path: KeyError in _model_mapping falls back to type(model)."""

        class UnknownConfig:
            name_or_path = "test-model"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

        fake_config = UnknownConfig()
        fake_model = FakeModel()

        cls = self._make_cls({})
        cls._from_config_parent_class = MagicMock(return_value=fake_model)

        with (
            patch("nemo_automodel._transformers.model_init.get_hf_config", return_value=fake_config),
            patch("nemo_automodel._transformers.model_init.get_architectures", return_value=[]),
            patch("nemo_automodel._transformers.model_init._get_mixin_wrapped_class") as mock_wrap,
        ):
            mock_wrap.return_value = type("WrappedModel", (HFCheckpointingMixin, FakeModel), {})
            is_custom, model = _init_model(
                cls,
                fake_config,
                attn_implementation="eager",
                torch_dtype="auto",
                quantization_config=None,
                force_hf=False,
            )

        assert is_custom is False
        mock_wrap.assert_called_once_with(FakeModel)


class TestNeMoAutoModelForMultimodalLM:
    """Tests for the NeMoAutoModelForMultimodalLM class and its exports."""

    def test_class_exists_and_inherits_correctly(self):
        from transformers import AutoModelForMultimodalLM

        from nemo_automodel._transformers.auto_model import NeMoAutoModelForMultimodalLM, _BaseNeMoAutoModelClass

        assert issubclass(NeMoAutoModelForMultimodalLM, _BaseNeMoAutoModelClass)
        assert issubclass(NeMoAutoModelForMultimodalLM, AutoModelForMultimodalLM)

    def test_has_from_pretrained_and_from_config(self):
        from nemo_automodel._transformers.auto_model import NeMoAutoModelForMultimodalLM

        assert callable(NeMoAutoModelForMultimodalLM.from_pretrained)
        assert callable(NeMoAutoModelForMultimodalLM.from_config)

    def test_lazy_export_from_transformers_subpackage(self):
        from nemo_automodel._transformers import NeMoAutoModelForMultimodalLM

        assert NeMoAutoModelForMultimodalLM is not None

    def test_lazy_export_from_top_level_package(self):
        from nemo_automodel import NeMoAutoModelForMultimodalLM

        assert NeMoAutoModelForMultimodalLM is not None

    def test_top_level_dir_includes_multimodal(self):
        import nemo_automodel

        assert "NeMoAutoModelForMultimodalLM" in dir(nemo_automodel)

    def test_transformers_subpackage_all_includes_multimodal(self):
        import nemo_automodel._transformers as pkg

        assert "NeMoAutoModelForMultimodalLM" in pkg.__all__
