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
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import transformers
from transformers import AutoConfig

from nemo_automodel.components._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    _get_next_fallback_attn,
    _patch_attention,
)
from nemo_automodel import __version__
from nemo_automodel.components.quantization.fp8 import FP8Config

HAS_LIGER_KERNEL = False
try:
    import liger_kernel
    HAS_LIGER_KERNEL = True
except Exception:
    pass

class TestNeMoAutoModelForCausalLM:
    """Test cases for NeMoAutoModelForCausalLM class."""
    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model

            # Test line 208 - warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
                assert model.config["nemo_version"] == __version__

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model
            assert mock_from_pretrained.call_count == 1

    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_config") as mock_from_config,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_config.return_value = mock_model

            config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

            # Test line 297 - warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForCausalLM.from_config(config)
                assert model.config["nemo_version"] == __version__

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model
            assert mock_from_config.call_count == 1

    def test_from_config_happy_path(self):
        """Test the basic from_config functionality works."""
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        model = NeMoAutoModelForCausalLM.from_config(config, attn_implementation="eager")
        assert model.config.nemo_version == __version__

    def test_from_pretrained_runtimeerror_triggers_reload(self):
        """When _patch_liger_kernel raises, the loader should retry with
        use_liger_kernel=False and return the second model instance."""
        # first and second dummy model objects
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}

        # record every call to _patch_liger_kernel
        patch_calls = []

        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel.components._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=[model1, model2],  # first, then retry
            ) as mock_from_pretrained,
        ):
            returned = NeMoAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
            assert returned.config["nemo_version"] == __version__

        # _patch_liger_kernel called twice, first with ligand=True, then False
        assert patch_calls == [model1]
        # The underlying HF loader is also called twice
        assert mock_from_pretrained.call_count == 2
        # The final object returned by our helper is the *second* model
        assert returned is model2

    def test_from_config_runtimeerror_triggers_reload(self):
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}

        patch_calls = []
        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel.components._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForCausalLM, "from_config", side_effect=[model1, model2]
            ) as mock_from_config,
        ):
            returned = NeMoAutoModelForCausalLM.from_config(cfg)
            assert returned.config["nemo_version"] == __version__

        assert patch_calls == [model1]
        assert mock_from_config.call_count == 2
        assert returned is model2

    def test_from_pretrained_valueerror_attention_fallback(self, caplog):
        """Test ValueError exception handling when attention implementation is not supported.

        When super().from_pretrained() raises ValueError with "does not support" message,
        the method should:
        1. Delete the model if it exists
        2. Fall back to the next attention implementation
        3. Log a warning
        4. Retry with the fallback attention implementation
        """
        # Create two model instances - first for failed attempt, second for successful retry
        model1, model2 = Mock(name="failed_model"), Mock(name="success_model")
        model1.config = {}
        model2.config = {}

        # Mock the call sequence: first call fails with ValueError, second succeeds
        def mock_from_pretrained_side_effect(*args, **kwargs):
            # Check the attn_implementation parameter to determine which call this is
            attn_impl = kwargs.get("attn_implementation", "sdpa")
            if attn_impl == "flash_attention_2":
                # First call with flash_attention_2 - should fail
                raise ValueError("Model does not support flash_attention_2 attention implementation")
            else:
                # Second call with fallback (sdpa) - should succeed
                return model2

        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=mock_from_pretrained_side_effect
            ) as mock_from_pretrained,
            caplog.at_level(logging.WARNING)
        ):
            # Test the exception path by starting with flash_attention_2
            returned = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                attn_implementation="flash_attention_2"
            )
            assert returned.config["nemo_version"] == __version__

        # Verify the warning was logged
        assert "Falling back to sdpa attention." in caplog.text

        # Verify from_pretrained was called twice (first failed, second succeeded)
        assert mock_from_pretrained.call_count == 2 + int(HAS_LIGER_KERNEL)

        # Verify the final returned model is the successful one
        assert returned is model2

        # Verify the calls were made with correct attention implementations
        call_args_list = mock_from_pretrained.call_args_list
        assert call_args_list[0][1]["attn_implementation"] == "flash_attention_2"
        assert call_args_list[1][1]["attn_implementation"] == "sdpa"

    def test_from_pretrained_valueerror_non_attention_reraises(self):
        """Test that ValueError not related to attention implementation is re-raised.

        When super().from_pretrained() raises ValueError that doesn't contain
        "does not support", the exception should be re-raised without fallback.
        """
        def mock_from_pretrained_side_effect(*args, **kwargs):
            raise ValueError("Some other error not related to attention")

        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=mock_from_pretrained_side_effect
            ) as mock_from_pretrained,
        ):
            # Test that the ValueError is re-raised
            with pytest.raises(ValueError, match="Some other error not related to attention"):
                NeMoAutoModelForCausalLM.from_pretrained(
                    "hf-internal-testing/tiny-random-gpt2",
                    attn_implementation="flash_attention_2"
                )

        # Verify from_pretrained was called only once (no retry)
        assert mock_from_pretrained.call_count == 1

    def test_from_pretrained_model_deletion_on_exception(self):
        """Test that partially created model is properly deleted when exception occurs.

        When super().from_pretrained() raises ValueError with "does not support" and
        a model object was created, it should be deleted before retrying.
        """
        model1, model2 = Mock(name="failed_model"), Mock(name="success_model")
        model1.config = {}
        model2.config = {}

        # Track which models are created and when deletion logic is triggered
        models_created = []
        call_count = 0

        def mock_from_pretrained_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            attn_impl = kwargs.get("attn_implementation", "sdpa")

            if call_count == 1 and attn_impl == "flash_attention_2":
                # First call - create model1 and add to tracking, then raise exception
                models_created.append(model1)
                raise ValueError("Model does not support flash_attention_2 attention implementation")
            else:
                # Second call - succeed with model2
                models_created.append(model2)
                return model2

        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=mock_from_pretrained_side_effect
            ) as mock_from_pretrained,
        ):
            returned = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                attn_implementation="flash_attention_2"
            )
            assert returned.config["nemo_version"] == __version__

        # Verify the method was called twice for retry
        assert mock_from_pretrained.call_count == 2 + int(HAS_LIGER_KERNEL)

        # Verify both models were created during the process
        assert len(models_created) == 2 + int(HAS_LIGER_KERNEL)
        assert models_created[0] is model1  # First attempt
        assert models_created[1] is model2  # Successful retry

        # Verify the final returned model is the successful one
        assert returned is model2

        # Verify the calls were made with correct attention implementations
        call_args_list = mock_from_pretrained.call_args_list
        assert call_args_list[0][1]["attn_implementation"] == "flash_attention_2"
        assert call_args_list[1][1]["attn_implementation"] == "sdpa"

    def test_from_config_valueerror_attention_fallback(self, caplog):
        """Test ValueError exception handling in from_config when attention implementation is not supported.

        When super().from_config() raises ValueError with "does not support" message,
        the method should:
        1. Fall back to eager attention implementation
        2. Log a warning
        3. Retry with the fallback attention implementation
        """
        # Create two model instances - first for failed attempt, second for successful retry
        model1, model2 = Mock(name="failed_model"), Mock(name="success_model")
        model1.config = {}
        model2.config = {}

        # Mock the call sequence: first call fails with ValueError, second succeeds
        def mock_from_config_side_effect(*args, **kwargs):
            # Check the attn_implementation parameter to determine which call this is
            attn_impl = kwargs.get("attn_implementation", "flash_attention_2")
            if attn_impl == "flash_attention_2":
                # First call with flash_attention_2 - should fail
                raise ValueError("Model does not support flash_attention_2 attention implementation")
            else:
                # Second call with fallback (eager) - should succeed
                return model2

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForCausalLM,
                "from_config",
                side_effect=mock_from_config_side_effect
            ) as mock_from_config,
            caplog.at_level(logging.WARNING)
        ):
            # Test the exception path by starting with flash_attention_2
            returned = NeMoAutoModelForCausalLM.from_config(
                cfg,
                attn_implementation="flash_attention_2"
            )
            assert returned.config["nemo_version"] == __version__

        # Verify the warning was logged
        assert "Falling back to eager attention." in caplog.text

        # Verify from_config was called twice (first failed, second succeeded)
        assert mock_from_config.call_count == 2 + int(HAS_LIGER_KERNEL)

        # Verify the final returned model is the successful one
        assert returned is model2

        # Verify the calls were made with correct attention implementations
        call_args_list = mock_from_config.call_args_list
        assert call_args_list[0][1]["attn_implementation"] == "flash_attention_2"
        assert call_args_list[1][1]["attn_implementation"] == "eager"


class TestNeMoAutoModelForImageTextToText:
    """Test cases for NeMoAutoModelForImageTextToText class."""

    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForImageTextToText, "from_pretrained") as mock_from_pretrained,
        ):
            mock_model = Mock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model

            # Test line 356 - warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")
                assert model.config["nemo_version"] == __version__

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model
            assert mock_from_pretrained.call_count == 1

    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForImageTextToText, "from_config") as mock_from_config,
        ):
            mock_model = Mock()
            mock_model.config = Mock()
            mock_from_config.return_value = mock_model

            config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

            # Test warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForImageTextToText.from_config(config)

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model
            assert mock_from_config.call_count == 1

    def test_from_pretrained_runtimeerror_triggers_reload(self):
        """When _patch_liger_kernel raises, the loader should retry with
        use_liger_kernel=False and return the second model instance."""
        # first and second dummy model objects
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}

        patch_calls = []
        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel.components._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForImageTextToText,
                "from_pretrained",
                side_effect=[model1, model2],  # first, then retry
            ) as mock_from_pretrained,
        ):
            returned = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")
            assert returned.config["nemo_version"] == __version__


        # _patch_liger_kernel called twice, first with ligand=True, then False
        assert patch_calls == [model1]
        # The underlying HF loader is also called twice
        assert mock_from_pretrained.call_count == 2
        # The final object returned by our helper is the *second* model
        assert returned is model2


    def test_from_pretrained_sdpa_runtimeerror_triggers_reload(self):
        """When _patch_liger_kernel raises, the loader should retry with
        use_liger_kernel=False and return the second model instance."""
        # first and second dummy model objects
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}

        patch_calls = []
        def fake__patch_attention(model, sdpa_method):
            patch_calls.append(model)
            raise RuntimeError("boom")

        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel.components._transformers.auto_model._patch_liger_kernel", lambda x: x),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", fake__patch_attention),
            patch.object(
                transformers.AutoModelForImageTextToText,
                "from_pretrained",
                side_effect=[model1, model2],  # first, then retry
            ) as mock_from_pretrained,
        ):
            returned = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")
            assert returned.config["nemo_version"] == __version__


        # _patch_liger_kernel called twice, first with ligand=True, then False
        assert patch_calls == [model1]
        # The underlying HF loader is also called twice
        assert mock_from_pretrained.call_count == 2
        # The final object returned by our helper is the *second* model
        assert returned is model2

    def test_from_config_runtimeerror_triggers_reload(self):
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}

        patch_calls = []

        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel.components._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(
                transformers.AutoModelForImageTextToText, "from_config", side_effect=[model1, model2]
            ) as mock_from_config,
        ):
            returned = NeMoAutoModelForImageTextToText.from_config(cfg)
            assert returned.config["nemo_version"] == __version__

        assert patch_calls == [model1]
        assert mock_from_config.call_count == 2
        assert returned is model2

    def test_from_config_sdap_runtimeerror_triggers_reload(self):
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}

        patch_calls = []

        def fake__patch_attention(model, sdpa_method):
            patch_calls.append(model)
            raise RuntimeError("boom")

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel.components._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel.components._transformers.auto_model._patch_liger_kernel", lambda x: x),
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", fake__patch_attention),
            patch.object(
                transformers.AutoModelForImageTextToText, "from_config", side_effect=[model1, model2]
            ) as mock_from_config,
        ):
            returned = NeMoAutoModelForImageTextToText.from_config(cfg)
            assert returned.config["nemo_version"] == __version__

        assert patch_calls == [model1]
        assert mock_from_config.call_count == 2
        assert returned is model2

class TestPatchAttention:
    """Test cases for _patch_attention function."""

    def test__patch_attention_basic(self):
        """Test basic _patch_attention functionality."""
        # Create a mock object with a forward method
        mock_obj = Mock()
        mock_forward = Mock()
        mock_obj.forward = mock_forward

        # Mock the forward method to be a bound method
        mock_forward.__func__ = Mock()
        mock_forward.__self__ = mock_obj

        with (
            patch("nemo_automodel.components._transformers.auto_model.sdpa_kernel") as mock_sdpa_kernel,  # noqa: F841
            patch("nemo_automodel.components._transformers.auto_model._assert_same_signature"),
        ):
            result = _patch_attention(mock_obj)

            assert result is mock_obj
            # Verify that the forward method was replaced
            assert mock_obj.forward != mock_forward

    def test__patch_attention_with_custom_sdpa_method(self):
        """Test _patch_attention with custom SDPA method."""
        from torch.nn.attention import SDPBackend

        mock_obj = Mock()
        mock_forward = Mock()
        mock_obj.forward = mock_forward

        # Mock the forward method to be a bound method
        mock_forward.__func__ = Mock()
        mock_forward.__self__ = mock_obj

        custom_sdpa_method = [SDPBackend.FLASH_ATTENTION]

        with (
            patch("nemo_automodel.components._transformers.auto_model.sdpa_kernel") as mock_sdpa_kernel,  # noqa: F841
            patch("nemo_automodel.components._transformers.auto_model._assert_same_signature"),
        ):
            result = _patch_attention(mock_obj, custom_sdpa_method)

            assert result is mock_obj
            # Verify that the forward method was replaced
            assert mock_obj.forward != mock_forward


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_assert_same_signature_matching(self):
        """Test _assert_same_signature with matching signatures."""
        from nemo_automodel.components._transformers.auto_model import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, c=None):
            pass

        # Should not raise an exception
        _assert_same_signature(func1, func2)

    def test_assert_same_signature_different(self):
        """Test _assert_same_signature with different signatures."""
        from nemo_automodel.components._transformers.auto_model import _assert_same_signature

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


@pytest.mark.parametrize("use_liger,has_liger", [(True, True), (False, True)])
def test_success_paths(monkeypatch, use_liger, has_liger):
    """
    1. Liger available & requested  -> kernel applied, _patch_attention called.
    2. Liger *not* requested        -> kernel *not* applied, _patch_attention called.
    """
    import nemo_automodel.components._transformers.auto_model as tgt

    apply_mock, attn_mock = prepare_env(monkeypatch, tgt, has_liger=has_liger, apply_ok=True)

    model = DummyModel()
    if use_liger:
        patched = tgt._patch_liger_kernel(model)
    else:
        patched = model

    # Always returns same instance (unless exception path)
    assert patched is model

    if use_liger:
        apply_mock.assert_called_once()
        assert model.called is True
    else:
        apply_mock.assert_not_called()
        assert model.called is False

    # SDPA not called inside _patch_liger_kernel
    attn_mock.assert_not_called()



def test_liger_not_available(monkeypatch):
    """
    Asked for Liger but HAS_LIGER_KERNEL is False.
    Expect: return untouched model, _patch_attention still invoked,
            no exceptions thrown.
    """
    import nemo_automodel.components._transformers.auto_model as tgt

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
    import nemo_automodel.components._transformers.auto_model as tgt

    prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=False,  # force failure
    )

    with pytest.raises(RuntimeError, match="Failed to patch model"):
        tgt._patch_liger_kernel(DummyModel())


class TestNeMoAutoModelFP8Integration:
    """Test cases for FP8 functionality in NeMo auto models."""
    
    def test_from_pretrained_without_fp8_config_works(self):
        """Test that from_pretrained works normally without fp8_config."""
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model

            # Should work fine without fp8_config - no FP8 will be applied
            result = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                fp8_config=None
            )
            
            assert result == mock_model
            assert result.config["nemo_version"] == __version__
    
    def test_from_config_without_fp8_config_works(self):
        """Test that from_config works normally without fp8_config."""
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_config") as mock_from_config,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_config.return_value = mock_model

            # Should work fine without fp8_config - no FP8 will be applied
            result = NeMoAutoModelForCausalLM.from_config(
                config,
                fp8_config=None
            )
            
            assert result == mock_model
            assert result.config["nemo_version"] == __version__
    
    def test_from_pretrained_with_fp8_config_object(self):
        """Test from_pretrained with FP8Config object."""
        fp8_config = FP8Config(
            recipe_name="tensorwise",
            enable_fsdp_float8_all_gather=True,
            filter_fqns=["lm_head"]
        )
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model
            mock_apply_fp8.return_value = mock_model

            result = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                fp8_config=fp8_config
            )

            # Should call apply_fp8_to_model with correct parameters
            mock_apply_fp8.assert_called_once()
            call_kwargs = mock_apply_fp8.call_args[1]
            assert call_kwargs['recipe_name'] == "tensorwise"
            assert call_kwargs['enable_fsdp_float8_all_gather'] is True
            assert call_kwargs['filter_fqns'] == ["lm_head"]
            assert result.config["nemo_version"] == __version__
    
    def test_from_config_with_fp8_config_object(self):
        """Test from_config with FP8Config object."""
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        fp8_config = FP8Config(
            recipe_name="rowwise",
            emulate=True,
            filter_fqns=["embed_tokens"]
        )
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_config") as mock_from_config,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_config.return_value = mock_model
            mock_apply_fp8.return_value = mock_model

            result = NeMoAutoModelForCausalLM.from_config(
                config,
                fp8_config=fp8_config
            )

            mock_apply_fp8.assert_called_once()
            call_kwargs = mock_apply_fp8.call_args[1]
            assert call_kwargs['recipe_name'] == "rowwise"
            assert call_kwargs['emulate'] is True
            assert call_kwargs['filter_fqns'] == ["embed_tokens"]
            assert result.config["nemo_version"] == __version__
    
    def test_from_pretrained_with_fp8_config_direct_access(self):
        """Test from_pretrained with fp8_config accessing attributes directly."""
        fp8_config = FP8Config(
            recipe_name="tensorwise",
            emulate=False,
            enable_fsdp_float8_all_gather=True,
            precompute_float8_dynamic_scale_for_fsdp=True,
            filter_fqns=["lm_head"]
        )
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model
            mock_apply_fp8.return_value = mock_model

            result = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                fp8_config=fp8_config
            )

            mock_apply_fp8.assert_called_once()
            call_kwargs = mock_apply_fp8.call_args[1]
            assert call_kwargs['recipe_name'] == "tensorwise"
            assert call_kwargs['emulate'] is False
            assert call_kwargs['enable_fsdp_float8_all_gather'] is True
            assert call_kwargs['filter_fqns'] == ["lm_head"]
            
            assert mock_model.precompute_float8_dynamic_scale_for_fsdp is True
    
    def test_from_pretrained_with_dict_like_object(self):
        """Test from_pretrained with dict-like object."""
        class MockDictLikeObject:
            def __init__(self):
                self.recipe_name = 'rowwise'
                self.emulate = True
                self.filter_fqns = ['lm_head']
                self.enable_fsdp_float8_all_gather = False
                self.force_recompute_fp8_weight_in_bwd = False
                self.precompute_float8_dynamic_scale_for_fsdp = True
                
            def to_dict(self):
                return {
                    'recipe_name': self.recipe_name,
                    'emulate': self.emulate,
                    'filter_fqns': self.filter_fqns,
                    'enable_fsdp_float8_all_gather': self.enable_fsdp_float8_all_gather,
                    'force_recompute_fp8_weight_in_bwd': self.force_recompute_fp8_weight_in_bwd,
                    'precompute_float8_dynamic_scale_for_fsdp': self.precompute_float8_dynamic_scale_for_fsdp
                }
        
        mock_dict_obj = MockDictLikeObject()
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model
            mock_apply_fp8.return_value = mock_model

            result = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                fp8_config=mock_dict_obj
            )

            # Should call apply_fp8_to_model
            mock_apply_fp8.assert_called_once()
            call_kwargs = mock_apply_fp8.call_args[1]
            assert call_kwargs['recipe_name'] == "rowwise"
            assert call_kwargs['emulate'] is True
            assert call_kwargs['filter_fqns'] == ['lm_head']
            assert call_kwargs['enable_fsdp_float8_all_gather'] is False
            assert call_kwargs['force_recompute_fp8_weight_in_bwd'] is False
            
            # Should set precompute flag on model (precompute=True, but recipe!=tensorwise and all_gather=False -> False)
            assert mock_model.precompute_float8_dynamic_scale_for_fsdp is False
    
    def test_from_pretrained_precompute_logic_validation(self):
        """Test that precompute is only True when recipe=tensorwise AND enable_fsdp_float8_all_gather=True."""
        test_cases = [
            # (precompute, recipe, all_gather, expected_result)
            (True, "tensorwise", True, True),     # All conditions met -> True
            (True, "tensorwise", False, False),   # Missing all_gather -> False
            (True, "rowwise", True, False),       # Wrong recipe -> False
            (True, "rowwise", False, False),      # Wrong recipe and missing all_gather -> False
            (False, "tensorwise", True, False),   # precompute disabled -> False
        ]
        
        for precompute, recipe, all_gather, expected in test_cases:
            fp8_config = FP8Config(
                recipe_name=recipe,
                enable_fsdp_float8_all_gather=all_gather,
                precompute_float8_dynamic_scale_for_fsdp=precompute,
            )
            
            with (
                patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
                patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
                patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
            ):
                mock_model = MagicMock()
                mock_model.config = {}
                mock_from_pretrained.return_value = mock_model
                mock_apply_fp8.return_value = mock_model

                result = NeMoAutoModelForCausalLM.from_pretrained(
                    "hf-internal-testing/tiny-random-gpt2",
                    fp8_config=fp8_config
                )

                assert mock_model.precompute_float8_dynamic_scale_for_fsdp is expected, \
                    f"Failed for precompute={precompute}, recipe={recipe}, all_gather={all_gather}. Expected {expected}, got {mock_model.precompute_float8_dynamic_scale_for_fsdp}"
    
    def test_from_pretrained_no_fp8_when_config_none(self):
        """Test that no FP8 is applied when fp8_config is None."""
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model

            # Should work fine without fp8_config - no FP8 should be applied
            result = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                fp8_config=None
            )

            assert result == mock_model
            assert result.config["nemo_version"] == __version__
            # The auto_model may have retry logic, so just check it was called
            assert mock_from_pretrained.call_count >= 1
            # FP8 should not be applied when fp8_config is None
            mock_apply_fp8.assert_not_called()
    
    def test_from_pretrained_fp8_runtime_error_retry(self):
        """Test that FP8 RuntimeError triggers retry without FP8."""
        fp8_config = FP8Config(recipe_name="tensorwise")
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_from_pretrained,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model
            
            # First call raises RuntimeError, second call (retry) should succeed
            mock_apply_fp8.side_effect = [RuntimeError("FP8 failed"), mock_model]

            with patch('logging.warning') as mock_warning:
                result = NeMoAutoModelForCausalLM.from_pretrained(
                    "hf-internal-testing/tiny-random-gpt2",
                    fp8_config=fp8_config
                )

            # Should log warning about retrying without FP8
            mock_warning.assert_called_with("Retrying without FP8 quantization.")
            
            # Should call from_pretrained multiple times due to retry logic
            assert mock_from_pretrained.call_count >= 2
            
            # The FP8 application should only happen once (before the retry)
            assert mock_apply_fp8.call_count == 1
            
            assert result.config["nemo_version"] == __version__
    
    def test_vlm_model_fp8_integration(self):
        """Test that VLM models also support FP8 integration."""
        fp8_config = FP8Config(
            recipe_name="rowwise", 
            emulate=True,
            precompute_float8_dynamic_scale_for_fsdp=False,
            enable_fsdp_float8_all_gather=False
        )
        
        with (
            patch("nemo_automodel.components._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch.object(transformers.AutoModelForImageTextToText, "from_pretrained") as mock_from_pretrained,
            patch("nemo_automodel.components._transformers.auto_model.apply_fp8_to_model") as mock_apply_fp8,
        ):
            mock_model = MagicMock()
            mock_model.config = {}
            mock_from_pretrained.return_value = mock_model
            mock_apply_fp8.return_value = mock_model

            result = NeMoAutoModelForImageTextToText.from_pretrained(
                "microsoft/Phi-3.5-vision-instruct", 
                fp8_config=fp8_config
            )

            mock_apply_fp8.assert_called_once()
            call_kwargs = mock_apply_fp8.call_args[1]
            assert call_kwargs['recipe_name'] == "rowwise"
            assert call_kwargs['emulate'] is True
            assert call_kwargs['enable_fsdp_float8_all_gather'] is False
            assert result.config["nemo_version"] == __version__
            
            assert mock_model.precompute_float8_dynamic_scale_for_fsdp is False
