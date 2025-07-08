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
import pytest
from unittest.mock import MagicMock, Mock, patch
import transformers
from transformers import AutoConfig
import torch
from nemo_automodel._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    patch_attention,
    patch_model,
)


class TestNeMoAutoModelForCausalLM:
    """Test cases for NeMoAutoModelForCausalLM class."""

    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False), \
            patch('nemo_automodel._transformers.auto_model.patch_attention', lambda obj, sdpa_method=None: obj):
            with patch.object(transformers.AutoModelForCausalLM, 'from_pretrained') as mock_from_pretrained:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_pretrained.return_value = mock_model

                # Test line 208 - warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForCausalLM.from_pretrained("gpt2")

                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_pretrained.assert_called_once()


    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False), \
            patch('nemo_automodel._transformers.auto_model.patch_attention', lambda obj, sdpa_method=None: obj):
            with patch.object(transformers.AutoModelForCausalLM, 'from_config') as mock_from_config:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_config.return_value = mock_model

                config = AutoConfig.from_pretrained("gpt2")

                # Test line 297 - warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForCausalLM.from_config(config)

                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_config.assert_called_once()

class TestNeMoAutoModelForImageTextToText:
    """Test cases for NeMoAutoModelForImageTextToText class."""

    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False), \
            patch('nemo_automodel._transformers.auto_model.patch_attention', lambda obj, sdpa_method=None: obj):
            with patch.object(transformers.AutoModelForImageTextToText, 'from_pretrained') as mock_from_pretrained:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_pretrained.return_value = mock_model

                # Test line 356 - warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")

                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_pretrained.assert_called_once()

    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False), \
            patch('nemo_automodel._transformers.auto_model.patch_attention', lambda obj, sdpa_method=None: obj):
            with patch.object(transformers.AutoModelForImageTextToText, 'from_config') as mock_from_config:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_config.return_value = mock_model

                config = AutoConfig.from_pretrained("gpt2")

                # Test warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForImageTextToText.from_config(config)

                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_config.assert_called_once()


class TestPatchAttention:
    """Test cases for patch_attention function."""

    def test_patch_attention_basic(self):
        """Test basic patch_attention functionality."""
        # Create a mock object with a forward method
        mock_obj = Mock()
        mock_forward = Mock()
        mock_obj.forward = mock_forward

        # Mock the forward method to be a bound method
        mock_forward.__func__ = Mock()
        mock_forward.__self__ = mock_obj

        with patch('nemo_automodel._transformers.auto_model.sdpa_kernel') as mock_sdpa_kernel:
            with patch('nemo_automodel._transformers.auto_model._assert_same_signature'):
                result = patch_attention(mock_obj)

                assert result is mock_obj
                # Verify that the forward method was replaced
                assert mock_obj.forward != mock_forward

    def test_patch_attention_with_custom_sdpa_method(self):
        """Test patch_attention with custom SDPA method."""
        from torch.nn.attention import SDPBackend

        mock_obj = Mock()
        mock_forward = Mock()
        mock_obj.forward = mock_forward

        # Mock the forward method to be a bound method
        mock_forward.__func__ = Mock()
        mock_forward.__self__ = mock_obj

        custom_sdpa_method = [SDPBackend.FLASH_ATTENTION]

        with patch('nemo_automodel._transformers.auto_model.sdpa_kernel') as mock_sdpa_kernel:
            with patch('nemo_automodel._transformers.auto_model._assert_same_signature'):
                result = patch_attention(mock_obj, custom_sdpa_method)

                assert result is mock_obj
                # Verify that the forward method was replaced
                assert mock_obj.forward != mock_forward


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_assert_same_signature_matching(self):
        """Test _assert_same_signature with matching signatures."""
        from nemo_automodel._transformers.auto_model import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, c=None):
            pass

        # Should not raise an exception
        _assert_same_signature(func1, func2)

    def test_assert_same_signature_different(self):
        """Test _assert_same_signature with different signatures."""
        from nemo_automodel._transformers.auto_model import _assert_same_signature

        def func1(a, b, c=None):
            pass

        def func2(a, b, d=None):
            pass

        # Should raise an AssertionError
        with pytest.raises(AssertionError):
            _assert_same_signature(func1, func2)



class DummyModel(torch.nn.Module):
    """A tiny nn.Module that behaves enough like a HF/BERT style model."""
    def __init__(self):
        super().__init__()
        self.config = {}          # patch_model calls  model.config.update(...)
        self.called = False       # turned on by fake liger kernel

    def mark(self):
        self.called = True

def prepare_env(monkeypatch, target_mod, *,
                has_liger=True,
                apply_ok=True):
    """
    Patch every external symbol that patch_model touches.

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
    monkeypatch.setattr(target_mod, "patch_attention", patch_attn_mock, raising=True)

    monkeypatch.setattr(target_mod, "__version__", "unit-test-ver", raising=False)

    return apply_mock, patch_attn_mock

@pytest.mark.parametrize("use_liger,has_liger", [(True, True), (False, True)])
def test_success_paths(monkeypatch, use_liger, has_liger):
    """
    1. Liger available & requested  -> kernel applied, patch_attention called.
    2. Liger *not* requested        -> kernel *not* applied, patch_attention called.
    """
    import nemo_automodel._transformers.auto_model as tgt
    apply_mock, attn_mock = prepare_env(
        monkeypatch,
        tgt,
        has_liger=has_liger,
        apply_ok=True
    )

    model = DummyModel()
    patched = tgt.patch_model(
        model,
        use_liger_kernel=use_liger,
        use_sdpa_patching=True,
    )

    # Always returns same instance (unless exception path)
    assert patched is model
    # nemo_version must be set
    assert patched.config["nemo_version"] == "unit-test-ver"

    if use_liger:
        apply_mock.assert_called_once()
        assert model.called is True
    else:
        apply_mock.assert_not_called()
        assert model.called is False

    # SDPA path always taken in these tests
    attn_mock.assert_called_once_with(model, None)


def test_liger_not_available(monkeypatch):
    """
    Asked for Liger but HAS_LIGER_KERNEL is False.
    Expect: return untouched model, patch_attention still invoked,
            no exceptions thrown.
    """
    import nemo_automodel._transformers.auto_model as tgt
    apply_mock, attn_mock = prepare_env(
        monkeypatch,
        tgt,
        has_liger=False, # unavailable
        apply_ok=True
    )

    model = DummyModel()
    out = tgt.patch_model(model, use_liger_kernel=True, use_sdpa_patching=True)

    # untouched instance returned
    assert out is model
    assert model.called is False
    # _apply never called, because we short-circuit when HAS_LIGER_KERNEL==False
    apply_mock.assert_not_called()
    attn_mock.assert_called_once()


def test_liger_apply_failure_raises(monkeypatch):
    """
    If _apply_liger_kernel_to_instance throws, patch_model must
    clean up and raise RuntimeError.
    """
    import nemo_automodel._transformers.auto_model as tgt
    prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=False   # force failure
    )

    with pytest.raises(RuntimeError, match="Failed to patch model"):
        tgt.patch_model(DummyModel(), use_liger_kernel=True, use_sdpa_patching=False)