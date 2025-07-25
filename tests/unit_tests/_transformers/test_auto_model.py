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
    _patch_attention,
)
from nemo_automodel import __version__


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
