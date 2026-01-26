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

from nemo_automodel._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    _get_next_fallback_attn,
    _patch_attention,
)


HAS_LIGER_KERNEL = False
try:
    import liger_kernel
    HAS_LIGER_KERNEL = True
except Exception:
    pass

def _create_mock_wrapped_class(from_pretrained_side_effect=None, from_config_side_effect=None):
    """Helper to create a mock wrapped class for testing the HF fallback path.

    Args:
        from_pretrained_side_effect: Either a model to return, a list of models to return
            sequentially, or a function(callable but not Mock) to call.
        from_config_side_effect: Same as above for from_config.
    """
    class MockWrappedClass:
        """A mock class that simulates the mixin-wrapped HF model class."""
        _checkpointer = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            if from_pretrained_side_effect is not None:
                # Use types.FunctionType to check for real functions, not MagicMock
                if isinstance(from_pretrained_side_effect, types.FunctionType):
                    return from_pretrained_side_effect(*args, **kwargs)
                elif isinstance(from_pretrained_side_effect, list):
                    return from_pretrained_side_effect.pop(0)
                else:
                    return from_pretrained_side_effect
            mock_model = MagicMock()
            mock_model.config = {}
            return mock_model

        @classmethod
        def from_config(cls, *args, **kwargs):
            if from_config_side_effect is not None:
                # Use types.FunctionType to check for real functions, not MagicMock
                if isinstance(from_config_side_effect, types.FunctionType):
                    return from_config_side_effect(*args, **kwargs)
                elif isinstance(from_config_side_effect, list):
                    return from_config_side_effect.pop(0)
                else:
                    return from_config_side_effect
            mock_model = MagicMock()
            mock_model.config = {}
            return mock_model

    return MockWrappedClass


class TestNeMoAutoModelForCausalLM:
    """Test cases for NeMoAutoModelForCausalLM class."""
    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        mock_model = MagicMock()
        mock_model.config = {}

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=mock_model)),
        ):
            # Test - warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model

    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        mock_model = MagicMock()
        mock_model.config = {}

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_config_side_effect=mock_model)),
        ):
            config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

            # Test - warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForCausalLM.from_config(config)

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model

    def test_from_pretrained_uses_registry_when_available(self):
        """If AutoConfig.architectures[0] maps to a custom class in ModelRegistry,
        ensure that the registry path is taken and HF loader is not called."""
        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch("nemo_automodel._transformers.auto_model.os.path.isdir", return_value=True),
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_hf_loader,
        ):
            # Prepare a fake config with architectures
            cfg = Mock()
            cfg.architectures = ["CustomArch"]
            mock_cfg_from_pretrained.return_value = cfg

            # Prepare a fake custom model class with from_pretrained method
            custom_model_instance = Mock()
            custom_model_instance.config = Mock()
            custom_cls = Mock()
            custom_cls.__name__ = "MockMockMock"
            custom_cls.from_pretrained = Mock(return_value=custom_model_instance)
            mock_registry.model_arch_name_to_cls = {"CustomArch": custom_cls}

            returned = NeMoAutoModelForCausalLM.from_pretrained("dummy/path")

            # Should have returned the custom model instance directly
            assert returned is custom_model_instance
            # HF path should not be invoked
            mock_hf_loader.assert_not_called()
            # Custom cls.from_pretrained should be invoked
            custom_cls.from_pretrained.assert_called()

    def test_from_config_uses_registry_when_available(self):
        """If config.architectures[0] maps to a custom class in ModelRegistry,
        ensure that the registry path is taken and HF loader is not called."""
        with (
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch.object(transformers.AutoModelForCausalLM, "from_config") as mock_hf_loader,
        ):
            # Fake config with architectures attribute
            cfg = Mock()
            cfg.architectures = ["CustomArch"]
            # Provide a concrete path string to avoid Mock flowing into os.path.isdir
            cfg.name_or_path = "custom/model"

            # Registry provides a custom class with from_config method
            custom_model_instance = Mock()
            custom_cls = Mock()
            custom_cls.__name__ = "MockMockMock"
            custom_cls.from_config = Mock(return_value=custom_model_instance)
            mock_registry.model_arch_name_to_cls = {"CustomArch": custom_cls}

            returned = NeMoAutoModelForCausalLM.from_config(cfg)

            # Should return custom model instance
            assert returned is custom_model_instance
            mock_hf_loader.assert_not_called()
            # Custom cls.from_config should be invoked (not __call__)
            custom_cls.from_config.assert_called()
            args, _ = custom_cls.from_config.call_args
            assert args[0] is cfg

    def test_from_pretrained_registry_downloads_checkpoint_files_rank0(self):
        """When using a custom model implementation, ensure rank0 downloads weights and we barrier."""
        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_hf_loader,
            patch("nemo_automodel._transformers.auto_model._get_resolved_checkpoint_files") as mock_get_files,
            patch("nemo_automodel._transformers.auto_model.os.path.isdir", return_value=False),
            patch("nemo_automodel.components.distributed.utils.FirstRankPerNode") as mock_barrier,
        ):
            # Prepare a fake config with architectures and commit hash
            cfg = Mock()
            cfg.architectures = ["CustomArch"]
            cfg._commit_hash = "abc123"
            mock_cfg_from_pretrained.return_value = cfg

            # Prepare a fake custom model class with from_pretrained method
            custom_model_instance = Mock()
            custom_model_instance.config = Mock()
            custom_cls = Mock()
            custom_cls.__name__ = "MockMockMock"
            custom_cls.from_pretrained = Mock(return_value=custom_model_instance)
            mock_registry.model_arch_name_to_cls = {"CustomArch": custom_cls}

            returned = NeMoAutoModelForCausalLM.from_pretrained("dummy/repo-id")

            # Should have returned the custom model instance directly
            assert returned is custom_model_instance
            # HF path should not be invoked
            mock_hf_loader.assert_not_called()
            # Rank 0 should trigger a download
            assert mock_get_files.call_count == 1
            _, kwargs = mock_get_files.call_args
            assert kwargs["pretrained_model_name_or_path"] == "dummy/repo-id"
            assert kwargs["commit_hash"] == "abc123"
            # Distributed barrier should be called when initialized
            mock_barrier.assert_called_once()

    def test_from_pretrained_registry_downloads_when_dist_uninitialized(self):
        """When dist is not initialized, we still download but do not barrier."""
        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch.object(transformers.AutoModelForCausalLM, "from_pretrained") as mock_hf_loader,
            patch("nemo_automodel._transformers.auto_model._get_resolved_checkpoint_files") as mock_get_files,
            patch("nemo_automodel._transformers.auto_model.os.path.isdir", return_value=False),
        ):
            # Prepare a fake config with architectures and commit hash
            cfg = Mock()
            cfg.architectures = ["CustomArch"]
            cfg._commit_hash = "commit456"
            mock_cfg_from_pretrained.return_value = cfg

            # Prepare a fake custom model class with from_pretrained method
            custom_model_instance = Mock()
            custom_model_instance.config = Mock()
            custom_cls = Mock()
            custom_cls.__name__ = "MockMockMock"
            custom_cls.from_pretrained = Mock(return_value=custom_model_instance)
            mock_registry.model_arch_name_to_cls = {"CustomArch": custom_cls}

            returned = NeMoAutoModelForCausalLM.from_pretrained("dummy/repo-id")

            # Should have returned the custom model instance directly
            assert returned is custom_model_instance
            # HF path should not be invoked
            mock_hf_loader.assert_not_called()
            # Not initialized -> still downloads
            assert mock_get_files.call_count == 1
            _, kwargs = mock_get_files.call_args
            assert kwargs["pretrained_model_name_or_path"] == "dummy/repo-id"
            assert kwargs["commit_hash"] == "commit456"

    def test_from_config_happy_path(self):
        """Test the basic from_config functionality works."""
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        model = NeMoAutoModelForCausalLM.from_config(config, attn_implementation="eager")

        assert model is not None
        assert hasattr(model, "config")

    def test_from_config_with_string_calls_autoconfig(self):
        """Test that from_config calls AutoConfig.from_pretrained when config is a string."""
        mock_model = MagicMock()
        mock_model.config = {}
        # Get a real config to use for the mock return value (this call is NOT patched)
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        from_config_call_count = [0]

        def mock_from_config_side_effect(*args, **kwargs):
            from_config_call_count[0] += 1
            return mock_model

        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_autoconfig,
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_config_side_effect=mock_from_config_side_effect)),
        ):
            # Set up the mock chain
            mock_autoconfig.return_value = real_config

            model = NeMoAutoModelForCausalLM.from_config(
                "hf-internal-testing/tiny-random-gpt2",
                trust_remote_code=False
            )

            # Verify AutoConfig.from_pretrained was called with the string
            mock_autoconfig.assert_called_once_with(
                "hf-internal-testing/tiny-random-gpt2",
                trust_remote_code=False,
                attn_implementation="flash_attention_2",
            )
            # Verify from_config was called on the wrapped class
            assert from_config_call_count[0] >= 1
            # Verify the model was returned
            assert model is mock_model

    def test_from_pretrained_runtimeerror_triggers_reload(self):
        """When _patch_liger_kernel raises, the loader should retry with
        use_liger_kernel=False and return the second model instance."""
        # first and second dummy model objects
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}
        models = [model1, model2]

        # record every call to _patch_liger_kernel
        patch_calls = []

        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=models)),
        ):
            returned = NeMoAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        # _patch_liger_kernel called once on first model, then retry without liger
        assert patch_calls == [model1]
        # The final object returned by our helper is the *second* model
        assert returned is model2

    def test_from_config_runtimeerror_triggers_reload(self):
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}
        models = [model1, model2]

        patch_calls = []
        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_config_side_effect=models)),
        ):
            returned = NeMoAutoModelForCausalLM.from_config(cfg)

        assert patch_calls == [model1]
        assert returned is model2

    def test_from_pretrained_valueerror_attention_fallback(self, caplog):
        """Test ValueError exception handling when attention implementation is not supported.

        When wrapped class from_pretrained() raises ValueError with "does not support" message,
        the method should fall back to the next attention implementation.
        """
        model2 = Mock(name="success_model")
        model2.config = {}
        call_count = [0]
        attn_impls_seen = []

        def mock_from_pretrained_side_effect(*args, **kwargs):
            call_count[0] += 1
            attn_impl = kwargs.get("attn_implementation", "sdpa")
            attn_impls_seen.append(attn_impl)
            if attn_impl == "flash_attention_2":
                raise ValueError("Model does not support flash_attention_2 attention implementation")
            else:
                return model2

        with (
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=mock_from_pretrained_side_effect)),
            caplog.at_level(logging.WARNING)
        ):
            returned = NeMoAutoModelForCausalLM.from_pretrained(
                "hf-internal-testing/tiny-random-gpt2",
                attn_implementation="flash_attention_2"
            )

        # Verify the warning was logged
        assert "Falling back to sdpa attention." in caplog.text

        # Verify the final returned model is the successful one
        assert returned is model2

        # Verify the calls were made with correct attention implementations
        assert "flash_attention_2" in attn_impls_seen
        assert "sdpa" in attn_impls_seen

    def test_from_pretrained_valueerror_non_attention_reraises(self):
        """Test that ValueError not related to attention implementation is re-raised."""
        def mock_from_pretrained_side_effect(*args, **kwargs):
            raise ValueError("Some other error not related to attention")

        with (
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=mock_from_pretrained_side_effect)),
        ):
            with pytest.raises(ValueError, match="Some other error not related to attention"):
                NeMoAutoModelForCausalLM.from_pretrained(
                    "hf-internal-testing/tiny-random-gpt2",
                    attn_implementation="flash_attention_2"
                )

    def test_from_config_valueerror_attention_fallback(self, caplog):
        """Test ValueError exception handling in from_config when attention implementation is not supported.

        When wrapped_cls.from_config() raises ValueError with "does not support" message,
        the method should:
        1. Fall back to eager attention implementation
        2. Log a warning
        3. Retry with the fallback attention implementation
        """
        model2 = Mock(name="success_model")
        model2.config = {}
        attn_impls_seen = []

        # Mock the call sequence: first call fails with ValueError, second succeeds
        def mock_from_config_side_effect(*args, **kwargs):
            # Check the attn_implementation parameter to determine which call this is
            attn_impl = kwargs.get("attn_implementation", "flash_attention_2")
            attn_impls_seen.append(attn_impl)
            if attn_impl == "flash_attention_2":
                # First call with flash_attention_2 - should fail
                raise ValueError("Model does not support flash_attention_2 attention implementation")
            else:
                # Second call with fallback (eager) - should succeed
                return model2

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_config_side_effect=mock_from_config_side_effect)),
            caplog.at_level(logging.WARNING)
        ):
            # Test the exception path by starting with flash_attention_2
            returned = NeMoAutoModelForCausalLM.from_config(
                cfg,
                attn_implementation="flash_attention_2"
            )

        # Verify the warning was logged
        assert "Falling back to eager attention." in caplog.text

        # Verify the final returned model is the successful one
        assert returned is model2

        # Verify the calls were made with correct attention implementations
        assert "flash_attention_2" in attn_impls_seen
        assert "eager" in attn_impls_seen

    @pytest.mark.parametrize(
        "has_packed_sequence,is_hf_model,cp_size,expected_attn,expect_raises",
        [
            (True, True, 1, "flash_attention_2", None),
            (True, True, 2, None, ValueError),
            (True, False, 1, None, None),
            (True, False, 2, None, None),
            (False, True, 1, "flash_attention_2", None),
            (False, True, 2, "sdpa", None),
            (False, False, 1, None, None),
            (False, False, 2, None, None),
        ],
    )
    def test_packed_sequence_and_cp_overrides_from_pretrained(
        self, has_packed_sequence, is_hf_model, cp_size, expected_attn, expect_raises
    ):
        # Get a real config for HF model tests to avoid _model_mapping[Mock] KeyError
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        # Track attn_implementation values passed to from_pretrained
        attn_impls_seen = []

        def mock_from_pretrained_side_effect(*args, **kwargs):
            attn_impl = kwargs.get("attn_implementation")
            attn_impls_seen.append(attn_impl)
            mock_model = MagicMock()
            mock_model.config = {}
            return mock_model

        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch("nemo_automodel._transformers.auto_model.os.path.isdir", return_value=True),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", lambda obj: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=mock_from_pretrained_side_effect)),
        ):
            if is_hf_model:
                # Use real config to ensure _model_mapping lookup works
                mock_cfg_from_pretrained.return_value = real_config
                mock_registry.model_arch_name_to_cls = {}
            else:
                cfg = Mock()
                cfg.architectures = ["CustomArch"]
                mock_cfg_from_pretrained.return_value = cfg
                custom_model_instance = Mock()
                custom_model_instance.config = Mock()
                custom_cls = Mock()
                custom_cls.__name__ = "MockMockMock"
                custom_cls.from_pretrained = Mock(return_value=custom_model_instance)
                mock_registry.model_arch_name_to_cls = {"CustomArch": custom_cls}

            def do_call():
                return NeMoAutoModelForCausalLM.from_pretrained(
                    "dummy/model",
                    cp_size=cp_size,
                    has_packed_sequence=has_packed_sequence,
                )

            if expect_raises:
                with pytest.raises(expect_raises):
                    do_call()
                if not is_hf_model:
                    custom_cls = mock_registry.model_arch_name_to_cls["CustomArch"]
                    assert custom_cls.from_pretrained.call_count == 0
                return

            model = do_call()
            assert hasattr(model, "config")

            if is_hf_model:
                # Verify from_pretrained was called with expected attention implementation
                assert len(attn_impls_seen) >= 1
                if expected_attn is not None:
                    assert expected_attn in attn_impls_seen
            else:
                custom_cls = mock_registry.model_arch_name_to_cls["CustomArch"]
                # Custom models use from_pretrained method now
                assert custom_cls.from_pretrained.call_count == 1

    def test_trust_remote_code_whitelist_nvidia_from_pretrained(self):
        mock_model = MagicMock()
        mock_model.config = {}
        # Get a real config to avoid _model_mapping[Mock] KeyError
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch("nemo_automodel._transformers.auto_model.os.path.isdir", return_value=False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", lambda obj: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=mock_model)),
        ):
            mock_registry.model_arch_name_to_cls = {}
            mock_cfg_from_pretrained.return_value = real_config

            NeMoAutoModelForCausalLM.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")

            _, kwargs = mock_cfg_from_pretrained.call_args
            assert kwargs["trust_remote_code"] is True

    def test_trust_remote_code_respects_explicit_kwarg_from_pretrained(self):
        mock_model = MagicMock()
        mock_model.config = {}
        # Get a real config to avoid _model_mapping[Mock] KeyError
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model.ModelRegistry") as mock_registry,
            patch("nemo_automodel._transformers.auto_model.os.path.isdir", return_value=False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", lambda obj: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=_create_mock_wrapped_class(from_pretrained_side_effect=mock_model)),
        ):
            mock_registry.model_arch_name_to_cls = {}
            mock_cfg_from_pretrained.return_value = real_config

            NeMoAutoModelForCausalLM.from_pretrained("custom/model", trust_remote_code=False)

            _, kwargs = mock_cfg_from_pretrained.call_args
            assert kwargs["trust_remote_code"] is False


def _create_mock_model_mapping(mock_wrapped_class):
    """Create a mock _model_mapping that returns the given class for any config type."""
    class MockModelMapping:
        def __getitem__(self, key):
            return mock_wrapped_class
    return MockModelMapping()


class TestNeMoAutoModelForImageTextToText:
    """Test cases for NeMoAutoModelForImageTextToText class."""

    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        mock_model = Mock()
        mock_model.config = {}
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        mock_wrapped_class = _create_mock_wrapped_class(from_pretrained_side_effect=mock_model)

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=mock_wrapped_class),
            patch.object(NeMoAutoModelForImageTextToText, "_model_mapping",
                        _create_mock_model_mapping(mock_wrapped_class)),
        ):
            mock_cfg_from_pretrained.return_value = real_config

            # Test - warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model

    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_wrapped_class = _create_mock_wrapped_class(from_config_side_effect=mock_model)

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", False),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=mock_wrapped_class),
            patch.object(NeMoAutoModelForImageTextToText, "_model_mapping",
                        _create_mock_model_mapping(mock_wrapped_class)),
        ):
            config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

            # Test warning when HAS_LIGER_KERNEL is False
            with caplog.at_level(logging.WARNING):
                model = NeMoAutoModelForImageTextToText.from_config(config)

            assert "Asked to use Liger Kernel, but could not import" in caplog.text
            assert model is mock_model

    def test_from_pretrained_runtimeerror_triggers_reload(self):
        """When _patch_liger_kernel raises, the loader should retry with
        use_liger_kernel=False and return the second model instance."""
        # first and second dummy model objects
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}
        models = [model1, model2]
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        mock_wrapped_class = _create_mock_wrapped_class(from_pretrained_side_effect=models)

        patch_calls = []
        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=mock_wrapped_class),
            patch.object(NeMoAutoModelForImageTextToText, "_model_mapping",
                        _create_mock_model_mapping(mock_wrapped_class)),
        ):
            mock_cfg_from_pretrained.return_value = real_config
            returned = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")

        # _patch_liger_kernel called once, then retry without liger
        assert patch_calls == [model1]
        # The final object returned is the *second* model
        assert returned is model2


    def test_from_pretrained_sdpa_runtimeerror_triggers_reload(self):
        """When _patch_attention raises, the loader should retry with
        use_sdpa_patching=False and return the second model instance."""
        # first and second dummy model objects
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}
        models = [model1, model2]
        real_config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        mock_wrapped_class = _create_mock_wrapped_class(from_pretrained_side_effect=models)

        patch_calls = []
        def fake__patch_attention(model, sdpa_method):
            patch_calls.append(model)
            raise RuntimeError("boom")

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", lambda x: x),
            patch("nemo_automodel._transformers.auto_model._patch_attention", fake__patch_attention),
            patch("nemo_automodel._transformers.auto_model.AutoConfig.from_pretrained") as mock_cfg_from_pretrained,
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=mock_wrapped_class),
            patch.object(NeMoAutoModelForImageTextToText, "_model_mapping",
                        _create_mock_model_mapping(mock_wrapped_class)),
        ):
            mock_cfg_from_pretrained.return_value = real_config
            returned = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")

        # _patch_attention called once, then retry without sdpa patching
        assert patch_calls == [model1]
        # The final object returned is the *second* model
        assert returned is model2

    def test_from_config_runtimeerror_triggers_reload(self):
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}
        models = [model1, model2]
        mock_wrapped_class = _create_mock_wrapped_class(from_config_side_effect=models)

        patch_calls = []

        def fake__patch_liger_kernel(model):
            patch_calls.append(model)
            raise RuntimeError("boom")

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", new=fake__patch_liger_kernel),
            patch("nemo_automodel._transformers.auto_model._patch_attention", lambda obj, sdpa_method=None: obj),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=mock_wrapped_class),
            patch.object(NeMoAutoModelForImageTextToText, "_model_mapping",
                        _create_mock_model_mapping(mock_wrapped_class)),
        ):
            returned = NeMoAutoModelForImageTextToText.from_config(cfg)

        assert patch_calls == [model1]
        assert returned is model2

    def test_from_config_sdap_runtimeerror_triggers_reload(self):
        model1, model2 = Mock(name="m1"), Mock(name="m2")
        model1.config = {}
        model2.config = {}
        models = [model1, model2]
        mock_wrapped_class = _create_mock_wrapped_class(from_config_side_effect=models)

        patch_calls = []

        def fake__patch_attention(model, sdpa_method):
            patch_calls.append(model)
            raise RuntimeError("boom")

        cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with (
            patch("nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL", True),
            patch("nemo_automodel._transformers.auto_model._patch_liger_kernel", lambda x: x),
            patch("nemo_automodel._transformers.auto_model._patch_attention", fake__patch_attention),
            patch("nemo_automodel._transformers.auto_model._get_mixin_wrapped_class",
                  return_value=mock_wrapped_class),
            patch.object(NeMoAutoModelForImageTextToText, "_model_mapping",
                        _create_mock_model_mapping(mock_wrapped_class)),
        ):
            returned = NeMoAutoModelForImageTextToText.from_config(cfg)

        assert patch_calls == [model1]
        assert returned is model2

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

        with patch("nemo_automodel._transformers.auto_model.sdpa_kernel") as mock_sdpa_kernel:
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

        with patch("nemo_automodel._transformers.auto_model.sdpa_kernel") as mock_sdpa_kernel:
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
    import nemo_automodel._transformers.auto_model as tgt

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
    import nemo_automodel._transformers.auto_model as tgt

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
    import nemo_automodel._transformers.auto_model as tgt

    prepare_env(
        monkeypatch,
        tgt,
        has_liger=True,
        apply_ok=False,  # force failure
    )

    with pytest.raises(RuntimeError, match="Failed to patch model"):
        tgt._patch_liger_kernel(DummyModel())
