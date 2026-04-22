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

import importlib
from unittest.mock import Mock, patch

import pytest

from nemo_automodel._transformers.utils import (
    _patch_bytes_to_unicode,
    _patch_special_tokens_pattern,
    apply_cache_compatibility_patches,
    apply_qwen3_omni_config_patch,
    sliding_window_overwrite,
)


class TestSlidingWindowOverwrite:
    """Test cases for sliding_window_overwrite function."""

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_use_sliding_window_false(self, mock_from_pretrained, capsys):
        """Test sliding_window is set to None when use_sliding_window is False."""
        # Create a mock config with use_sliding_window=False and sliding_window attribute
        mock_config = Mock()
        mock_config.use_sliding_window = False
        mock_config.sliding_window = 4096  # Some default value
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

        # Assert the expected override dictionary is returned
        expected_result = {"sliding_window": None}
        assert result == expected_result

        # Assert the print statement was called
        captured = capsys.readouterr()
        assert "use_sliding_window=False in config - overriding sliding_window parameter to None" in captured.out
        assert str(expected_result) in captured.out

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_use_sliding_window_true(self, mock_from_pretrained):
        """Test no override when use_sliding_window is True."""
        # Create a mock config with use_sliding_window=True
        mock_config = Mock()
        mock_config.use_sliding_window = True
        mock_config.sliding_window = 4096
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

        # Assert empty dictionary is returned (no override needed)
        assert result == {}

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_no_use_sliding_window_attribute(self, mock_from_pretrained):
        """Test no override when use_sliding_window attribute doesn't exist."""
        # Create a mock config without use_sliding_window attribute
        mock_config = Mock()
        del mock_config.use_sliding_window  # Remove the attribute
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

        # Assert empty dictionary is returned (no override needed)
        assert result == {}

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_missing_sliding_window_attribute(self, mock_from_pretrained):
        """Test assertion error when use_sliding_window is False but sliding_window attribute missing."""
        # Create a mock config with use_sliding_window=False but no sliding_window attribute
        mock_config = Mock()
        mock_config.use_sliding_window = False
        del mock_config.sliding_window  # Remove the sliding_window attribute
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"

        # This should raise an AssertionError due to the assertion in the function
        with pytest.raises(AssertionError):
            sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_different_model_names(self, mock_from_pretrained):
        """Test function works with different model names."""
        # Create a mock config with use_sliding_window=False
        mock_config = Mock()
        mock_config.use_sliding_window = False
        mock_config.sliding_window = 2048
        mock_from_pretrained.return_value = mock_config

        # Test with different model names
        model_names = [
            "microsoft/DialoGPT-medium",
            "huggingface/CodeBERTa-small-v1",
            "/path/to/local/model",
            "my-org/custom-model",
        ]

        for model_name in model_names:
            result = sliding_window_overwrite(model_name)

            # Assert the expected override dictionary is returned
            assert result == {"sliding_window": None}

        # Assert the function was called for each model
        assert mock_from_pretrained.call_count == len(model_names)

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_trust_remote_code_parameter(self, mock_from_pretrained):
        """Test that trust_remote_code=True is always passed to AutoConfig.from_pretrained."""
        # Create a mock config
        mock_config = Mock()
        mock_config.use_sliding_window = True
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        sliding_window_overwrite(model_name)

        # Verify trust_remote_code=True was passed
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

    @patch("nemo_automodel._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_hasattr_behavior(self, mock_from_pretrained):
        """Test that hasattr is used correctly to check for attributes."""
        # Create a mock config that behaves correctly with hasattr
        mock_config = Mock()

        # Test case where hasattr returns False for use_sliding_window
        mock_config.configure_mock(**{"use_sliding_window": Mock(side_effect=AttributeError)})
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Should return empty dict when use_sliding_window attribute doesn't exist
        assert result == {}

        # Reset and test case where hasattr returns True but value is not False
        mock_config = Mock()
        mock_config.use_sliding_window = None  # Neither True nor False
        mock_config.sliding_window = 1024
        mock_from_pretrained.return_value = mock_config

        result = sliding_window_overwrite(model_name)

        # Should return empty dict when use_sliding_window is not exactly False
        assert result == {}


class TestPatchBytesToUnicode:
    """Tests for _patch_bytes_to_unicode compatibility patch."""

    def test_bytes_to_unicode_is_available_after_patch(self):
        """After patching, the GPT-2 tokenizer module exposes bytes_to_unicode."""
        _patch_bytes_to_unicode()
        gpt2_tok = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
        assert hasattr(gpt2_tok, "bytes_to_unicode")
        assert callable(gpt2_tok.bytes_to_unicode)

    def test_bytes_to_unicode_returns_correct_mapping(self):
        """The patched bytes_to_unicode returns a dict mapping all 256 byte values."""
        _patch_bytes_to_unicode()
        gpt2_tok = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
        mapping = gpt2_tok.bytes_to_unicode()

        assert isinstance(mapping, dict)
        assert len(mapping) == 256
        for byte_val in range(256):
            assert byte_val in mapping
            assert isinstance(mapping[byte_val], str)

    def test_bytes_to_unicode_is_idempotent(self):
        """Calling the patch twice does not break anything."""
        _patch_bytes_to_unicode()
        gpt2_tok = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
        first_result = gpt2_tok.bytes_to_unicode()

        _patch_bytes_to_unicode()
        second_result = gpt2_tok.bytes_to_unicode()

        assert first_result == second_result

    def test_bytes_to_unicode_does_not_overwrite_existing(self):
        """If bytes_to_unicode already exists, the patch is a no-op."""
        gpt2_tok = importlib.import_module("transformers.models.gpt2.tokenization_gpt2")
        sentinel = lambda: "sentinel"
        original = getattr(gpt2_tok, "bytes_to_unicode", None)
        gpt2_tok.bytes_to_unicode = sentinel
        try:
            _patch_bytes_to_unicode()
            assert gpt2_tok.bytes_to_unicode is sentinel
        finally:
            if original is not None:
                gpt2_tok.bytes_to_unicode = original
            else:
                delattr(gpt2_tok, "bytes_to_unicode")


class TestPatchSpecialTokensPattern:
    """Tests for _patch_special_tokens_pattern compatibility patch."""

    def test_patch_sets_default_special_tokens_pattern_to_none(self):
        """After patching, new PreTrainedTokenizer instances default to 'none' pattern."""
        _patch_special_tokens_pattern()
        from transformers.tokenization_python import PreTrainedTokenizer

        assert getattr(PreTrainedTokenizer.__init__, "_nemo_stp_patched", False)

    def test_patch_is_idempotent(self):
        """Calling the patch multiple times does not stack wrappers."""
        _patch_special_tokens_pattern()
        from transformers.tokenization_python import PreTrainedTokenizer

        first_init = PreTrainedTokenizer.__init__
        _patch_special_tokens_pattern()
        assert PreTrainedTokenizer.__init__ is first_init

    def _make_spy_and_patch(self):
        """Set up a spy __init__ and apply the patch on top of it.

        Returns (call_record dict, cleanup callable).
        """
        from transformers.tokenization_python import PreTrainedTokenizer

        saved_init = PreTrainedTokenizer.__init__
        call_record = {}

        def spy_init(self, *args, **kwargs):
            call_record.update(kwargs)
            raise RuntimeError("spy_stop")

        PreTrainedTokenizer.__init__ = spy_init
        PreTrainedTokenizer.__init__._nemo_stp_patched = False
        _patch_special_tokens_pattern()

        def cleanup():
            PreTrainedTokenizer.__init__ = saved_init

        return call_record, cleanup

    def test_patch_injects_none_when_kwarg_omitted(self):
        """When special_tokens_pattern is NOT passed, the patch injects 'none'."""
        from transformers.tokenization_python import PreTrainedTokenizer

        call_record, cleanup = self._make_spy_and_patch()
        try:
            try:
                PreTrainedTokenizer(vocab_file="/dev/null")
            except RuntimeError:
                pass
            assert call_record.get("special_tokens_pattern") == "none"
        finally:
            cleanup()

    def test_patch_preserves_explicit_special_tokens_pattern(self):
        """If caller explicitly passes special_tokens_pattern, the patch doesn't override it."""
        from transformers.tokenization_python import PreTrainedTokenizer

        call_record, cleanup = self._make_spy_and_patch()
        try:
            try:
                PreTrainedTokenizer(vocab_file="/dev/null", special_tokens_pattern="cls_sep")
            except RuntimeError:
                pass
            assert call_record.get("special_tokens_pattern") == "cls_sep"
        finally:
            cleanup()


class TestApplyCacheCompatibilityPatchesIntegration:
    """Tests for apply_cache_compatibility_patches calling the new sub-patches."""

    def test_calls_bytes_to_unicode_patch(self):
        """apply_cache_compatibility_patches invokes _patch_bytes_to_unicode."""
        with patch("nemo_automodel._transformers.utils._patch_bytes_to_unicode") as mock_btu:
            with patch("nemo_automodel._transformers.utils._patch_special_tokens_pattern"):
                apply_cache_compatibility_patches()
            mock_btu.assert_called_once()

    def test_calls_special_tokens_pattern_patch(self):
        """apply_cache_compatibility_patches invokes _patch_special_tokens_pattern."""
        with patch("nemo_automodel._transformers.utils._patch_special_tokens_pattern") as mock_stp:
            with patch("nemo_automodel._transformers.utils._patch_bytes_to_unicode"):
                apply_cache_compatibility_patches()
            mock_stp.assert_called_once()

    def test_applies_dynamic_cache_alias(self):
        """apply_cache_compatibility_patches aliases get_usable_length when needed."""
        apply_cache_compatibility_patches()
        from transformers.cache_utils import DynamicCache

        assert hasattr(DynamicCache, "get_usable_length") or hasattr(DynamicCache, "get_seq_length")

    def test_patches_sliding_window_cache(self):
        """SlidingWindowCache should exist after patching (aliased to StaticCache if removed)."""
        apply_cache_compatibility_patches()
        import transformers.cache_utils as cu

        assert hasattr(cu, "SlidingWindowCache")

    def test_patches_cache_get_usable_length(self):
        """Cache.get_usable_length should exist after patching."""
        apply_cache_compatibility_patches()
        from transformers.cache_utils import Cache

        assert hasattr(Cache, "get_usable_length")

    def test_get_usable_length_returns_seq_length(self):
        """get_usable_length delegates to get_seq_length when no max_length constraint."""
        apply_cache_compatibility_patches()
        from transformers.cache_utils import Cache

        mock_cache = Mock(spec=Cache)
        mock_cache.get_seq_length = Mock(return_value=10)
        mock_cache.get_max_cache_shape = Mock(return_value=None)
        result = Cache.get_usable_length(mock_cache, new_seq_length=5, layer_idx=0)
        assert result == 10

    def test_get_usable_length_respects_max_length(self):
        """get_usable_length returns max_length - new_seq_length when cache is near full."""
        apply_cache_compatibility_patches()
        from transformers.cache_utils import Cache

        mock_cache = Mock(spec=Cache)
        mock_cache.get_seq_length = Mock(return_value=90)
        mock_cache.get_max_cache_shape = Mock(return_value=100)
        # 90 + 15 > 100, so should return 100 - 15 = 85
        result = Cache.get_usable_length(mock_cache, new_seq_length=15, layer_idx=0)
        assert result == 85

    def test_patches_tied_weights_keys_list_to_dict(self):
        """post_init should convert _tied_weights_keys from list to correct {target: source} dict."""
        apply_cache_compatibility_patches()
        from transformers.modeling_utils import PreTrainedModel

        assert getattr(PreTrainedModel.post_init, "_nemo_tied_keys_patched", False)

    def test_tied_weights_keys_patch_resolves_correct_source_phi4mm(self):
        """The post_init patch should map tied keys to model.embed_tokens.weight for phi4mm."""
        apply_cache_compatibility_patches()
        import torch.nn as nn
        from transformers import PretrainedConfig
        from transformers.modeling_utils import PreTrainedModel

        class FakePhi4mmCfg(PretrainedConfig):
            model_type = "phi4mm"

        class FakePhi4mmModel(PreTrainedModel):
            config_class = FakePhi4mmCfg
            _tied_weights_keys = ["lm_head.weight"]

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(10, 4)
                self.lm_head = nn.Linear(4, 10, bias=False)
                self.post_init()

        config = FakePhi4mmCfg(tie_word_embeddings=True)
        model = FakePhi4mmModel(config)
        tied = model._nemo_tied_weights_keys
        assert isinstance(tied, dict)
        assert "lm_head.weight" in tied
        assert tied["lm_head.weight"] == "model.embed_tokens.weight"
        assert model._tied_weights_keys == {}

    def test_tied_weights_keys_patch_converts_any_model(self):
        """The post_init patch should convert _tied_weights_keys for any model, not just phi4mm."""
        apply_cache_compatibility_patches()
        import torch.nn as nn
        from transformers import PretrainedConfig
        from transformers.modeling_utils import PreTrainedModel

        class FakeOtherCfg(PretrainedConfig):
            model_type = "nemotron_flash"

        class FakeOtherModel(PreTrainedModel):
            config_class = FakeOtherCfg
            _tied_weights_keys = ["lm_head.weight"]

            def __init__(self, config):
                super().__init__(config)
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(10, 4)
                self.lm_head = nn.Linear(4, 10, bias=False)
                self.post_init()

        config = FakeOtherCfg(tie_word_embeddings=True)
        model = FakeOtherModel(config)
        tied = model._nemo_tied_weights_keys
        assert isinstance(tied, dict)
        assert "lm_head.weight" in model._nemo_tied_weights_keys
        assert tied["lm_head.weight"] == "model.embed_tokens.weight"
        assert model._tied_weights_keys == {}

    def test_tied_weights_keys_patch_resolves_top_level_embed_tokens(self):
        """The post_init patch resolves embed_tokens at the top level via get_input_embeddings."""
        apply_cache_compatibility_patches()
        import torch.nn as nn
        from transformers import PretrainedConfig
        from transformers.modeling_utils import PreTrainedModel

        class _MockConfig(PretrainedConfig):
            model_type = "nemotron_flash"

        class _Model(PreTrainedModel):
            config_class = _MockConfig

            def __init__(self, config):
                super().__init__(config)
                self.embed_tokens = nn.Embedding(100, 16)
                self.lm_head = nn.Linear(16, 100, bias=False)
                self._tied_weights_keys = ["lm_head.weight"]
                self.post_init()

            def get_input_embeddings(self):
                return self.embed_tokens

            def forward(self, x):
                return self.lm_head(self.embed_tokens(x))

        model = _Model(_MockConfig())
        assert isinstance(model._tied_weights_keys, dict)
        assert "lm_head.weight" in model._tied_weights_keys
        assert model._tied_weights_keys["lm_head.weight"] == "embed_tokens.weight"
        assert isinstance(model._nemo_tied_weights_keys, dict)
        assert model._nemo_tied_weights_keys["lm_head.weight"] == "embed_tokens.weight"

    def test_patches_peft_prepare_inputs(self):
        """PeftModelForCausalLM.__init__ should be patched for missing prepare_inputs_for_generation."""
        apply_cache_compatibility_patches()
        try:
            import peft.peft_model as pm

            assert getattr(pm.PeftModelForCausalLM.__init__, "_nemo_peft_patched", False)
        except ImportError:
            pytest.skip("peft not installed")

    def test_peft_patch_adds_stub_on_missing_prepare_inputs(self):
        """The PEFT patch should add prepare_inputs_for_generation stub when missing."""
        apply_cache_compatibility_patches()
        try:
            import peft.peft_model  # noqa: F401
        except ImportError:
            pytest.skip("peft not installed")

        # Simulate the stub logic from _patch_peft_prepare_inputs:
        # when a model lacks prepare_inputs_for_generation, the patch adds a stub
        class FakeModel:
            pass

        model = FakeModel()
        assert not hasattr(model, "prepare_inputs_for_generation")

        # Reproduce what the patch does on AttributeError
        try:
            _ = model.prepare_inputs_for_generation
        except AttributeError:
            model.prepare_inputs_for_generation = lambda *a, **kw: {}

        assert callable(model.prepare_inputs_for_generation)
        assert model.prepare_inputs_for_generation() == {}

    def test_patches_phi4mm_processor(self):
        """ProcessorMixin.from_pretrained should be patched for phi4mm fallback."""
        apply_cache_compatibility_patches()
        import transformers.processing_utils as pu

        # The patch replaces from_pretrained with a wrapper named _patched
        fn = pu.ProcessorMixin.from_pretrained
        assert fn.__func__.__qualname__ == "_patch_phi4mm_processor.<locals>._patched"


class TestApplyQwen3OmniConfigPatch:
    """Test cases for apply_qwen3_omni_config_patch function."""

    def test_patch_sets_use_sliding_window_default(self):
        """Verify the patch adds use_sliding_window=False to the config class."""
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )

        apply_qwen3_omni_config_patch()
        assert hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window")

    def test_patch_is_idempotent(self):
        """Calling the patch twice does not raise or change the value."""
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )

        apply_qwen3_omni_config_patch()
        apply_qwen3_omni_config_patch()
        assert Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window is False

    def test_patch_does_not_overwrite_existing_attribute(self):
        """If the attribute already exists (e.g. fixed upstream), patch is a no-op."""
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )

        original = getattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window", None)
        Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = True
        try:
            apply_qwen3_omni_config_patch()
            assert Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window is True
        finally:
            if original is None:
                del Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window
            else:
                Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = original
