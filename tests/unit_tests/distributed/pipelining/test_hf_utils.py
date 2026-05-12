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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.distributed.pipelining.hf_utils import (
    create_pipeline_forward_causal_lm,
    create_pipeline_forward_inner,
    init_hf_model_buffers,
    model_keeps_self_forward,
    patch_hf_model_for_pp,
    validate_hf_model_for_pipeline_support,
)


class TestCreatePipelineForwardInner:
    """Test create_pipeline_forward_inner function."""

    def test_returns_callable(self):
        forward_fn = create_pipeline_forward_inner("AutoModel")
        assert callable(forward_fn)

    @patch("torch.arange")
    def test_forward_with_embeddings(self, mock_arange):
        # Create mock model with embeddings
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False, use_cache=True)
        mock_model.gradient_checkpointing = False

        # Mock embed_tokens
        mock_embed_tokens = Mock()
        mock_embed_tokens.return_value = torch.randn(1, 10, 768)
        mock_model.embed_tokens = mock_embed_tokens

        # Layers as nn.ModuleDict with nn.Module children (not plain Mocks)
        class DummyLayer(nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states

        mock_model.layers = nn.ModuleDict({"0": DummyLayer()})

        # Mock norm
        mock_norm = Mock()
        mock_norm.return_value = torch.randn(1, 10, 768)
        mock_model.norm = mock_norm

        # Mock rotary_emb
        mock_rotary = Mock()
        mock_rotary.return_value = (torch.randn(1, 10, 768), torch.randn(1, 10, 768))
        mock_model.rotary_emb = mock_rotary

        # Setup mock arange
        mock_arange.return_value = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # Get forward function and bind to model
        forward_fn = create_pipeline_forward_inner("AutoModel")

        # Call forward
        input_ids = torch.randint(0, 1000, (1, 10))
        output = forward_fn(mock_model, input_ids=input_ids)

        # Verify embed_tokens was called
        mock_embed_tokens.assert_called_once_with(input_ids)

        # Verify output type
        assert isinstance(output, BaseModelOutputWithPast)

    def test_forward_without_embeddings(self):
        # Create mock model without embeddings
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False, use_cache=False)
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.layers = None
        mock_model.norm = None
        mock_model.rotary_emb = None

        forward_fn = create_pipeline_forward_inner("PipelineStage")

        # Should expect inputs_embeds for stages without embed_tokens
        inputs_embeds = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=inputs_embeds)

        # For PipelineStage, should return tensor directly
        assert isinstance(output, torch.Tensor)

    def test_forward_with_float_input_ids(self):
        # Test when input_ids is actually hidden states (float type)
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False, use_cache=False)
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.layers = None
        mock_model.norm = None
        mock_model.rotary_emb = None

        forward_fn = create_pipeline_forward_inner("PipelineStage")

        # Pass float tensor as input_ids
        float_input = torch.randn(1, 10, 768).half()
        output = forward_fn(mock_model, input_ids=float_input)

        assert isinstance(output, torch.Tensor)


class TestCreatePipelineForwardCausalLM:
    """Test create_pipeline_forward_causal_lm function."""

    def test_returns_callable(self):
        forward_fn = create_pipeline_forward_causal_lm()
        assert callable(forward_fn)

    def test_forward_with_inner_model(self):
        # Create mock causal LM model
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)

        # Mock inner model
        mock_inner = Mock()
        mock_inner.return_value = BaseModelOutputWithPast(last_hidden_state=torch.randn(1, 10, 768))
        mock_model.model = mock_inner

        # Mock lm_head
        mock_lm_head = Mock()
        mock_lm_head.return_value = torch.randn(1, 10, 1000)
        mock_model.lm_head = mock_lm_head

        forward_fn = create_pipeline_forward_causal_lm()

        input_ids = torch.randint(0, 1000, (1, 10))
        output = forward_fn(mock_model, input_ids=input_ids)

        # Verify inner model was called
        mock_inner.assert_called_once()
        # Verify lm_head was called
        mock_lm_head.assert_called_once()

        assert isinstance(output, torch.Tensor)

    def test_forward_without_inner_model(self):
        # Create mock without inner model (pipeline stage)
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)
        mock_model.model = None
        mock_model.lm_head = None

        forward_fn = create_pipeline_forward_causal_lm()

        # Pass hidden states as inputs_embeds
        hidden_states = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=hidden_states)

        # Should return hidden states as-is
        assert torch.equal(output, hidden_states)

    def test_forward_with_logits_to_keep(self):
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)
        mock_model.model = None

        # Mock lm_head
        mock_lm_head = Mock()
        mock_lm_head.return_value = torch.randn(1, 5, 1000)
        mock_model.lm_head = mock_lm_head

        forward_fn = create_pipeline_forward_causal_lm()

        hidden_states = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=hidden_states, logits_to_keep=5)

        # Verify lm_head was called with sliced hidden states
        called_hidden = mock_lm_head.call_args[0][0]
        assert called_hidden.shape[1] == 5  # Only last 5 positions

    def test_forward_with_non_basemodel_output(self):
        """Test handling when inner model returns non-BaseModelOutputWithPast."""
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)

        # Mock inner model that returns tensor directly
        mock_inner = Mock()
        hidden_tensor = torch.randn(1, 10, 768)
        mock_inner.return_value = hidden_tensor  # Return tensor, not BaseModelOutputWithPast
        mock_model.model = mock_inner

        # Mock lm_head
        mock_lm_head = Mock()
        mock_lm_head.return_value = torch.randn(1, 10, 1000)
        mock_model.lm_head = mock_lm_head

        forward_fn = create_pipeline_forward_causal_lm()

        input_ids = torch.randint(0, 1000, (1, 10))
        output = forward_fn(mock_model, input_ids=input_ids)

        # Verify inner model was called
        mock_inner.assert_called_once()
        # Verify lm_head was called with the tensor output
        mock_lm_head.assert_called_once()

        assert isinstance(output, torch.Tensor)

    def test_forward_with_float_input_ids_causal_lm(self):
        """Test handling float input_ids in causal LM without inner model."""
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)
        mock_model.model = None
        mock_model.lm_head = None

        forward_fn = create_pipeline_forward_causal_lm()

        # Pass float tensor as input_ids
        float_input = torch.randn(1, 10, 768).half()
        output = forward_fn(mock_model, input_ids=float_input)

        # Should return the float input as-is
        assert torch.equal(output, float_input)

    def test_forward_invalid_input_causal_lm(self):
        """Test error when invalid input provided to causal LM stage."""
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)
        mock_model.model = None
        mock_model.lm_head = None

        forward_fn = create_pipeline_forward_causal_lm()

        # Provide invalid input_ids (integer tensor) without inputs_embeds
        input_ids = torch.randint(0, 1000, (1, 10))  # Integer tensor

        # Should raise ValueError
        with pytest.raises(ValueError, match="Expected hidden states as input for pipeline stage without inner model"):
            forward_fn(mock_model, input_ids=input_ids)


class TestPatchHfModelForPp:
    """Test patch_hf_model_for_pp function."""

    def test_patch_model_with_inner_model(self):
        """Test patching model that has inner .model attribute."""

        # Create model with inner model
        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()

        model = OuterModel()
        original_forward = model.forward
        original_inner_forward = model.model.forward

        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        # Both forwards should be patched
        assert model.forward != original_forward
        assert model.model.forward != original_inner_forward

    def test_patch_model_without_inner_model(self):
        """Test patching model without inner .model attribute."""
        model = nn.Module()
        original_forward = model.forward

        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=False)

        # Only model forward should be patched
        assert model.forward != original_forward

    def test_patch_model_selective_patching(self):
        """Test selective patching with flags."""

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()

        model = OuterModel()
        original_forward = model.forward
        original_inner_forward = model.model.forward

        # Only patch inner model
        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=False)

        # Only inner forward should be patched
        assert model.forward == original_forward
        assert model.model.forward != original_inner_forward

    def test_patch_model_with_none_inner(self):
        """Test patching when model.model is None."""

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = None

        model = OuterModel()
        original_forward = model.forward

        # Should not crash when model.model is None
        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        # Outer forward should still be patched
        assert model.forward != original_forward

    def test_patch_gemma4_vlm_uses_gemma4_forward(self):
        """Gemma4 VLM (config.model_type == 'gemma4') gets the Gemma4-specific forwards."""

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()

        class _Gemma4(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock(model_type="gemma4")
                self.model = _Inner()

        model = _Gemma4()
        original_outer = model.forward
        original_text_backbone = model.model.language_model.forward

        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        # The text backbone is the one patched (not model.model itself).
        assert model.model.language_model.forward is not original_text_backbone
        assert model.forward is not original_outer
        # Sanity: the Gemma4 forward is bound to the text backbone and outer VLM.
        assert model.model.language_model.forward.__func__.__name__ == "pipeline_forward_gemma4_text"
        assert model.forward.__func__.__name__ == "pipeline_forward_gemma4_vlm"

    def test_patch_non_gemma4_vlm_falls_back_to_generic(self):
        """VLMs that happen to expose model.language_model but are NOT Gemma4 use the generic path."""

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                # Many HF VLMs (KimiVL / Mistral4 / Qwen3VL MoE / LlavaOneVision / ...)
                # also expose language_model here. These must NOT hit Gemma4's forward.
                self.language_model = nn.Module()

        class _OtherVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock(model_type="llava_onevision", text_config=None)
                self.model = _Inner()

        model = _OtherVLM()
        original_inner = model.model.forward
        original_outer = model.forward

        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        # Generic path patches model.model (inner) directly, not language_model.
        assert model.model.forward is not original_inner
        assert model.forward is not original_outer
        assert model.model.forward.__func__.__name__ == "pipeline_forward"
        assert model.forward.__func__.__name__ == "pipeline_forward_causal_lm"

    def test_patch_gemma4_vlm_via_text_config_model_type(self):
        """Gemma4 detection also works when model_type is only in config.text_config."""

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()

        class _Gemma4(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock(model_type="gemma4_vision", text_config=Mock(model_type="gemma4"))
                self.model = _Inner()

        model = _Gemma4()
        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        assert model.model.language_model.forward.__func__.__name__ == "pipeline_forward_gemma4_text"
        assert model.forward.__func__.__name__ == "pipeline_forward_gemma4_vlm"

    def test_model_keeps_self_forward_helper(self):
        """``model_keeps_self_forward`` reflects the class-level opt-out flag.

        Regression for the silent-vision bug where chunk-aware VLMs (Qwen3-VL-MoE,
        KimiVL, Kimi-K2.5-VL, Qwen3.5-MoE) had their pixel_values-fetching forward
        replaced by the generic CausalLM forward, causing vision_tower to never
        run. The fix splits responsibility: the model class declares the flag,
        and the pipeline build call site uses ``model_keeps_self_forward`` to
        decide whether to invoke ``patch_hf_model_for_pp`` at all.
        """

        class _OptedIn(nn.Module):
            _pp_keep_self_forward = True

        class _Default(nn.Module):
            pass

        assert model_keeps_self_forward(_OptedIn()) is True
        assert model_keeps_self_forward(_Default()) is False


class TestInitHfModelBuffers:
    """Test init_hf_model_buffers function."""

    def test_init_buffers_with_rotary_emb(self):
        """Test buffer initialization for model with rotary embeddings."""

        class MockRotaryEmb(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock()

            def rope_init_fn(self, config, device):
                inv_freq = torch.randn(64)  # Mock inv_freq
                return inv_freq, None

            def register_buffer(self, name, tensor, persistent=False):
                # Mock register_buffer
                setattr(self, name, tensor)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.rotary_emb = MockRotaryEmb()

        model = MockModel()
        device = torch.device("cpu")

        # Should not raise error
        init_hf_model_buffers(model, device)

        # Verify buffer was registered (mock implementation sets attribute)
        assert hasattr(model.model.rotary_emb, "inv_freq")

    def test_init_buffers_without_rotary_emb(self):
        """Test buffer initialization for model without rotary embeddings."""
        model = nn.Module()
        device = torch.device("cpu")

        # Should not raise error
        init_hf_model_buffers(model, device)

    def test_init_buffers_with_direct_rotary_emb(self):
        """Test buffer initialization when rotary_emb is directly on model."""

        class MockRotaryEmb(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock()

            def rope_init_fn(self, config, device):
                inv_freq = torch.randn(64)  # Mock inv_freq
                return inv_freq, None

            def register_buffer(self, name, tensor, persistent=False):
                # Mock register_buffer
                setattr(self, name, tensor)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rotary_emb = MockRotaryEmb()

        model = MockModel()
        device = torch.device("cpu")

        # Should not raise error
        init_hf_model_buffers(model, device)

        # Verify buffer was registered (mock implementation sets attribute)
        assert hasattr(model.rotary_emb, "inv_freq")


class TestValidateHfModelForPipelineSupport:
    """Test validate_hf_model_for_pipeline_support function."""

    def test_validate_valid_model(self):
        """Test validation of compatible model."""

        class MockConfig:
            pretrained_model_name_or_path = "test/model"
            tie_word_embeddings = False
            is_encoder_decoder = False

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()

        model = MockModel()

        # Should not raise any error
        validate_hf_model_for_pipeline_support(model)

    def test_validate_model_with_tied_embeddings(self):
        """Validation fails only when lm_head and embed_tokens actually share storage."""

        class MockConfig:
            pretrained_model_name_or_path = "test/model"
            tie_word_embeddings = True  # Needed to enable the tied-weights check
            is_encoder_decoder = False

        class _Inner(nn.Module):
            def __init__(self, shared_embed):
                super().__init__()
                self.embed_tokens = shared_embed

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.lm_head = nn.Linear(4, 4, bias=False)
                self.model = _Inner(nn.Embedding(4, 4))
                # Actually tie the weights so the validator's stricter check triggers.
                self.model.embed_tokens.weight = self.lm_head.weight

        model = MockModel()

        with pytest.raises(ValueError, match="tie_word_embeddings=True is not supported"):
            validate_hf_model_for_pipeline_support(model)

    def test_validate_encoder_decoder_model(self):
        """Test validation fails for encoder-decoder model."""

        class MockConfig:
            pretrained_model_name_or_path = "test/model"
            tie_word_embeddings = False
            is_encoder_decoder = True  # This should cause validation to fail

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()

        model = MockModel()

        with pytest.raises(ValueError, match="Encoder-Decoder models with cross-attention are not supported"):
            validate_hf_model_for_pipeline_support(model)

    def test_validate_multiple_issues(self):
        """Test validation with multiple issues."""

        class MockConfig:
            pretrained_model_name_or_path = "test/model"
            tie_word_embeddings = True  # Issue 1 (only fires when weights are actually tied)
            is_encoder_decoder = True  # Issue 2

        class _Inner(nn.Module):
            def __init__(self, shared_embed):
                super().__init__()
                self.embed_tokens = shared_embed

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.lm_head = nn.Linear(4, 4, bias=False)
                self.model = _Inner(nn.Embedding(4, 4))
                self.model.embed_tokens.weight = self.lm_head.weight

        model = MockModel()

        with pytest.raises(ValueError) as exc_info:
            validate_hf_model_for_pipeline_support(model)

        error_msg = str(exc_info.value)
        # Should contain both issues
        assert "tie_word_embeddings=True" in error_msg
        assert "Encoder-Decoder models" in error_msg
        assert "1." in error_msg  # First issue
        assert "2." in error_msg  # Second issue

    def test_validate_model_without_config(self):
        """Test validation of model without config."""
        model = nn.Module()  # No config attribute

        # Should not raise any error
        validate_hf_model_for_pipeline_support(model)

    def test_validate_model_with_empty_config(self):
        """Test validation of model with empty config."""

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = object()  # Empty config without relevant attributes

        model = MockModel()

        # Should not raise any error (getattr with default False)
        validate_hf_model_for_pipeline_support(model)

    def test_validate_unsupported_vlm_pp_combination_raises(self):
        """VLMs without dedicated PP forward AND without _pp_keep_self_forward must fail validation."""

        class _TextCfg:
            tie_word_embeddings = False

        class MockConfig:
            pretrained_model_name_or_path = "test/some_vlm"
            tie_word_embeddings = False
            is_encoder_decoder = False
            model_type = "some_unknown_vlm"
            text_config = _TextCfg()

        class MockVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.vision_tower = nn.Linear(4, 4)

        model = MockVLM()

        with pytest.raises(ValueError, match="not on the pipeline-aware list"):
            validate_hf_model_for_pipeline_support(model)

    def test_validate_chunk_aware_vlm_passes(self):
        """VLMs that opt into _pp_keep_self_forward must pass validation."""

        class _TextCfg:
            tie_word_embeddings = False

        class MockConfig:
            pretrained_model_name_or_path = "test/qwen3_vl_moe"
            tie_word_embeddings = False
            is_encoder_decoder = False
            model_type = "qwen3_vl_moe"
            text_config = _TextCfg()

        class MockVLM(nn.Module):
            _pp_keep_self_forward = True

            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.vision_tower = nn.Linear(4, 4)

        model = MockVLM()
        # Should not raise.
        validate_hf_model_for_pipeline_support(model)

    def test_validate_dedicated_vlm_passes(self):
        """VLMs on the dedicated-forward list (gemma4 / mistral3) must pass validation."""

        class _TextCfg:
            tie_word_embeddings = False
            model_type = "gemma4"

        class MockConfig:
            pretrained_model_name_or_path = "test/gemma4"
            tie_word_embeddings = False
            is_encoder_decoder = False
            model_type = "gemma4_vlm"
            text_config = _TextCfg()

        class MockVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.vision_tower = nn.Linear(4, 4)

        model = MockVLM()
        validate_hf_model_for_pipeline_support(model)

    def test_no_gradient_checkpointing_warning(self):
        """No warning should be emitted; past_key_values remains None by default."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.gradient_checkpointing = True
        mock_model.training = True
        mock_model.embed_tokens = None
        mock_model.layers = None
        mock_model.norm = None
        mock_model.rotary_emb = None

        forward_fn = create_pipeline_forward_inner("AutoModel")

        inputs_embeds = torch.randn(1, 10, 768)

        with patch("nemo_automodel.components.distributed.pipelining.hf_utils.logger") as mock_logger:
            output = forward_fn(mock_model, inputs_embeds=inputs_embeds)
            # No warning should be called in the new style
            assert not mock_logger.warning_once.called

        assert isinstance(output, BaseModelOutputWithPast)
        assert output.past_key_values is None

    def test_missing_input_error(self):
        """Test error when neither input_ids nor inputs_embeds provided with embed_tokens."""
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False, use_cache=False)
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = Mock()
        mock_model.layers = None
        mock_model.norm = None
        mock_model.rotary_emb = None

        forward_fn = create_pipeline_forward_inner("AutoModel")

        # Should raise ValueError when no inputs provided
        with pytest.raises(ValueError, match="You must provide either input_ids or inputs_embeds"):
            forward_fn(mock_model)

    def test_invalid_inputs_embeds_error(self):
        """Test error when inputs_embeds not provided for stage without embed_tokens."""
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False, use_cache=False)
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.layers = None
        mock_model.norm = None
        mock_model.rotary_emb = None

        forward_fn = create_pipeline_forward_inner("PipelineStage")

        # Provide invalid input_ids (integer tensor)
        input_ids = torch.randint(0, 1000, (1, 10))

        # Should raise ValueError
        with pytest.raises(ValueError, match="inputs_embeds must be provided for pipeline stages without embed_tokens"):
            forward_fn(mock_model, input_ids=input_ids)

    def test_hidden_states_not_collected(self):
        """Hidden states are not collected in the new inner forward."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.rotary_emb = None
        mock_model.norm = None

        class DummyLayer(nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states + 1

        mock_model.layers = nn.ModuleList([DummyLayer(), DummyLayer()])

        forward_fn = create_pipeline_forward_inner("AutoModel")

        inputs_embeds = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=inputs_embeds)

        assert isinstance(output, BaseModelOutputWithPast)
        assert output.hidden_states is None

    def test_attention_type_handling(self):
        """Test attention type handling for layers."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.rotary_emb = None
        mock_model.norm = None

        # Create layer with attention_type attribute
        class DummyLayerWithAttentionType(nn.Module):
            def __init__(self, attention_type):
                super().__init__()
                self.attention_type = attention_type

            def forward(self, hidden_states, attention_mask=None, **kwargs):
                return hidden_states

        layer = DummyLayerWithAttentionType("sliding_attention")
        mock_model.layers = nn.ModuleList([layer])

        # Mock the masking functions and create causal_mask_mapping
        with (
            patch("transformers.masking_utils.create_causal_mask") as mock_create_causal,
            patch("transformers.masking_utils.create_sliding_window_causal_mask") as mock_create_sliding,
        ):
            mock_create_causal.return_value = torch.ones(1, 1, 10, 10)
            mock_create_sliding.return_value = torch.ones(1, 1, 10, 10) * 2

            forward_fn = create_pipeline_forward_inner("AutoModel")

            inputs_embeds = torch.randn(1, 10, 768)
            attention_mask = torch.ones(1, 10)

            # Mock has_sliding_layers to trigger sliding window creation
            mock_model.has_sliding_layers = True

            output = forward_fn(mock_model, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            assert isinstance(output, BaseModelOutputWithPast)
            assert "inputs_embeds" in mock_create_causal.call_args.kwargs
            assert "input_embeds" not in mock_create_causal.call_args.kwargs
            assert "inputs_embeds" in mock_create_sliding.call_args.kwargs
            assert "input_embeds" not in mock_create_sliding.call_args.kwargs

    def test_attentions_not_collected(self):
        """Attentions are not collected in the new inner forward."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.rotary_emb = None
        mock_model.norm = None

        class DummyLayer(nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states

        mock_model.layers = nn.ModuleList([DummyLayer(), DummyLayer()])

        forward_fn = create_pipeline_forward_inner("AutoModel")

        inputs_embeds = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=inputs_embeds)

        assert isinstance(output, BaseModelOutputWithPast)
        assert output.attentions is None

    @patch("nemo_automodel.components.distributed.pipelining.hf_utils.get_text_module")
    def test_rotary_emb_via_get_text_module(self, mock_get_text_module):
        """Test that rotary_emb is accessed via get_text_module for multimodal model support."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.norm = None
        mock_model.layers = None

        # Create a mock text module with rotary_emb. The pipeline_forward now
        # routes embed_tokens / layers / norm through the text module too, so
        # explicitly stub them out to None to skip those branches.
        mock_text_module = Mock()
        mock_rotary = Mock()
        mock_rotary.return_value = (torch.randn(1, 10, 64), torch.randn(1, 10, 64))
        mock_text_module.rotary_emb = mock_rotary
        mock_text_module.embed_tokens = None
        mock_text_module.layers = None
        mock_text_module.norm = None

        mock_get_text_module.return_value = mock_text_module

        forward_fn = create_pipeline_forward_inner("AutoModel")

        inputs_embeds = torch.randn(1, 10, 768)
        position_ids = torch.arange(10).unsqueeze(0)
        output = forward_fn(mock_model, inputs_embeds=inputs_embeds, position_ids=position_ids)

        # Verify get_text_module was called with the model
        mock_get_text_module.assert_called_with(mock_model)

        # Verify rotary_emb was called
        mock_rotary.assert_called_once()

    @patch("nemo_automodel.components.distributed.pipelining.hf_utils.get_text_module")
    def test_rotary_emb_none_via_get_text_module(self, mock_get_text_module):
        """Test that None rotary_emb from get_text_module is handled correctly."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.norm = None
        mock_model.layers = None

        # Create a mock text module with None rotary_emb. Stub out the text
        # module's embed_tokens / layers / norm too (now routed through text
        # module by pipeline_forward).
        mock_text_module = Mock()
        mock_text_module.rotary_emb = None
        mock_text_module.embed_tokens = None
        mock_text_module.layers = None
        mock_text_module.norm = None

        mock_get_text_module.return_value = mock_text_module

        forward_fn = create_pipeline_forward_inner("AutoModel")

        inputs_embeds = torch.randn(1, 10, 768)
        # Should not raise error when rotary_emb is None
        output = forward_fn(mock_model, inputs_embeds=inputs_embeds)

        assert isinstance(output, BaseModelOutputWithPast)


# -----------------------------------------------------------------------------
# Tests for get_text_module, TEXT_MODULE_ATTRS, MULTIMODAL_SUFFIXES
# -----------------------------------------------------------------------------

from nemo_automodel.components.distributed.pipelining.hf_utils import (
    MULTIMODAL_SUFFIXES,
    TEXT_MODULE_ATTRS,
    get_text_module,
)


class TestGetTextModule:
    """Tests for get_text_module function."""

    def test_returns_language_model_when_present(self):
        """Test that language_model attribute is returned when present."""

        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Linear(10, 10)
                self.visual = nn.Linear(5, 5)

        model = VLMModel()
        result = get_text_module(model)
        assert result is model.language_model

    def test_returns_text_model_when_present(self):
        """Test that text_model attribute is returned when present."""

        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_model = nn.Linear(10, 10)
                self.vision_encoder = nn.Linear(5, 5)

        model = VLMModel()
        result = get_text_module(model)
        assert result is model.text_model

    def test_returns_text_decoder_when_present(self):
        """Test that text_decoder attribute is returned when present."""

        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_decoder = nn.Linear(10, 10)

        model = VLMModel()
        result = get_text_module(model)
        assert result is model.text_decoder

    def test_returns_model_when_no_text_attr(self):
        """Test that model itself is returned when no text module attribute exists."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Linear(10, 10)

        model = SimpleModel()
        result = get_text_module(model)
        assert result is model

    def test_returns_none_when_model_is_none(self):
        """Test that None is returned when model is None."""
        result = get_text_module(None)
        assert result is None

    def test_priority_order_language_model_first(self):
        """Test that language_model has priority over text_model."""

        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Linear(10, 10)
                self.text_model = nn.Linear(5, 5)

        model = VLMModel()
        result = get_text_module(model)
        assert result is model.language_model

    def test_skips_none_attribute(self):
        """Test that None attributes are skipped."""

        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = None
                self.text_model = nn.Linear(10, 10)

        model = VLMModel()
        result = get_text_module(model)
        assert result is model.text_model


class TestConstants:
    """Tests for TEXT_MODULE_ATTRS and MULTIMODAL_SUFFIXES constants."""

    def test_text_module_attrs_contains_expected_values(self):
        """Test TEXT_MODULE_ATTRS contains the expected attribute names."""
        assert "language_model" in TEXT_MODULE_ATTRS
        assert "text_model" in TEXT_MODULE_ATTRS
        assert "text_decoder" in TEXT_MODULE_ATTRS

    def test_multimodal_suffixes_contains_vision_attrs(self):
        """Test MULTIMODAL_SUFFIXES contains vision-related suffixes."""
        assert "vision_tower" in MULTIMODAL_SUFFIXES
        assert "visual" in MULTIMODAL_SUFFIXES
        assert "image_encoder" in MULTIMODAL_SUFFIXES
        assert "vision_encoder" in MULTIMODAL_SUFFIXES

    def test_multimodal_suffixes_contains_audio_attrs(self):
        """Test MULTIMODAL_SUFFIXES contains audio-related suffixes."""
        assert "audio_tower" in MULTIMODAL_SUFFIXES
        assert "audio_encoder" in MULTIMODAL_SUFFIXES
        assert "audio_model" in MULTIMODAL_SUFFIXES

    def test_multimodal_suffixes_contains_projector_attrs(self):
        """Test MULTIMODAL_SUFFIXES contains projector-related suffixes."""
        assert "mm_projector" in MULTIMODAL_SUFFIXES
        assert "multi_modal_projector" in MULTIMODAL_SUFFIXES
        assert "multimodal_projector" in MULTIMODAL_SUFFIXES


# --------------------------------------------------------------------------- #
# Mistral3 VLM PP additions                                                   #
# --------------------------------------------------------------------------- #
class TestIsMistral3Vlm:
    """`_is_mistral3_vlm` predicate gates the Mistral3 PP forward dispatch."""

    def test_model_type_mistral3_returns_true(self):
        from nemo_automodel.components.distributed.pipelining.hf_utils import _is_mistral3_vlm

        m = Mock()
        m.config = Mock(model_type="mistral3")
        assert _is_mistral3_vlm(m) is True

    def test_model_type_other_returns_false(self):
        from nemo_automodel.components.distributed.pipelining.hf_utils import _is_mistral3_vlm

        m = Mock()
        m.config = Mock(model_type="llama")
        assert _is_mistral3_vlm(m) is False

    def test_no_config_returns_false(self):
        from nemo_automodel.components.distributed.pipelining.hf_utils import _is_mistral3_vlm

        # Use a real object without `config` to avoid Mock auto-attributing.
        class _Bare:
            pass

        assert _is_mistral3_vlm(_Bare()) is False


class TestPatchHfModelForPpMistral3:
    """`patch_hf_model_for_pp` dispatches Mistral3 VLM to the dedicated forward."""

    def test_patch_mistral3_vlm_uses_mistral3_forward(self):
        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Module()

        class _Mistral3(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock(model_type="mistral3")
                self.model = _Inner()

        model = _Mistral3()
        original_outer = model.forward
        original_text_backbone = model.model.language_model.forward

        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        # Outer is patched with Mistral3 VLM forward.
        assert model.forward is not original_outer
        assert model.forward.__func__.__name__ == "pipeline_forward_mistral3_vlm"
        # Text backbone gets the generic inner forward.
        assert model.model.language_model.forward is not original_text_backbone
        assert model.model.language_model.forward.__func__.__name__ == "pipeline_forward"


class TestPipelineForwardMistral3Vlm:
    """`create_pipeline_forward_mistral3_vlm` returns a forward that:

    1. Retrieves pixel_values from `_vlm_pixel_values_chunks` when
       called with pixel_values=None and the chunks are pre-staged
       (this was the bug fix that dropped step-0 loss 6.6 → 3.2).
    2. Calls `vision_tower` + `multi_modal_projector` on stage 0 when
       pixel_values are present.
    3. Resolves `vision_feature_layer` from config when None
       (HF outer forward does this via @merge_with_config_defaults; we
       bypass that decorator).
    4. Skips vision path on non-first stages.
    """

    def _make_first_stage_model(self, image_token_id=10, vision_feature_layer_default=-1):
        """Construct a stage-0 mock with vision_tower, mm_projector, embed_tokens."""
        from nemo_automodel.components.distributed.pipelining.hf_utils import (
            create_pipeline_forward_mistral3_vlm,
        )

        class _Stage0(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock(
                    model_type="mistral3",
                    image_token_id=image_token_id,
                    vision_feature_layer=vision_feature_layer_default,
                )
                # The patched forward reads self.model.{language_model,
                # vision_tower, get_image_features, get_placeholder_mask}.
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                # Real embed_tokens so the first-stage branch fires.
                self.model.language_model.embed_tokens = nn.Embedding(32, 4)
                # vision_tower presence (any non-None object passes the gate).
                self.model.vision_tower = Mock()
                # Spy callables for image-features path.
                self.model.get_image_features = Mock()
                self.model.get_placeholder_mask = Mock()
                # Lang model forward — return a plain tensor (last_hidden_state path).
                self.model.language_model.forward = Mock(return_value=torch.zeros(1, 5, 4, dtype=torch.bfloat16))
                # No lm_head on stage 0 (only the last stage has it).

        m = _Stage0()
        m.forward = create_pipeline_forward_mistral3_vlm().__get__(m, type(m))
        return m

    def test_chunk_retrieval_fires_when_pixel_values_none(self):
        """When pixel_values=None and `_vlm_pixel_values_chunks` is set, the
        forward retrieves the current microbatch's chunk and advances the
        chunk index — fixing the original PP-VLM bug."""
        m = self._make_first_stage_model(image_token_id=10)
        # Pre-stage chunks like stage_vlm_media_for_pp does for schedule.step.
        chunk0 = torch.zeros(1, 3, 8, 8, dtype=torch.bfloat16)
        chunk1 = torch.ones(1, 3, 8, 8, dtype=torch.bfloat16)
        m._vlm_pixel_values_chunks = [chunk0, chunk1]
        m._vlm_image_grid_hws_chunks = [
            torch.tensor([[8, 8]]),
            torch.tensor([[8, 8]]),
        ]
        m._vlm_chunk_idx = 0

        # input_ids contains image tokens — required by the chunk-retrieval gate.
        input_ids = torch.tensor([[10, 10, 1, 2, 3]])
        # image_features stub: HF returns a tuple-like; cat picks shape [N, D].
        m.model.get_image_features.return_value = SimpleNamespace(
            pooler_output=(torch.zeros(2, 4, dtype=torch.bfloat16),)
        )
        # Mask everything to True so masked_scatter doesn't shape-mismatch
        # (we only care that vision_tower fired, not pixel-perfect routing).
        m.model.get_placeholder_mask.return_value = torch.zeros(1, 5, 4, dtype=torch.bool)

        m(input_ids=input_ids, pixel_values=None)

        # Vision path was entered with the staged chunk — mock receives chunk0.
        m.model.get_image_features.assert_called_once()
        call_kwargs = m.model.get_image_features.call_args.kwargs
        assert torch.equal(call_kwargs["pixel_values"], chunk0)
        # Index advanced for the next microbatch.
        assert m._vlm_chunk_idx == 1

    def test_vision_feature_layer_resolved_from_config(self):
        """When vision_feature_layer is None, the forward must pull it from
        config — HF's outer forward does this via @merge_with_config_defaults,
        which our patched forward bypasses."""
        from types import SimpleNamespace as _SN

        m = self._make_first_stage_model(vision_feature_layer_default=-1)
        m.model.get_image_features.return_value = _SN(pooler_output=(torch.zeros(1, 4, dtype=torch.bfloat16),))
        m.model.get_placeholder_mask.return_value = torch.zeros(1, 5, 4, dtype=torch.bool)
        # Avoid chunk-retrieval branch by passing pixel_values directly.
        m(
            input_ids=torch.tensor([[10, 1, 2, 3, 4]]),
            pixel_values=torch.zeros(1, 3, 8, 8, dtype=torch.bfloat16),
            image_sizes=torch.tensor([[8, 8]]),
            # vision_feature_layer left None
        )
        call_kwargs = m.model.get_image_features.call_args.kwargs
        assert call_kwargs["vision_feature_layer"] == -1

    def test_chunk_retrieval_skipped_when_no_image_tokens(self):
        """If input_ids has no image_token_id, the chunk-retrieval gate
        does not fire (defensive — text-only batches may still pass through)."""
        m = self._make_first_stage_model(image_token_id=10)
        m._vlm_pixel_values_chunks = [torch.zeros(1, 3, 8, 8)]
        m._vlm_image_grid_hws_chunks = None
        m._vlm_chunk_idx = 0
        # input_ids contains NO image tokens (token 10 absent).
        m.model.language_model.forward = Mock(return_value=torch.zeros(1, 5, 4, dtype=torch.bfloat16))
        m(input_ids=torch.tensor([[1, 2, 3, 4, 5]]), pixel_values=None)
        m.model.get_image_features.assert_not_called()
        assert m._vlm_chunk_idx == 0  # unchanged

    def test_non_first_stage_passes_hidden_states(self):
        """Non-first stage: input_ids carries float hidden states — promote to
        inputs_embeds and forward through language_model directly (no embed,
        no vision_tower)."""
        from nemo_automodel.components.distributed.pipelining.hf_utils import (
            create_pipeline_forward_mistral3_vlm,
        )

        class _StageN(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = Mock(model_type="mistral3", image_token_id=10)
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                # Non-first stage: embed_tokens is None (pruned in the PP split).
                self.model.language_model.embed_tokens = None
                self.model.language_model.forward = Mock(return_value=torch.zeros(1, 5, 4, dtype=torch.bfloat16))
                self.model.vision_tower = None  # also pruned
                # No lm_head — middle stage.

        m = _StageN()
        m.forward = create_pipeline_forward_mistral3_vlm().__get__(m, type(m))
        # input_ids dtype is float ⇒ promoted to inputs_embeds.
        hidden = torch.zeros(1, 5, 4, dtype=torch.bfloat16)
        m(input_ids=hidden)

        # Vision path NOT called (vision_tower is None).
        assert m.model.language_model.forward.called
        kwargs = m.model.language_model.forward.call_args.kwargs
        assert kwargs["input_ids"] is None
        assert torch.equal(kwargs["inputs_embeds"], hidden)
