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

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.distributed.pipelining.hf_utils import (
    create_pipeline_forward_causal_lm,
    create_pipeline_forward_inner,
    model_keeps_self_forward,
    patch_hf_model_for_pp,
    validate_hf_model_for_pipeline_support,
)
from nemo_automodel.shared.pipeline import PipelineForwardStyle, PipelineModelMixin


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
        assert isinstance(mock_model._pp_causal_mask_cache, dict)

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
        assert isinstance(mock_model._pp_causal_mask_cache, dict)

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
        forward_fn(mock_model, inputs_embeds=hidden_states, logits_to_keep=5)

        # Verify lm_head was called with sliced hidden states
        called_hidden = mock_lm_head.call_args[0][0]
        assert called_hidden.shape[1] == 5  # Only last 5 positions

    def test_forward_returns_hidden_states_when_flagged(self):
        """Fused-CE path: with _pp_return_hidden_states=True the last stage skips
        lm_head and returns the hidden states unprojected."""
        mock_model = Mock()
        mock_model.config = Mock(output_attentions=False, output_hidden_states=False)
        mock_model.model = None
        mock_lm_head = Mock()
        mock_model.lm_head = mock_lm_head
        # Flag set by train_ft._configure_pipeline_loss_fn for FusedLinearCrossEntropy.
        mock_model._pp_return_hidden_states = True

        forward_fn = create_pipeline_forward_causal_lm()

        hidden_states = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=hidden_states, logits_to_keep=1)

        # lm_head must NOT be applied; hidden states are returned as-is.
        mock_lm_head.assert_not_called()
        assert torch.equal(output, hidden_states)

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

    def test_patched_causal_lm_declares_the_fused_loss_capability(self):
        """The generic CausalLM forward honors ``_pp_return_hidden_states``, so the
        patched model must declare ``pipeline_supports_hidden_state_output``.

        The capability used to be recorded under the private name
        ``_pp_return_hidden_states_supported``, which only existed as a side
        effect of monkey-patching; models keeping their own forward therefore
        silently lost FusedLinearCrossEntropy.
        """

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()

        model = OuterModel()

        patch_hf_model_for_pp(model, patch_inner_model=True, patch_causal_lm_model=True)

        assert getattr(model, "pipeline_supports_hidden_state_output", False) is True
        assert not hasattr(model, "_pp_return_hidden_states_supported")

    def test_pipeline_mixin_defaults_to_no_fused_loss_capability(self):
        """A model owning its forward opts in explicitly; the default is off."""

        class _Default(PipelineModelMixin, nn.Module):
            pipeline_forward_style = PipelineForwardStyle.MODEL

        class _OptedIn(PipelineModelMixin, nn.Module):
            pipeline_forward_style = PipelineForwardStyle.MODEL
            pipeline_supports_hidden_state_output = True

        assert _Default().pipeline_supports_hidden_state_output is False
        assert _OptedIn().pipeline_supports_hidden_state_output is True

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

    def test_patch_vlm_with_nested_language_model_uses_the_generic_forward(self):
        """A VLM exposing ``model.language_model`` is patched on ``model.model``."""

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                # Many HF VLMs (KimiVL / Mistral4 / Qwen3VL MoE / LlavaOneVision / ...)
                # expose language_model here without owning a pipeline forward.
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

        # The generic path patches model.model (inner) directly, not language_model.
        assert model.model.forward is not original_inner
        assert model.forward is not original_outer
        assert model.model.forward.__func__.__name__ == "pipeline_forward"
        assert model.forward.__func__.__name__ == "pipeline_forward_causal_lm"

    def test_model_keeps_self_forward_helper(self):
        """``model_keeps_self_forward`` reflects the typed model contract.

        Regression for the silent-vision bug where chunk-aware VLMs (Qwen3-VL-MoE,
        KimiVL, Kimi-K2.5-VL, Qwen3.5-MoE) had their pixel_values-fetching forward
        replaced by the generic CausalLM forward, causing vision_tower to never
        run. The fix splits responsibility: the model class declares the contract,
        and the pipeline build call site uses ``model_keeps_self_forward`` to
        decide whether to invoke ``patch_hf_model_for_pp`` at all.
        """

        class _OptedIn(PipelineModelMixin, nn.Module):
            pipeline_forward_style = PipelineForwardStyle.MODEL

        class _Default(nn.Module):
            pass

        assert model_keeps_self_forward(_OptedIn()) is True
        assert model_keeps_self_forward(_Default()) is False


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

        with pytest.raises(ValueError, match="Pipeline parallelism does not support tie_word_embeddings=True"):
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
        """VLMs without a dedicated or model-owned PP forward must fail validation."""

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

        with pytest.raises(ValueError, match="does not own its pipeline forward"):
            validate_hf_model_for_pipeline_support(model)

    def test_validate_chunk_aware_vlm_passes(self):
        """VLMs with a model-owned pipeline forward must pass validation."""

        class _TextCfg:
            tie_word_embeddings = False

        class MockConfig:
            pretrained_model_name_or_path = "test/qwen3_vl_moe"
            tie_word_embeddings = False
            is_encoder_decoder = False
            model_type = "qwen3_vl_moe"
            text_config = _TextCfg()

        class MockVLM(PipelineModelMixin, nn.Module):
            pipeline_forward_style = PipelineForwardStyle.MODEL

            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.vision_tower = nn.Linear(4, 4)

        model = MockVLM()
        # Should not raise.
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
        forward_fn(mock_model, inputs_embeds=inputs_embeds, position_ids=position_ids)

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
        assert "vision_model" in MULTIMODAL_SUFFIXES
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
        assert "vit_large_projector" in MULTIMODAL_SUFFIXES
