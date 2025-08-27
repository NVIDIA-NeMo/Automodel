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
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.distributed.pipelining.hf_utils import (
    create_pipeline_forward_inner,
    create_pipeline_forward_causal_lm,
    patch_hf_model_for_pp,
    init_hf_model_buffers,
    validate_hf_model_for_pipeline_support,
)

class TestCreatePipelineForwardInner:
    """Test create_pipeline_forward_inner function."""

    def test_returns_callable(self):
        forward_fn = create_pipeline_forward_inner("AutoModel")
        assert callable(forward_fn)

    @patch('torch.arange')
    def test_forward_with_embeddings(self, mock_arange):
        # Create mock model with embeddings
        mock_model = Mock()
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False
        )

        # Mock inner model
        mock_inner = Mock()
        mock_inner.return_value = BaseModelOutputWithPast(
            last_hidden_state=torch.randn(1, 10, 768)
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False
        )

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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False
        )
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
        assert hasattr(model.model.rotary_emb, 'inv_freq')

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
        assert hasattr(model.rotary_emb, 'inv_freq')


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
        """Test validation fails for model with tied embeddings."""
        class MockConfig:
            pretrained_model_name_or_path = "test/model"
            tie_word_embeddings = True  # This should cause validation to fail
            is_encoder_decoder = False

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()

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
            tie_word_embeddings = True  # Issue 1
            is_encoder_decoder = True   # Issue 2

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()

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

        with patch('nemo_automodel.components.distributed.pipelining.hf_utils.logger') as mock_logger:
            output = forward_fn(mock_model, inputs_embeds=inputs_embeds)
            # No warning should be called in the new style
            assert not mock_logger.warning_once.called

        assert isinstance(output, BaseModelOutputWithPast)
        assert output.past_key_values is None

    def test_missing_input_error(self):
        """Test error when neither input_ids nor inputs_embeds provided with embed_tokens."""
        mock_model = Mock()
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False
        )
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
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False
        )
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
        with patch('transformers.masking_utils.create_causal_mask') as mock_create_causal, \
             patch('transformers.masking_utils.create_sliding_window_causal_mask') as mock_create_sliding:

            mock_create_causal.return_value = torch.ones(1, 1, 10, 10)
            mock_create_sliding.return_value = torch.ones(1, 1, 10, 10) * 2

            forward_fn = create_pipeline_forward_inner("AutoModel")

            inputs_embeds = torch.randn(1, 10, 768)
            attention_mask = torch.ones(1, 10)

            # Mock has_sliding_layers to trigger sliding window creation
            mock_model.has_sliding_layers = True

            output = forward_fn(mock_model, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

            assert isinstance(output, BaseModelOutputWithPast)

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
