# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.distributed.autopipeline.hf_utils import create_pipeline_forward_inner, create_pipeline_forward_causal_lm

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
                return (hidden_states,)

        mock_model.layers = nn.ModuleDict({"0": DummyLayer()})

        # Mock norm
        mock_norm = Mock()
        mock_norm.return_value = torch.randn(1, 10, 768)
        mock_model.norm = mock_norm

        # Mock rotary_emb
        mock_rotary = Mock()
        mock_rotary.return_value = torch.randn(1, 10, 768)
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

    def test_forward_with_cache(self):
        mock_model = Mock()
        mock_model.config = Mock(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True
        )
        mock_model.gradient_checkpointing = False
        mock_model.embed_tokens = None
        mock_model.layers = None
        mock_model.norm = None
        mock_model.rotary_emb = None

        forward_fn = create_pipeline_forward_inner("AutoModel")

        # Test with cache
        mock_cache = Mock(spec=Cache)
        mock_cache.get_seq_length.return_value = 5

        inputs_embeds = torch.randn(1, 10, 768)
        output = forward_fn(mock_model, inputs_embeds=inputs_embeds, past_key_values=mock_cache)

        assert isinstance(output, BaseModelOutputWithPast)
        assert output.past_key_values is not None


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
