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

"""
Llama Bidirectional Model for NeMo AutoModel.

This module provides a bidirectional attention variant of Llama that is useful
for embedding and retrieval tasks. Unlike the standard causal Llama model,
this version can attend to all tokens bidirectionally.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def pool(last_hidden_states: torch.Tensor,
         attention_mask: torch.Tensor,
         pool_type: str) -> torch.Tensor:
    """
    Pool hidden states using the specified pooling method.
    
    Args:
        last_hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
        pool_type: Type of pooling to apply
        
    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "weighted_avg":
        emb = last_hidden.sum(dim=1)
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    elif pool_type == "cls_last":
        emb = last_hidden[:, 0]
    elif pool_type == "colbert":
        emb = last_hidden
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


class LlamaBidirectionalConfig(LlamaConfig):
    """
    Configuration class for LlamaBidirectionalModel.
    
    Extends LlamaConfig with additional parameters for bidirectional attention
    and pooling configurations.
    """
    model_type = "llama_bidirec"

    def __init__(
        self, 
        pooling: str = "avg", 
        temperature: float = 1.0, 
        **kwargs,
    ):
        """
        Initialize LlamaBidirectionalConfig.
        
        Args:
            pooling: Pooling strategy ('avg', 'cls', 'last', etc.)
            temperature: Temperature for scaling logits
            **kwargs: Additional arguments passed to LlamaConfig
        """
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(**kwargs)


class LlamaBidirectionalModel(LlamaModel):
    """
    Llama Model with bidirectional attention.
    
    This model removes causal masking from all attention layers, allowing tokens
    to attend to all other tokens in the sequence. This is useful for embedding
    and retrieval tasks where bidirectional context is beneficial.
    """
    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig):
        """
        Initialize LlamaBidirectionalModel.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        # Disable causal attention for all layers
        for layer in self.layers:
            layer.self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        """
        Override the causal mask to support bidirectional attention.
        
        Args:
            attention_mask: Attention mask
            input_tensor: Input tensor
            cache_position: Cache position for generation
            past_key_values: Past key values for generation
            output_attentions: Whether to output attentions
            
        Returns:
            Bidirectional attention mask
        """
        assert self.config._attn_implementation in ["flash_attention_2", "eager"], (
            f"Unsupported attention implementation: {self.config._attn_implementation}, "
            "only support flash_attention_2 or eager"
        )
 
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        elif self.config._attn_implementation == "eager":
            # Generates bi-directional attention mask
            causal_mask = _prepare_4d_attention_mask(
                attention_mask,
                dtype=input_tensor.dtype,
            )
            return causal_mask


class LlamaBidirectionalForSequenceClassification(LlamaForSequenceClassification):
    """
    Llama Bidirectional Model with a sequence classification/regression head.
    
    This model adds a classification head on top of the bidirectional Llama model
    and includes configurable pooling strategies.
    """
    config_class = LlamaBidirectionalConfig

    def __init__(self, config):
        """
        Initialize LlamaBidirectionalForSequenceClassification.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        # Release the parameters of LlamaModel
        # created by parent LlamaForSequenceClassification
        del self.model

        self.model = LlamaBidirectionalModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for generation
            inputs_embeds: Input embeddings (alternative to input_ids)
            labels: Labels for computing loss
            use_cache: Whether to use cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict
            
        Returns:
            SequenceClassifierOutputWithPast with loss, logits, and optional outputs
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Pool hidden states using configured pooling strategy
        pooled_hidden_states = pool(
            last_hidden_states=hidden_states,
            attention_mask=attention_mask,
            pool_type=self.config.pooling,
        )

        # Apply classification head and temperature scaling
        pooled_logits = self.score(pooled_hidden_states)
        pooled_logits = pooled_logits / self.config.temperature

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
                
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

