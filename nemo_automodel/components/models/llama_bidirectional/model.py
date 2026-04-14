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
Llama Bidirectional model for embedding and retrieval tasks.

This module provides a bidirectional variant of Llama that is auto-discovered
by the ModelRegistry via the ModelClass export.

To add support for other backbones (e.g., Qwen2, Mistral), create a similar
module in a new directory (e.g., qwen2_bidirectional/) with its own ModelClass export.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel


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

    The model is auto-discovered by ModelRegistry via the ModelClass export,
    enabling it to be loaded via NeMoAutoModelBiEncoder.from_pretrained().
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig):
        """
        Initialize LlamaBidirectionalModel.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        # Enable bidirectional attention: config flag for mask generation,
        # module flag for SDPA/FA2 kernel fallback.
        config.is_causal = False
        for layer in self.layers:
            layer.self_attn.is_causal = False


def _pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str) -> torch.Tensor:
    """Pool hidden states using the specified pooling method."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    elif pool_type == "last":
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            emb = last_hidden[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    else:
        emb = last_hidden[:, 0]  # default to CLS

    return emb


class LlamaBidirectionalForSequenceClassification(LlamaPreTrainedModel):
    """
    Llama Bidirectional Model with a sequence classification/regression head.

    This model adds a classification head on top of the bidirectional Llama model
    and includes configurable pooling strategies.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
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
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            **kwargs,
        )
        hidden_states = transformer_outputs[0]

        if attention_mask is None:
            raise ValueError("attention_mask is required for pooling")

        pooled_hidden_states = _pool(
            last_hidden_states=hidden_states,
            attention_mask=attention_mask,
            pool_type=getattr(self.config, "pooling", "avg"),
        )

        pooled_logits = self.score(pooled_hidden_states)
        pooled_logits = pooled_logits / getattr(self.config, "temperature", 1.0)

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
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
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
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


# Export for ModelRegistry auto-discovery
ModelClass = [LlamaBidirectionalModel, LlamaBidirectionalForSequenceClassification]


def _register_with_hf_auto_classes():
    """Register bidirectional models with HuggingFace Auto classes.

    This is needed so that AutoModel.from_config(LlamaBidirectionalConfig)
    works inside LlamaForSequenceClassification.__init__.
    """
    from transformers import AutoConfig, AutoModel

    try:
        AutoConfig.register(LlamaBidirectionalConfig.model_type, LlamaBidirectionalConfig)
    except ValueError:
        pass  # Already registered
    try:
        AutoModel.register(LlamaBidirectionalConfig, LlamaBidirectionalModel)
    except ValueError:
        pass  # Already registered


_register_with_hf_auto_classes()

__all__ = [
    "LlamaBidirectionalModel",
    "LlamaBidirectionalConfig",
    "LlamaBidirectionalForSequenceClassification",
    "ModelClass",
]
