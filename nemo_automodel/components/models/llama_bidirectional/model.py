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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification, LlamaModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

try:
    from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator

    check_model_inputs = get_check_model_inputs_decorator()
except ImportError:
    # Fallback to no-op decorator if import fails
    def check_model_inputs(func):
        return func


class LlamaBidirectionalConfig(LlamaConfig):
    """
    Configuration class for LlamaBidirectionalModel.

    Extends LlamaConfig with additional parameters for bidirectional attention
    and pooling configurations.
    """

    model_type = "LlamaBidirectionalModel"

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
    enabling it to be loaded via NeMoAutoModelForBiencoder.from_pretrained().
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config: LlamaConfig):
        """
        Initialize LlamaBidirectionalModel.

        Args:
            config: Model configuration
        """
        super().__init__(config)

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
    ):
        """Override causal mask to allow bidirectional attention."""
        if attention_mask is not None and (attention_mask == 0.0).any():
            return attention_mask
        return None

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # enforce bidirectional attention by disabling causal masking
        causal_mask = self._update_causal_mask(attention_mask=attention_mask)
        kwargs["is_causal"] = False

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


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


class LlamaBidirectionalForSequenceClassification(LlamaForSequenceClassification):
    """
    Llama Bidirectional Model with a sequence classification/regression head.

    This model adds a classification head on top of the bidirectional Llama model
    and includes configurable pooling strategies.
    """

    config_class = LlamaBidirectionalConfig

    def __init__(self, config):
        super().__init__(config)
        del self.model
        self.model = LlamaBidirectionalModel(config)
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
        )
        hidden_states = transformer_outputs[0]

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

__all__ = [
    "LlamaBidirectionalModel",
    "LlamaBidirectionalConfig",
    "LlamaBidirectionalForSequenceClassification",
    "ModelClass",
]
