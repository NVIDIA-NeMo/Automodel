# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Portable Transformers implementation of pooled Mistral3 sequence scoring."""

import os
from typing import Any

import torch
from torch import nn
from transformers import Mistral3Config, Mistral3Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.mistral3.modeling_mistral3 import Mistral3PreTrainedModel


class Mistral3ForSequenceClassification(Mistral3PreTrainedModel):
    """Mistral3 backbone with masked pooling and an FP32 scoring projection."""

    config_class = Mistral3Config
    base_model_prefix = "model"

    def __init__(self, config: Mistral3Config) -> None:
        super().__init__(config)
        if config.pooling != "avg":
            raise ValueError("Portable Mistral3 reranker export requires mean pooling (pooling=avg).")
        self.model = Mistral3Model(config)
        for layer in self.model.language_model.layers:
            layer.self_attn.is_causal = config.text_config.is_causal
        self.score = nn.Linear(config.text_config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike | None, *model_args: Any, **kwargs: Any
    ) -> "Mistral3ForSequenceClassification":
        """Load the shared classifier head without changing global Transformers conversion mappings."""
        key_mapping = dict(kwargs.pop("key_mapping", None) or {})
        key_mapping.setdefault(r"^language_model\.score\.weight$", "score.weight")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, key_mapping=key_mapping, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutputWithPast:
        """Score tokenized text or multimodal pairs.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Padding mask of shape [batch, sequence]; required for pooling.
            pixel_values: Optional image tensor of shape [images, channels, height, width].
            image_sizes: Optional integer tensor of shape [images, 2], height then width.
            position_ids: Optional integer tensor of shape [batch, sequence].
            **kwargs: Additional Mistral3Model inputs, following its forward tensor contract.

        Returns:
            Classifier output with raw, unscaled FP32 logits of shape [batch, num_labels].
        """
        if attention_mask is None:
            raise ValueError("attention_mask is required for sequence pooling.")
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        pooled = hidden.masked_fill(~attention_mask[..., None].bool(), 0).sum(1) / attention_mask.sum(1)[:, None]
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            logits = nn.functional.linear(pooled.float(), self.score.weight.float())
        return SequenceClassifierOutputWithPast(logits=logits)
