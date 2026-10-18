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

"""D1-3B causal image/text backbone with its released tied vocabulary readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Lfm2VlConfig, PreTrainedModel
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlCausalLMOutputWithPast, Lfm2VlModel

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.d1.causal import (
    D1CausalAttention,
    D1CausalRotaryEmbedding,
    D1CausalShortConv,
)


class Lfm2VlForConditionalGeneration(HFCheckpointingMixin, PreTrainedModel):
    """D1-3B's chain execution, using HF vision and native precision-correct LFM2.

    Questions are independent causal chains. The release's shared-trunk tree
    batching is an inference optimization and is not used by this training path.
    Parameter names and the shared input/output embedding identity are retained.
    """

    config_class = Lfm2VlConfig
    base_model_prefix = "model"
    tie_word_embeddings_support: TieSupport = TieSupport.TIED_ONLY
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _supports_sdpa = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallelism is undeclared until corresponding recipes are verified."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: Lfm2VlConfig) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.model = Lfm2VlModel(config)
        language = self.model.language_model
        for layer in language.layers:
            if layer.is_attention_layer:
                layer.self_attn = D1CausalAttention(layer.self_attn)
            else:
                layer.conv = D1CausalShortConv(layer.conv)
        with torch.device("cpu"):
            language.rotary_emb = D1CausalRotaryEmbedding(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self, **kwargs: Any) -> None:
        """Share the released vocabulary weight with the text input embeddings."""
        reject_unsupported_tie_word_embeddings(type(self), self.config)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the released text embedding table."""
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Return the tied vocabulary projection."""
        return self.lm_head

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace and retie the text embedding table."""
        self.model.language_model.embed_tokens = value
        self.tie_weights()

    @property
    def device(self) -> torch.device:
        """Device holding the text embedding table."""
        return self.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the text embedding table."""
        return self.get_input_embeddings().weight.dtype

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ) -> Lfm2VlCausalLMOutputWithPast:
        """Run a causal chain with optional processed images.

        Args:
            input_ids: Optional long tensor [batch, length].
            attention_mask: Optional tensor [batch, length] marking real tokens.
            position_ids: Optional long tensor [batch, length].
            pixel_values: Optional HF-processed image patches.
            spatial_shapes: Optional image-grid dimensions [images, 2].
            pixel_attention_mask: Optional real-patch mask.
            inputs_embeds: Optional tensor [batch, length, hidden_size], replacing IDs.
            labels: Optional long next-token targets [batch, length], -100 for ignored targets.
            logits_to_keep: Number of trailing logits to return; zero returns all positions.
            **kwargs: Additional HF backbone arguments, including use_cache.

        Returns:
            HF-compatible causal output with tied-vocabulary logits and optional loss.
        """
        if logits_to_keep < 0 or (labels is not None and logits_to_keep):
            raise ValueError("logits_to_keep must be nonnegative and zero when computing a causal loss")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden = outputs.last_hidden_state[:, -logits_to_keep:] if logits_to_keep else outputs.last_hidden_state
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return Lfm2VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
