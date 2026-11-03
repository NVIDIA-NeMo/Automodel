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
from transformers import Lfm2VlConfig
from transformers.models.lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlForConditionalGeneration as HFLfm2VlForConditionalGeneration,
)

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
from nemo_automodel.components.models.d1.records import D1TextExample
from nemo_automodel.components.models.d1.text_readout import D1CausalTextReadout


class Lfm2VlForConditionalGeneration(HFCheckpointingMixin, HFLfm2VlForConditionalGeneration):
    """D1-3B's chain execution, using HF vision and native precision-correct LFM2.

    Questions are independent causal chains. The release's shared-trunk tree
    batching is an inference optimization and is not used by this training path.
    Parameter names and the checkpoint's input/output tying policy are retained.
    Ordinary LFM2-VL checkpoints retain the inherited HuggingFace generation,
    cache, forward-output and image-feature interfaces.
    """

    config_class = Lfm2VlConfig
    base_model_prefix = "model"
    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _supports_sdpa = True

    @classmethod
    def supports_config(cls, config: Lfm2VlConfig) -> bool:
        """Select D1 arithmetic only for the publisher's explicit D1 model mapping."""
        auto_map = getattr(config, "auto_map", None)
        model_reference = auto_map.get("AutoModel", "") if auto_map is not None else ""
        return model_reference.rsplit("--", 1)[-1] == "modeling_d1.D1Model"

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
        language = self.model.language_model
        for layer in language.layers:
            if layer.is_attention_layer:
                layer.self_attn = D1CausalAttention(layer.self_attn)
            else:
                layer.conv = D1CausalShortConv(layer.conv)
        with torch.device("cpu"):
            language.rotary_emb = D1CausalRotaryEmbedding(config.text_config)
        self.tie_weights()

    def tie_weights(self, **kwargs: Any) -> None:
        """Honor the checkpoint's tied or independently initialized vocabulary head."""
        reject_unsupported_tie_word_embeddings(type(self), self.config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the released text embedding table."""
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        """Return the vocabulary projection, honoring the checkpoint's tying policy."""
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

    def prepare_text_readout(self, examples: list[D1TextExample]) -> D1CausalTextReadout:
        """Prepare the released answer-alias adapter for text-only adaptation.

        Args:
            examples: Publisher-encoded questions used to select answer vocabulary rows.

        Returns:
            Final-token feature extractor and zero-initialized answer-row adapter.
        """
        return D1CausalTextReadout(self, examples)
