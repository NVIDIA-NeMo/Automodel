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

"""Native ``NemotronForDecision``: a bidirectional Ministral 3 encoder with the decision readout head.

Nemotron 3.5 Decision checkpoints ship ``architectures: ["NemotronForDecision"]`` with remote code that needs a newer
Transformers than this repository pins. This class loads the same checkpoint natively: the encoder is
``Ministral3BidirectionalModel`` (non-causal attention over the whole sequence) and the head keeps the checkpoint's
weight name, ``diffusion_head`` (named after the checkpoint it was initialized from), so the safetensors load and
export without renaming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from nemo_automodel.components.models.ministral_bidirectional.model import Ministral3BidirectionalModel

ARCHITECTURE = "NemotronForDecision"


class NemotronForDecision(PreTrainedModel):
    """Bidirectional Ministral 3 encoder plus a vocabulary head read at the decision mask token."""

    config_class = Ministral3Config
    base_model_prefix = "encoder"
    _no_split_modules = ["Ministral3DecoderLayer"]
    _supports_sdpa = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: Ministral3Config, **kwargs: Any) -> None:
        """Build the encoder and head.

        Args:
            config: Ministral 3 text config of the checkpoint; ``is_causal`` is forced to ``False``.
            **kwargs: Unused loader options.
        """
        config.is_causal = False
        super().__init__(config)
        self.encoder = Ministral3BidirectionalModel(config)
        self.diffusion_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def supports_config(cls, config: Any) -> bool:
        """True for text Ministral 3 decision checkpoints (no vision tower, no quantization)."""
        return (
            getattr(config, "model_type", None) == "ministral3"
            and not getattr(config, "vision_config", None)
            and not getattr(config, "quantization_config", None)
        )

    def get_input_embeddings(self) -> nn.Embedding:
        """Token embeddings of the encoder."""
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the encoder's token embeddings."""
        self.encoder.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """The decision readout head."""
        return self.diffusion_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Replace the decision readout head."""
        self.diffusion_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_last_hidden_states_only: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutput | MaskedLMOutput:
        """Read the sequence bidirectionally.

        Args:
            input_ids: Token ids ``[batch, sequence]``.
            attention_mask: Optional padding mask ``[batch, sequence]``.
            position_ids: Optional positions ``[batch, sequence]``; defaults to ``0..sequence-1``.
            inputs_embeds: Optional embeddings instead of ``input_ids``.
            output_last_hidden_states_only: Return the encoder's last hidden state without projecting the
                ``[batch, sequence, vocab]`` logits.
            **kwargs: Passed to the encoder (``use_cache`` is ignored; decision reads keep no cache).

        Returns:
            ``BaseModelOutput(last_hidden_state)`` or ``MaskedLMOutput(logits)``.
        """
        kwargs.pop("use_cache", None)
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        ).last_hidden_state
        if output_last_hidden_states_only:
            return BaseModelOutput(last_hidden_state=hidden)
        return MaskedLMOutput(logits=self.diffusion_head(hidden))
