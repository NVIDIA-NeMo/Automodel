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

"""Checkpoint-compatible Omni model with a differentiable decision API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import SequenceClassifierOutput

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.d1.config import D1OmniConfig
from nemo_automodel.components.models.d1.decision_head import D1OmniHeadConfig
from nemo_automodel.components.models.d1.encoder import D1OmniEncoder
from nemo_automodel.components.models.d1.records import D1TextExample
from nemo_automodel.components.models.d1.releases import RELEASES
from nemo_automodel.components.models.d1.text_readout import D1OmniTextReadout

OMNI_REPOSITORY, OMNI_REVISION = RELEASES["d1-omni"][:2]


class D1OmniModel(HFCheckpointingMixin, PreTrainedModel):
    """Omni's bidirectional trunk and released marker head, preserving all weights.

    Perception classes come from reviewed, immutable released code. The language
    encoder and decision API are model-owned. Training returns tensors with their
    autograd graph intact, unlike the released inference probability interface.
    """

    config_class = D1OmniConfig
    base_model_prefix = "d1"
    main_input_name = "input_ids"
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _supports_sdpa = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallelism is undeclared until the corresponding recipes are verified."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config: D1OmniConfig, *, trust_remote_code: bool = False) -> None:
        """Build Omni, requiring explicit consent for its pinned publisher perception code."""
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        if not config.vision_config or not config.audio_config:
            raise ValueError("Omni requires both released vision and audio configurations")
        width = config.text_config.hidden_size
        if not trust_remote_code:
            raise ValueError("D1 Omni perception modules require trust_remote_code=True for pinned publisher code")
        self.encoder = D1OmniEncoder(config.text_config)
        self.head = D1OmniHeadConfig(width, config.head_layers).build()
        vision_class = get_class_from_dynamic_module(
            "vision.Vision", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
        )
        audio_class = get_class_from_dynamic_module(
            "audio.Audio", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
        )
        self.vision = vision_class(config.vision_config, config.projector_hidden_size, width)
        self.audio = audio_class(config.audio_config, width)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text input embedding table."""
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the text embedding table."""
        self.encoder.embed_tokens = value

    def prepare_text_readout(self, examples: list[D1TextExample]) -> D1OmniTextReadout:
        """Prepare the released marker readout for text-only adaptation.

        Args:
            examples: Publisher-encoded questions; marker identity and calibration
                remain attached to each example and are consumed when scoring.

        Returns:
            Sequence feature extractor and the released float32 marker head.
        """
        return D1OmniTextReadout(self)

    def encode_images(self, images: list[Any]) -> torch.Tensor:
        """Encode ordered PIL images into a perception prefix.

        Returns:
            Tensor of shape [1, prefix_sequence, hidden], concatenating all images.
        """
        return self.vision(images)

    def encode_audio(self, audio: Any) -> torch.Tensor:
        """Encode a mono 16 kHz clip into a perception prefix.

        Args:
            audio: Float Tensor or array of shape [samples], containing a mono
                waveform sampled at 16 kHz, as accepted by the pinned perception module.

        Returns:
            Tensor of shape [1, prefix_sequence, hidden].
        """
        return self.audio(audio)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        media_prefix: torch.Tensor | None = None,
        media_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text with optional precomputed perception prefixes.

        Args:
            input_ids: Long Tensor of shape [batch, sequence], with right padding.
            attention_mask: Boolean Tensor of shape [batch, sequence], true for text tokens.
            media_prefix: Optional Tensor of shape [batch, padded_prefix_sequence, hidden].
            media_lengths: Optional long Tensor of shape [batch] specifying real prefix lengths.

        Returns:
            Pair of normalized text-state Tensor of shape
            [batch, longest_real_text, hidden] and boolean Tensor of shape
            [batch, longest_real_text] marking real text tokens. Prefix states
            are excluded; outputs do not alias the supplied prefix.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long with shape [batch, text_length]")
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean with the input_ids shape")
        lengths = attention_mask.sum(dim=1)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        if not torch.equal(attention_mask, positions.unsqueeze(0) < lengths.unsqueeze(1)):
            raise ValueError("only right-padded text is supported")
        if torch.any(lengths == 0):
            raise ValueError("each row requires at least one text token")
        embeddings = self.encoder.embed_tokens(input_ids)
        batch = input_ids.shape[0]
        if media_prefix is None:
            if media_lengths is not None and torch.any(media_lengths != 0):
                raise ValueError("nonzero media_lengths require media_prefix")
            media_lengths = torch.zeros(batch, device=input_ids.device, dtype=torch.long)
            sequences = [embeddings[i, :length] for i, length in enumerate(lengths.tolist())]
        else:
            if media_lengths is None or media_lengths.shape != (batch,) or media_lengths.dtype != torch.long:
                raise ValueError("media_prefix requires long media_lengths with shape [batch]")
            if (
                media_prefix.ndim != 3
                or media_prefix.shape[0] != batch
                or media_prefix.shape[-1] != embeddings.shape[-1]
            ):
                raise ValueError("media_prefix must have shape [batch, prefix_length, width]")
            if torch.any(media_lengths < 0) or torch.any(media_lengths > media_prefix.shape[1]):
                raise ValueError("media_lengths must index the supplied prefixes")
            sequences = [
                torch.cat([media_prefix[i, :prefix].to(embeddings), embeddings[i, :length]])
                for i, (prefix, length) in enumerate(zip(media_lengths.tolist(), lengths.tolist()))
            ]
        total_lengths = lengths + media_lengths
        if torch.any(total_lengths > self.config.max_length):
            raise ValueError("combined prefix and text exceed max_length")
        packed = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        valid = torch.arange(packed.shape[1], device=input_ids.device).unsqueeze(0) < total_lengths.unsqueeze(1)
        hidden = self.encoder(packed, valid, media_lengths)
        text = nn.utils.rnn.pad_sequence(
            [
                hidden[i, prefix : prefix + length]
                for i, (prefix, length) in enumerate(zip(media_lengths.tolist(), lengths.tolist()))
            ],
            batch_first=True,
        )
        valid_text = torch.arange(text.shape[1], device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        return text, valid_text

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_positions: torch.Tensor,
        marker_mask: torch.Tensor,
        question_types: torch.Tensor,
        media_prefix: torch.Tensor | None = None,
        media_lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        """Return differentiable option logits in the released internal ordering.

        Args:
            input_ids: Long Tensor of shape [batch, sequence], with right padding.
            attention_mask: Boolean Tensor of shape [batch, sequence], true for text tokens.
            marker_positions: Long Tensor of shape [batch, options], relative to text.
            marker_mask: Boolean Tensor of shape [batch, options] marking declared options.
            question_types: Long Tensor of shape [batch], choice=0, score=1, noul=2.
            media_prefix: Optional Tensor of shape [batch, padded_prefix_sequence, hidden].
            media_lengths: Optional long Tensor of shape [batch] giving real prefix lengths.
            labels: Optional long Tensor of shape [batch] of internal option indices.

        Returns:
            SequenceClassifierOutput containing float32 logits Tensor of shape
            [batch, options], optional scalar cross-entropy Tensor, and a
            one-element hidden_states tuple containing normalized text Tensor
            of shape [batch, longest_real_text, hidden].
        """
        text, valid = self.encode_text(input_ids, attention_mask, media_prefix, media_lengths)
        logits = self.head(text, valid, marker_positions, marker_mask, question_types)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=(text,))
