# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Model-owned frozen text features and released D1 option-scoring semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from nemo_automodel.components.models.d1.decision_head import D1TokenReadoutConfig
from nemo_automodel.components.models.d1.records import D1TextExample

if TYPE_CHECKING:
    from nemo_automodel.components.models.d1.model import D1OmniModel
    from nemo_automodel.components.models.d1.vl_model import Lfm2VlForConditionalGeneration


class D1TextReadout(Protocol):
    """One variant's feature boundary and trainable released readout.

    ``head`` is the only trainable module in this frozen-feature workflow.
    A feature must be scored by the readout that produced it: causal features
    have shape [hidden], while Omni features have shape [sequence, hidden].
    """

    @property
    def head(self) -> nn.Module:
        """Return the trainable released head or additive answer-row adapter."""
        ...

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode one unpadded text question without applying the readout.

        Args:
            input_ids: Long Tensor of shape [1, sequence] on the model device.
            attention_mask: Boolean Tensor of shape [1, sequence], all true.

        Returns:
            Tensor of shape [hidden] for a causal final-token feature, or
            [sequence, hidden] for Omni normalized text states. The output
            retains its autograd graph; the caching caller chooses no_grad.
        """
        ...

    def score(
        self,
        features: Sequence[torch.Tensor],
        examples: Sequence[D1TextExample],
        *,
        head: nn.Module,
    ) -> list[torch.Tensor]:
        """Score aligned features using the variant's option and calibration rules.

        Args:
            features: Tensors of shape [hidden] for the causal variant, or
                [sequence, hidden] for Omni, on the head device, one per example.
            examples: Publisher-encoded questions in the same order as features.
            head: This variant's trainable readout, or a frozen reference copy.

        Returns:
            List of float32 Tensors of shape [options], one per example in its
            declared example option order, retaining gradients to the head.
        """
        ...


def _validate_text_input(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    """Require the single-question, unpadded frozen-feature boundary.

    Args:
        input_ids: Long Tensor of shape [1, sequence] with nonempty sequence.
        attention_mask: Boolean Tensor of shape [1, sequence], all true, on the
            same device as input_ids.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0 or input_ids.dtype != torch.long:
        raise ValueError("D1 text features require long input_ids with shape [1, nonempty_sequence]")
    if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool or not attention_mask.all():
        raise ValueError("D1 text features require an all-true boolean attention_mask matching input_ids")
    if attention_mask.device != input_ids.device:
        raise ValueError("D1 input_ids and attention_mask must be on the same device")


def _validate_feature_pairs(features: Sequence[torch.Tensor], examples: Sequence[D1TextExample]) -> None:
    """Reject missing feature slots before variant-specific scoring.

    Args:
        features: Tensors of shape [hidden] or [sequence, hidden], one per example.
        examples: Publisher-encoded questions in feature order.
    """
    if not features or len(features) != len(examples):
        raise ValueError("D1 scoring requires equal, nonempty feature and example counts")


class D1CausalTextReadout:
    """Final-token features and max-over-answer-alias logits for D1-3B."""

    def __init__(self, model: Lfm2VlForConditionalGeneration, examples: Sequence[D1TextExample]) -> None:
        self.model = model
        self.token_ids = tuple(sorted({token for ex in examples for group in ex.alias_token_ids for token in group}))
        self.token_index = {token: index for index, token in enumerate(self.token_ids)}
        weight = model.get_output_embeddings().weight
        self.head = D1TokenReadoutConfig(weight.shape[1], self.token_ids).build(weight)

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the causal final-token state before the tied vocabulary projection.

        Args:
            input_ids: Long Tensor of shape [1, sequence] on the model device.
            attention_mask: Boolean Tensor of shape [1, sequence], all true.

        Returns:
            Tensor of shape [hidden] containing the final normalized token state.
            It retains the backbone graph unless the caller uses no_grad.
        """
        _validate_text_input(input_ids, attention_mask)
        return self.model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state[
            0, -1
        ]

    def score(
        self,
        features: Sequence[torch.Tensor],
        examples: Sequence[D1TextExample],
        *,
        head: nn.Module,
    ) -> list[torch.Tensor]:
        """Apply answer-alias maxima without introducing a new temperature.

        Args:
            features: Tensors of shape [hidden] on the head device.
            examples: Causal questions whose alias vocabulary IDs were prepared
                by this readout, in feature order.
            head: The additive answer-row adapter or a frozen reference copy.

        Returns:
            List of float32 Tensors of shape [options], in each example's option order.
        """
        _validate_feature_pairs(features, examples)
        result = []
        for feature, ex in zip(features, examples):
            if feature.ndim != 1 or ex.marker_positions or not ex.alias_token_ids or ex.temperature != 1.0:
                raise ValueError("D1-3B requires final-token features and uncalibrated answer-alias questions")
            aliases = tuple(tuple(self.token_index[token] for token in group) for group in ex.alias_token_ids)
            result.append(head(feature, aliases))
        return result


class D1OmniTextReadout:
    """Sequence features and calibrated marker logits for D1 Omni text."""

    def __init__(self, model: D1OmniModel) -> None:
        self.model = model
        self.head = model.head.float()

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized text states before the released marker head.

        Args:
            input_ids: Long Tensor of shape [1, sequence] on the model device.
            attention_mask: Boolean Tensor of shape [1, sequence], all true.

        Returns:
            Tensor of shape [sequence, hidden], retaining its backbone graph
            unless the caller uses no_grad.
        """
        _validate_text_input(input_ids, attention_mask)
        states, _ = self.model.encode_text(input_ids, attention_mask)
        return states[0]

    def score(
        self,
        features: Sequence[torch.Tensor],
        examples: Sequence[D1TextExample],
        *,
        head: nn.Module,
    ) -> list[torch.Tensor]:
        """Pad text features, score marker rows, and apply released calibration.

        Args:
            features: Tensors of shape [sequence, hidden] on the head device;
                sequence lengths may differ. Readout arithmetic uses float32.
            examples: Marker questions with released text temperatures, aligned
                with features and retaining internal false/true noul order.
            head: The released Omni marker head or a frozen reference copy.

        Returns:
            List of float32 Tensors of shape [options], in each example's option order,
            divided by that question's released text temperature.
        """
        _validate_feature_pairs(features, examples)
        for feature, ex in zip(features, examples):
            if feature.ndim != 2 or ex.alias_token_ids or not ex.marker_positions:
                raise ValueError("D1 Omni requires sequence features and marker questions")
            if (
                not math.isfinite(ex.temperature)
                or ex.temperature <= 0
                or len(ex.marker_positions) != len(ex.option_ids)
            ):
                raise ValueError("D1 Omni requires positive temperatures and one marker per encoded option")
            if min(ex.marker_positions) < 0 or max(ex.marker_positions) >= feature.shape[0]:
                raise ValueError("D1 Omni marker positions must index their unpadded text features")
        states = nn.utils.rnn.pad_sequence(features, batch_first=True).float()
        device = states.device
        lengths = torch.tensor([feature.shape[0] for feature in features], device=device)
        valid = torch.arange(states.shape[1], device=device)[None] < lengths[:, None]
        width = max(len(ex.marker_positions) for ex in examples)
        positions = torch.zeros(len(examples), width, device=device, dtype=torch.long)
        markers = torch.zeros_like(positions, dtype=torch.bool)
        for index, ex in enumerate(examples):
            positions[index, : len(ex.marker_positions)] = torch.tensor(ex.marker_positions, device=device)
            markers[index, : len(ex.marker_positions)] = True
        types = torch.tensor([ex.head_question_type for ex in examples], device=device)
        values = head(states, valid, positions, markers, types)
        return [row[: len(ex.option_ids)] / ex.temperature for row, ex in zip(values, examples)]
