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

"""Model-owned contracts and the frozen-target wrapper for DFly2 training.

DFly2 reads the target's own attention state, whose layout is architecture specific, so
the target model produces it (:class:`DFly2TargetFeatureProvider`) and the target config
builds the matching draft (:class:`ModelOwnedDFly2Provider`). This module only names those
contracts and runs the frozen forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.dfly2.common import DFly2TargetFeatures


class _DFly2DraftOptions(Protocol):
    """Recipe options consumed by model-owned DFly2 draft constructors."""

    target_layer_ids: list[int]
    block_size: int
    num_anchors: int
    mask_token_id: int
    enable_confidence_head: bool
    confidence_head_stop_gradient: bool
    attn_backend: str


@runtime_checkable
class ModelOwnedDFly2Provider(Protocol):
    """Target config that owns construction of its architecture-specific DFly2 draft."""

    def build_dfly2_draft(self, options: _DFly2DraftOptions) -> nn.Module:
        """Build a draft from the recipe's declarative options."""


@runtime_checkable
class DFly2TargetFeatureProvider(Protocol):
    """Target model that exposes the attention state its paired DFly2 draft layers read."""

    def dfly2_target_features(
        self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor, layer_ids: Sequence[int]
    ) -> DFly2TargetFeatures:
        """Run the frozen forward and return the paired layers' keys and key lists.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Binary right-padding tensor of shape [batch, sequence].
            layer_ids: Target layers paired with the draft layers, in draft-layer order.
        """


@dataclass
class DFly2TargetBatch:
    """Target features and the batch tensors the DFly2 trainer reads.

    Attributes:
        features: Frozen target state, documented by :class:`DFly2TargetFeatures`.
        input_ids: Integer tensor of shape [batch, sequence].
        loss_mask: Tensor of shape [batch, sequence] selecting supervised tokens.
    """

    features: DFly2TargetFeatures
    input_ids: torch.Tensor
    loss_mask: torch.Tensor


class DFly2TargetModel:
    """Run a frozen target that implements :class:`DFly2TargetFeatureProvider`."""

    def __init__(self, model: nn.Module, target_layer_ids: Sequence[int]):
        if not isinstance(model, DFly2TargetFeatureProvider):
            raise TypeError(f"{type(model).__name__} does not provide DFly2 target features")
        self.model = model.eval()
        self.target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]

    def get_input_embeddings(self) -> nn.Module:
        """Return the target's input embedding."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        """Return the target's LM head."""
        return self.model.get_output_embeddings()

    @torch.no_grad()
    def generate_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor
    ) -> DFly2TargetBatch:
        """Run the target once and collect the DFly2 features.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Binary right-padding tensor of shape [batch, sequence].
            loss_mask: Tensor of shape [batch, sequence] selecting supervised tokens.

        Returns:
            The batch with the features of :meth:`DFly2TargetFeatureProvider.dfly2_target_features`.
        """
        features = self.model.dfly2_target_features(
            input_ids, attention_mask=attention_mask, layer_ids=self.target_layer_ids
        )
        return DFly2TargetBatch(features=features, input_ids=input_ids, loss_mask=loss_mask)


__all__ = [
    "DFly2TargetBatch",
    "DFly2TargetFeatureProvider",
    "DFly2TargetModel",
    "ModelOwnedDFly2Provider",
]
