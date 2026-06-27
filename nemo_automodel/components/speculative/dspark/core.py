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

"""DSpark online training wrapper.

The DSpark draft is self-contained: it samples anchors, builds the block
attention mask, runs the semi-autoregressive backbone + Markov head, and emits
everything the objective needs. This module is therefore a thin wrapper that
calls the draft with the target supervision and computes the three-term loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.dspark.draft_qwen3 import Qwen3DSparkModel
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss


@dataclass
class DSparkStepMetrics:
    """Per-step training output for the DSpark draft."""

    loss: torch.Tensor


class DSparkTrainerModule(nn.Module):
    """DSpark online training wrapper computing the three-term objective."""

    def __init__(
        self,
        draft_model: Qwen3DSparkModel,
        *,
        loss_decay_gamma: Optional[float] = None,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        confidence_head_alpha: float = 1.0,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.loss_decay_gamma = loss_decay_gamma
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkStepMetrics:
        """Run the draft on the target supervision and compute the DSpark loss."""
        outputs = self.draft_model(
            input_ids=input_ids,
            target_hidden_states=target_hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden_states,
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            confidence_head_alpha=self.confidence_head_alpha,
        )
        return DSparkStepMetrics(loss=loss)


__all__ = ["DSparkTrainerModule", "DSparkStepMetrics"]
