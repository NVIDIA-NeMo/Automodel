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

"""DFly2 online training wrapper.

Like DSpark's, the draft is self-contained (anchors, key lists, backbone, corrections), so
this wrapper only calls it with the frozen target features and computes the DFly2
objective. It reports :class:`DSparkStepMetrics` so the DSpark recipe's metric window,
evaluation and logging apply unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.speculative.dfly2.loss import compute_dfly2_loss
from nemo_automodel.components.speculative.dspark.core import DSparkStepMetrics


class DFly2TrainerModule(nn.Module):
    """DFly2 online training wrapper computing the D-PACE-weighted LK objective."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        dpace_alpha: float = 0.5,
        lk_eta: float = 3.0,
        confidence_head_alpha: float = 1.0,
    ):
        super().__init__()
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        self.draft_model = draft_model
        self.dpace_alpha = float(dpace_alpha)
        self.lk_eta = float(lk_eta)
        self.confidence_head_alpha = float(confidence_head_alpha)

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        context_keys: Sequence[torch.Tensor],
        context_indices: Sequence[torch.Tensor],
        target_last_hidden_states: torch.Tensor,
    ) -> DSparkStepMetrics:
        """Run the draft on the target features and compute the DFly2 loss.

        Args:
            input_ids: Target-token tensor of shape [batch, sequence].
            loss_mask: Supervision tensor of shape [batch, sequence].
            context_keys: One tensor per draft layer of shape [batch, context_keys,
                key_dim], as in :class:`DFly2TargetFeatures`.
            context_indices: One integer tensor per draft layer of shape [batch,
                sequence, slots], as in :class:`DFly2TargetFeatures`.
            target_last_hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Step metrics. The per-position acceptance sums have ``block_size`` entries,
            entry ``k`` for the ``k``-th drafted token; the anchor slot is not drafted,
            so the last entry stays zero and is not logged.
        """
        outputs = self.draft_model(
            input_ids=input_ids,
            loss_mask=loss_mask,
            context_keys=context_keys,
            context_indices=context_indices,
            target_last_hidden_states=target_last_hidden_states,
        )
        loss, terms = compute_dfly2_loss(
            outputs=outputs,
            dpace_alpha=self.dpace_alpha,
            lk_eta=self.lk_eta,
            confidence_head_alpha=self.confidence_head_alpha,
        )
        return DSparkStepMetrics(
            loss=loss,
            ce_loss=terms["ce_loss"],
            l1_loss=terms["l1_loss"],
            confidence_loss=terms["confidence_loss"],
            accept_rate_per_pos_num=F.pad(terms["accept_rate_per_pos_num"], (0, 1)),
            accept_rate_per_pos_den=F.pad(terms["accept_rate_per_pos_den"], (0, 1)),
            tau_num=terms["tau_num"],
            tau_den=terms["tau_den"],
            confidence_abs_error_num=terms["confidence_abs_error_num"],
            confidence_bias_num=terms["confidence_bias_num"],
            confidence_cumprod_bias_num=terms["confidence_cumprod_bias_num"],
            confidence_diag_den=terms["confidence_diag_den"],
        )


__all__ = ["DFly2TrainerModule"]
