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
    """Per-step training outputs for the DSpark draft.

    Beyond the loss and its three terms, this carries the acceptance diagnostics as
    unreduced ``(num, den)`` sums so the recipe can reduce both across the log
    window and the DP group and form the exact global ratio once: acceptance rate
    (the ``[block_size]`` per-position ``accept_rate@k`` sums, whose totals also
    give the aggregate ``accept_rate``), ``tau`` (expected accepted block length),
    and the confidence-head calibration error/bias against the measured acceptance
    rate. A denominator is zero when the diagnostic was not computed (e.g. no
    confidence head, or no teacher signal), which the recipe uses to skip logging.
    """

    loss: torch.Tensor
    ce_loss: torch.Tensor
    l1_loss: torch.Tensor
    confidence_loss: torch.Tensor
    accept_rate_per_pos_num: torch.Tensor
    accept_rate_per_pos_den: torch.Tensor
    tau_num: torch.Tensor
    tau_den: torch.Tensor
    confidence_abs_error_num: torch.Tensor
    confidence_bias_num: torch.Tensor
    confidence_cumprod_bias_num: torch.Tensor
    confidence_diag_den: torch.Tensor


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
        loss, terms = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            confidence_head_alpha=self.confidence_head_alpha,
            return_terms=True,
        )
        return DSparkStepMetrics(
            loss=loss,
            ce_loss=terms["ce_loss"],
            l1_loss=terms["l1_loss"],
            confidence_loss=terms["confidence_loss"],
            accept_rate_per_pos_num=terms["accept_rate_per_pos_num"],
            accept_rate_per_pos_den=terms["accept_rate_per_pos_den"],
            tau_num=terms["tau_num"],
            tau_den=terms["tau_den"],
            confidence_abs_error_num=terms["confidence_abs_error_num"],
            confidence_bias_num=terms["confidence_bias_num"],
            confidence_cumprod_bias_num=terms["confidence_cumprod_bias_num"],
            confidence_diag_den=terms["confidence_diag_den"],
        )


__all__ = ["DSparkTrainerModule", "DSparkStepMetrics"]
