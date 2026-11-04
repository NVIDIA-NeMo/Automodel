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

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedScoreKLLoss(nn.Module):
    """Forward-KL distillation over per-query cross-encoder score groups.

    Both inputs are raw scores, i.e. the model's ``logits`` already multiplied back by its
    own ``config.temperature``. Softmax is taken over the candidate dimension so the
    distillation target is the teacher's ranking distribution within each query group.

    Args:
        temperature: Softmax temperature applied to both score sets. Must be > 0.
        reference_temperature: Scale the loss by ``(temperature / reference_temperature)**2``
            to keep KD gradient magnitudes comparable to a cross-entropy term computed at
            ``reference_temperature``. Never applied to the scores themselves. The
            cross-encoder recipe passes the student's internal ``config.temperature``, since
            the cross-entropy term is student-only. Pass ``None`` to disable the correction.
    """

    def __init__(self, temperature: float, reference_temperature: float | None = None) -> None:
        super().__init__()
        # isfinite first: NaN fails every comparison, so a bare `<= 0` check would accept it and
        # only surface as a NaN loss once training is under way.
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be > 0 and finite, got {temperature}")
        if reference_temperature is not None and (
            not math.isfinite(reference_temperature) or reference_temperature <= 0
        ):
            raise ValueError(f"reference_temperature must be > 0 and finite when provided, got {reference_temperature}")
        self.temperature = float(temperature)
        self.reference_temperature = None if reference_temperature is None else float(reference_temperature)

    def forward(self, student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        """Compute ``KL(teacher || student)`` averaged over query groups.

        Args:
            student_scores: Raw student scores of shape ``[batch, n_passages]``.
            teacher_scores: Raw teacher scores of shape ``[batch, n_passages]``. Detached
                internally, so no gradient flows to the teacher.

        Returns:
            Scalar FP32 tensor.
        """
        if student_scores.ndim != 2 or teacher_scores.ndim != 2:
            raise ValueError(
                "GroupedScoreKLLoss expects 2-D [batch, n_passages] scores, got "
                f"student {tuple(student_scores.shape)} and teacher {tuple(teacher_scores.shape)}. "
                "Reshape flattened cross-encoder logits with view(-1, n_passages) first."
            )
        if student_scores.shape != teacher_scores.shape:
            raise ValueError(
                f"Shape mismatch: student {tuple(student_scores.shape)} vs teacher {tuple(teacher_scores.shape)}."
            )

        student_log_probs = F.log_softmax(student_scores.float() / self.temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_scores.float().detach() / self.temperature, dim=-1)

        loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True)
        if self.reference_temperature is not None:
            # KL gradients scale roughly as 1/T**2, so using a KD temperature different from the
            # reference temperature of the blended loss changes the KD gradient magnitude by
            # roughly the squared temperature ratio. Compensating with (T / T_ref)**2 keeps
            # kd_loss_weight meaningful.
            loss = loss * (self.temperature / self.reference_temperature) ** 2
        return loss
