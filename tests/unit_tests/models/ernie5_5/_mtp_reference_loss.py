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

"""Reference row-mean ERNIE MTP objective for independent numerical tests.

Training semantics reference: NVIDIA-NeMo/Megatron-Bridge at
60f7921aa13bff30d801c6b37c063a29786e3626,
src/megatron/bridge/models/ernie/modeling_ernie55/mtp_loss.py (Apache-2.0).
Independent implementation; token shifting reuses AutoModel's common helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.common.mtp import shift_packed_tensor


@dataclass(frozen=True)
class Ernie5_5MTPLoss:
    """Loss and diagnostics retaining the autograd graph.

    Attributes:
        loss: Scalar coefficient-weighted mean across rows, including empty rows.
        row_means: Unweighted FP32 tensor of shape [batch].
        label_counts: Integer tensor of shape [batch], after shifting/masking.
        targets: Integer tensor of shape [batch, sequence].
    """

    loss: torch.Tensor
    row_means: torch.Tensor
    label_counts: torch.Tensor
    targets: torch.Tensor


def ernie5_5_mtp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    coefficient: float = 0.1,
    document_ids: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> Ernie5_5MTPLoss:
    """Predict one token beyond already-shifted main labels at CP1/PP1.

    This is deliberately separate from generic ``calculate_mtp_loss``: its
    main-label denominator is not this post-roll, per-row denominator. A test
    using the reference objective explicitly adds this result to its main loss.

    Args:
        logits: Post-MultiMax tensor of shape [batch, sequence, vocab].
        labels: Already-shifted main targets of shape [batch, sequence]. Invalid
            main targets, including document-boundary targets, must be ignored.
        coefficient: Nonnegative finite auxiliary-loss coefficient.
        document_ids: Optional integer tensor of shape [batch, sequence]; shifting
            never takes a target across different IDs. Global unsharded layout only.
        ignore_index: Sentinel for an ignored target.

    Returns:
        Scalar differentiable loss, row means/counts and independently owned targets
        in the layouts documented by ``Ernie5_5MTPLoss``. Empty rows contribute zero.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("MTP coefficient must be finite and nonnegative")
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("MTP loss requires nonempty batch and sequence dimensions")
    targets = shift_packed_tensor(labels, depth=1, seq_idx=document_ids, fill_value=ignore_index)
    counts = (targets != ignore_index).sum(dim=1)
    token_losses = F.cross_entropy(
        logits.float().flatten(0, 1), targets.flatten(), ignore_index=ignore_index, reduction="none"
    ).reshape_as(targets)
    row_means = token_losses.sum(dim=1) / counts.clamp_min(1)
    return Ernie5_5MTPLoss(coefficient * row_means.mean(), row_means, counts, targets)
