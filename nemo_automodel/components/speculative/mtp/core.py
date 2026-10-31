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

"""Self-speculative acceptance-length evaluation for a model's native MTP head.

MTP finetuning trains the existing :class:`MTPModule` depths directly (see
``nemo_automodel.components.loss.mtp.calculate_mtp_loss``), but that loss only
reports scalar per-depth CE -- it says nothing about how many tokens the MTP
head would actually get accepted under self-speculative decoding. This module
computes that acceptance-length proxy from the same per-depth outputs the loss
already consumes, without an extra model forward pass.

Acceptance is checked against the model's own main-head greedy prediction (the
actual verifier in self-speculative decoding), not against the dataset label:
a draft is "accepted" when it agrees with what the model itself would have
produced, which can diverge from the ground-truth label whenever the main head
is imperfect. The dataset label only decides which positions are supervised
(masking ignored/padded/cross-document positions), the same role it plays in
``calculate_mtp_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.mtp import resolve_mtp_seq_idx, roll_tensor
from nemo_automodel.components.speculative.eagle.core import simulated_accept_length
from nemo_automodel.components.utils.model_utils import get_lm_head_module

__all__ = ["MTPAcceptanceMetrics", "compute_mtp_prefix_hits", "simulated_accept_length"]


@dataclass
class MTPAcceptanceMetrics:
    """Per-depth joint-prefix-hit counts for :func:`simulated_accept_length`.

    ``prefix_hits[k]`` counts positions whose greedy MTP chain is still fully
    correct through depth ``k + 1`` (argmax hit at every depth ``<= k``), and
    ``valid[k]`` counts positions supervised at every depth ``<= k``. Kept
    unreduced (shape ``[num_depths]``) so a recipe can sum them across
    microbatches and the DP/CP group before dividing once.
    """

    prefix_hits: torch.Tensor
    valid: torch.Tensor


def _roll_per_depth(
    base: torch.Tensor,
    num_depths: int,
    fill_value: int,
    seq_idx: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Cumulative per-depth left-roll of ``base``, mirroring ``calculate_mtp_loss``'s label roll.

    ``fill_value`` only matters at trailing/cross-sub-sequence positions, which
    the caller always excludes from both the hit and valid counts.
    """
    cur = base
    rolled = []
    for k in range(num_depths):
        cur = roll_tensor(cur, shifts=-1, dim=-1)
        masked = cur.clone()
        n_invalid = min(k + 1, masked.shape[-1])
        masked[..., -n_invalid:] = fill_value
        if seq_idx is not None:
            rolled_seq_idx = roll_tensor(seq_idx, shifts=-(k + 1), dim=-1)
            cross_seq = rolled_seq_idx != seq_idx
            masked = torch.where(cross_seq, torch.full_like(masked, fill_value), masked)
        rolled.append(masked)
    return rolled


@torch.no_grad()
def compute_mtp_prefix_hits(
    model: nn.Module,
    *,
    mtp_per_depth_h: list[torch.Tensor] | None = None,
    mtp_per_depth_logits: list[torch.Tensor] | None = None,
    verifier_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    seq_idx: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> MTPAcceptanceMetrics:
    """Score a model's MTP depths against its own main-head greedy prediction.

    Args:
        model: The wrapped model; used to fetch the shared LM head when
            ``mtp_per_depth_h`` is supplied instead of logits.
        mtp_per_depth_h: Per-depth hidden-state tensors of shape
            ``[batch, sequence, hidden]``.
        mtp_per_depth_logits: Per-depth logit tensors of shape
            ``[batch, sequence, vocab]``.
        verifier_logits: The model's own main-head logits, ``[batch, sequence,
            vocab]``, from the same forward pass. Its argmax is the acceptance
            reference -- not ``labels``, which can disagree with what the
            model itself would produce.
        labels: Original unshifted label tensor of shape ``[batch, sequence]``,
            used only to mask ignored/padded positions out of the counts.
        ignore_index: Label value excluded from both the hit and valid counts.
        seq_idx: Optional per-token sub-sequence index used to mask rolls
            whose source position crosses a packed-sequence boundary. Takes
            precedence over ``cu_seqlens``.
        cu_seqlens: Optional cumulative sequence lengths ``[num_seqs+1]``
            (THD-pack layout) used to derive ``seq_idx`` when it is not given,
            the same boundary ``calculate_mtp_loss`` uses for this batch.

    Returns:
        :class:`MTPAcceptanceMetrics` with one raw sum per depth.
    """
    seq_idx = resolve_mtp_seq_idx(labels, seq_idx, cu_seqlens)
    if mtp_per_depth_logits is not None:
        if mtp_per_depth_h is not None:
            raise ValueError("Provide exactly one of mtp_per_depth_h or mtp_per_depth_logits")
        mtp_outputs = mtp_per_depth_logits
        lm_head = None
    elif mtp_per_depth_h is not None:
        mtp_outputs = mtp_per_depth_h
        lm_head = get_lm_head_module(model)
        if lm_head is None:
            raise ValueError("lm_head module not found in model")
    else:
        raise ValueError("Provide exactly one of mtp_per_depth_h or mtp_per_depth_logits")

    num_depths = len(mtp_outputs)
    if labels.dim() == 1:
        mtp_outputs = [h.squeeze(0) if (h.dim() == 3 and h.shape[0] == 1) else h for h in mtp_outputs]
        if verifier_logits.dim() == 3 and verifier_logits.shape[0] == 1:
            verifier_logits = verifier_logits.squeeze(0)

    label_targets = _roll_per_depth(labels, num_depths, ignore_index, seq_idx)
    verifier_targets = _roll_per_depth(verifier_logits.argmax(dim=-1), num_depths, ignore_index, seq_idx)

    prefix_correct: torch.Tensor | None = None
    prefix_valid: torch.Tensor | None = None
    hits = []
    valids = []
    for lt, vt, mtp_output in zip(label_targets, verifier_targets, mtp_outputs):
        logits = mtp_output if mtp_per_depth_logits is not None else lm_head(mtp_output)
        valid_mask = lt != ignore_index
        correct = (logits.argmax(dim=-1) == vt) & valid_mask

        prefix_correct = correct if prefix_correct is None else prefix_correct & correct
        prefix_valid = valid_mask if prefix_valid is None else prefix_valid & valid_mask
        hits.append(prefix_correct.sum())
        valids.append(prefix_valid.sum())

    return MTPAcceptanceMetrics(
        prefix_hits=torch.stack(hits).detach(),
        valid=torch.stack(valids).detach(),
    )
