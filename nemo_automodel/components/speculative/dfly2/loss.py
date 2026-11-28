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

"""DFly2 training objective: D-PACE-weighted LK distillation plus an optional confidence head.

Per drafted slot, with ``p`` the target distribution and ``q`` the draft's:

* ``TV = 1/2 sum|p - q|`` and ``KL = sum p (log p - log q)`` over the full vocabulary;
* the LK hybrid ``lambda * KL + (1 - lambda) * TV`` with ``lambda = exp(-eta * sg(1 - TV))``,
  which distills while the draft is far from the target and turns to the acceptance
  probability ``1 - TV`` as it gets close;
* D-PACE position weights (arXiv:2605.18810), computed detached from the draft's
  cross entropy on the corpus token, weight both terms;
* the confidence head regresses the slot's acceptance probability ``sum min(p, q)``.

Every term is a weighted sum over the local slots divided by the weight sum of all
data-parallel ranks, and the backward loss is scaled by the world size, so the
averaged gradient equals the global weighted mean regardless of how slots split
across ranks (the DSpark objective's convention).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.loss.dllm_loss import dpace_position_weights
from nemo_automodel.components.speculative.dspark.common import DSparkForwardOutput
from nemo_automodel.components.speculative.dspark.loss import _collect_acceptance_diagnostics

_PROBABILITY_CHUNK_TOKENS = 128


def _distance_chunk(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 total variation and forward KL for a chunk of slots.

    Args:
        draft_logits: Tensor of shape [tokens, vocab].
        target_logits: Tensor of shape [tokens, vocab].

    Returns:
        ``(tv, kl)``, each of shape [tokens].
    """
    draft_log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    target_log_probs = torch.log_softmax(target_logits.float(), dim=-1)
    target_probs = target_log_probs.exp()
    tv = 0.5 * (target_probs - draft_log_probs.exp()).abs().sum(dim=-1)
    kl = (target_probs * (target_log_probs - draft_log_probs)).sum(dim=-1)
    return tv, kl


def _compute_distances(
    draft_logits: torch.Tensor, target_logits: torch.Tensor, chunk_size: int = _PROBABILITY_CHUNK_TOKENS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-slot TV and KL without full-vocabulary FP32 temporaries for every slot.

    Each chunk is recomputed in the backward pass, so only one chunk's probabilities are
    alive at a time.

    Args:
        draft_logits: Tensor of shape [..., vocab].
        target_logits: Tensor of shape [..., vocab], detached.
        chunk_size: Slots per chunk.

    Returns:
        ``(tv, kl)``, each of shape [...].
    """
    output_shape = draft_logits.shape[:-1]
    vocab_size = draft_logits.shape[-1]
    flat_draft = draft_logits.reshape(-1, vocab_size)
    flat_target = target_logits.reshape(-1, vocab_size)
    tvs, kls = [], []
    for start in range(0, flat_draft.shape[0], chunk_size):
        draft_chunk = flat_draft[start : start + chunk_size]
        target_chunk = flat_target[start : start + chunk_size]
        if torch.is_grad_enabled() and draft_chunk.requires_grad:
            tv, kl = checkpoint(
                _distance_chunk, draft_chunk, target_chunk, use_reentrant=False, preserve_rng_state=False
            )
        else:
            tv, kl = _distance_chunk(draft_chunk, target_chunk)
        tvs.append(tv)
        kls.append(kl)
    return torch.cat(tvs).reshape(output_shape), torch.cat(kls).reshape(output_shape)


def compute_dfly2_loss(
    *,
    outputs: DSparkForwardOutput,
    dpace_alpha: float,
    lk_eta: float,
    confidence_head_alpha: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the DFly2 objective and its logged terms for one micro-batch.

    Args:
        outputs: Draft outputs over the drafted slots only: ``draft_logits`` and
            ``aligned_target_logits`` of shape [batch, num_anchors, drafted, vocab],
            ``target_ids`` and ``eval_mask`` of shape [batch, num_anchors, drafted],
            ``block_keep_mask`` of shape [batch, num_anchors], and an optional
            ``confidence_pred`` of shape [batch, num_anchors, drafted].
        dpace_alpha: D-PACE smoothing in ``[0, 1]``.
        lk_eta: Sharpness of the LK mixing weight; larger values switch to TV sooner.
        confidence_head_alpha: Weight of the confidence loss; ignored without a head.

    Returns:
        The backward loss (scaled by the data-parallel world size, see the module
        docstring) and detached terms: ``loss``, ``ce_loss`` (the corpus cross entropy,
        logged only), ``l1_loss`` (the TV term), ``lk_loss``, ``confidence_loss``, and the
        unreduced acceptance-diagnostic sums of the DSpark objective.

    Raises:
        ValueError: ``aligned_target_logits`` is missing.
    """
    if outputs.aligned_target_logits is None:
        raise ValueError("DFly2 distills from the target: aligned_target_logits is required")
    draft_logits = outputs.draft_logits
    target_logits = outputs.aligned_target_logits.detach()
    eval_mask = outputs.eval_mask
    mask = eval_mask.to(torch.float32)

    token_ce = F.cross_entropy(
        draft_logits.flatten(0, -2).float(), outputs.target_ids.reshape(-1), reduction="none"
    ).view_as(mask)
    weights = dpace_position_weights(token_ce.detach(), mask, eval_mask, dpace_alpha=dpace_alpha) * mask
    tv, kl = _compute_distances(draft_logits, target_logits)
    mixing = torch.exp(-float(lk_eta) * (1.0 - tv).detach())
    lk = mixing * kl + (1.0 - mixing) * tv
    accept_rate = (1.0 - tv).detach().clamp(0.0, 1.0)

    has_confidence = outputs.confidence_pred is not None
    confidence = mask.new_zeros(())
    if has_confidence:
        errors = F.binary_cross_entropy_with_logits(outputs.confidence_pred.float(), accept_rate, reduction="none")
        confidence = (errors * weights).sum()

    local_denominator = weights.sum()
    global_denominator = local_denominator.detach().clone()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
    numerator = (lk * weights).sum() + float(confidence_head_alpha) * confidence
    backward_loss = numerator / (global_denominator + 1e-6) * world_size

    local = 1.0 / (local_denominator.detach() + 1e-6)
    terms = {
        "loss": numerator.detach() * local,
        "ce_loss": (token_ce * weights).sum().detach() * local,
        "l1_loss": (tv * weights).sum().detach() * local,
        "lk_loss": (lk * weights).sum().detach() * local,
        "confidence_loss": confidence.detach() * local,
    }
    with torch.no_grad():
        diagnostics = _collect_acceptance_diagnostics(
            outputs=outputs, accept_rate_3d=accept_rate, loss_weight_mask=weights, has_confidence=has_confidence
        )
    terms["accept_rate_per_pos_num"] = diagnostics["accept_rate_pos_num"]
    terms["accept_rate_per_pos_den"] = diagnostics["accept_rate_pos_den"]
    for key in (
        "tau_num",
        "tau_den",
        "confidence_abs_error_num",
        "confidence_bias_num",
        "confidence_cumprod_bias_num",
        "confidence_diag_den",
    ):
        terms[key] = diagnostics[key]
    return backward_loss, terms


__all__ = ["compute_dfly2_loss"]
