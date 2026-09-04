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

"""DFlash 2 online training wrapper.

DFlash 2 (https://inco.ai/blog/dflash2/) keeps DFlash's parallel block draft and
adds a two-tap in-block convolution to the backbone plus a pairwise path selector
over each position's top-k candidates (see
``nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2``). The
convolution needs no new supervision -- it is part of the backbone and is trained
by the ordinary DFlash objective -- so this wrapper differs from
``DFlashTrainerModule`` in exactly one place: it adds a second term that trains
the selector::

    loss = base_loss + selector_loss_weight * selector_loss

``base_loss`` is DFlash's decay-weighted block CE over the full vocabulary, i.e.
what makes each position's candidate list good. ``selector_loss`` is a CE over
the ``selector_top_k`` candidates of that same position, scored against the
*ground-truth* predecessor (the token the walk would have committed had every
earlier position been right) and supervised with the index of the true token
inside the candidate list. Positions whose true token missed the candidate list
carry no selector signal -- there is nothing there to select -- and are excluded
from the selector term; ``candidate_recall`` reports how often that happens.

Both terms share ``base_loss``'s ``DFlashDecayLoss`` position-weighting scheme
(fixed decay for ``loss_type="dflash"``, detached D-PACE confidence for the
``"dpace*"`` variants, arXiv:2605.18810) via
:meth:`~nemo_automodel.components.loss.dllm_loss.DFlashDecayLoss.weighted_mean`,
so a position's importance is identical in the two objectives and switching
``loss_type`` rescales both consistently. The selector term's weight *schedule*
is always built from the full ``pred_mask`` -- the same mask ``base_loss`` uses
-- and only narrowed to the positions with selector signal (``has_target``)
afterward, via ``weighted_mean``'s ``value_mask``. D-PACE's weight is a
sequential cumprod/cumsum across the block, so building it from a narrower mask
directly (e.g. ``pred_mask * has_target``) would corrupt neighboring positions'
weights, not just the excluded one's. ``loss_type="variable_prefix"`` is
rejected: the selector teacher-forces the predecessor from the fixed-anchor
block layout, which a variable visible prefix breaks.

Both terms also share the base objective's ``total_blocks`` fix: D-PACE's
"mean" denominator is ``batch_size * num_anchors`` (the trainer's *configured*
block-sampling budget), not the batch's achieved sampled-block count, which
varies with each micro-batch's own content and therefore differs across DP
ranks -- see
:meth:`~nemo_automodel.components.loss.dllm_loss.DFlashDecayLoss._mean_denominator`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.speculative.dflash.core import (
    DFlashTrainerModule,
    _to_full_tensor,
    compute_acceptance_stats,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import Qwen3DFlash2DraftModel


@dataclass
class DFlash2StepMetrics:
    """Per-step training outputs for the DFlash 2 draft.

    ``loss`` / ``accuracy`` / ``valid_tokens`` mirror ``DFlashStepMetrics`` so the
    shared DFlash training loop consumes them unchanged. The primary accuracy and
    acceptance-length fields describe the *selector* path -- what the draft
    actually emits at decode time -- and the ``base_*`` fields the backbone's own
    top-1 picks, so the two are directly comparable on the same denominator.

    Attributes:
        loss: Scalar tensor containing the differentiable training loss.
        loss_weight: Scalar tensor containing the effective loss denominator.
        accuracy: Scalar tensor containing selector-path greedy token accuracy.
        valid_tokens: Scalar tensor containing the supervised-token count.
        correct_tokens: Scalar tensor containing selector-path correct-token count.
        accept_len: Scalar tensor containing selector-path mean acceptance length.
        accept_len_sum: Scalar tensor containing selector-path additive acceptance length.
        valid_blocks: Scalar tensor containing the number of evaluated draft blocks.
        base_loss: Scalar tensor containing the backbone block-CE term.
        selector_loss: Scalar tensor containing the candidate-selection CE term.
        base_accuracy: Scalar tensor containing backbone top-1 token accuracy.
        base_correct_tokens: Scalar tensor containing backbone correct-token count.
        base_accept_len: Scalar tensor containing backbone mean acceptance length.
        base_accept_len_sum: Scalar tensor containing backbone additive acceptance length.
        candidate_recall: Scalar tensor containing the fraction of supervised
            positions whose true token is in the backbone's top-k candidates --
            the ceiling the selector can reach.
    """

    loss: torch.Tensor
    loss_weight: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    correct_tokens: torch.Tensor
    accept_len: torch.Tensor
    accept_len_sum: torch.Tensor
    valid_blocks: torch.Tensor
    base_loss: torch.Tensor
    selector_loss: torch.Tensor
    base_accuracy: torch.Tensor
    base_correct_tokens: torch.Tensor
    base_accept_len: torch.Tensor
    base_accept_len_sum: torch.Tensor
    candidate_recall: torch.Tensor


class DFlash2TrainerModule(DFlashTrainerModule):
    """DFlash 2 online training wrapper: DFlash block CE + candidate-selection CE."""

    def __init__(
        self,
        draft_model: Qwen3DFlash2DraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: float | None = None,
        selector_loss_weight: float = 1.0,
        sliding_window: int | None = None,
        # Keyword-only, and appended after the pre-existing params above: an
        # old positional caller of this constructor must not have a later
        # argument silently rebound to one of these when they were inserted.
        *,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
    ):
        if loss_type == "variable_prefix":
            raise ValueError(
                "DFlash 2 does not support loss_type='variable_prefix': the candidate selector "
                "teacher-forces the predecessor from the fixed-anchor block layout, which a variable "
                "visible prefix breaks."
            )
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type=loss_type,
            dpace_alpha=dpace_alpha,
            sliding_window=sliding_window,
        )
        if getattr(draft_model, "candidate_selector", None) is None:
            raise ValueError(
                "DFlash2TrainerModule requires a DFlash 2 draft model (one carrying a candidate_selector); "
                f"got {type(draft_model).__name__}."
            )
        if selector_loss_weight < 0:
            raise ValueError(f"selector_loss_weight must be >= 0, got {selector_loss_weight}.")
        self.selector_loss_weight = float(selector_loss_weight)

    def _selector_scores(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score each supervised position's top-k candidates against its true predecessor.

        Teacher-forces the predecessor: position ``k`` is scored after the token at
        ``anchor + k - 1``, which is what the decode-time walk would have committed
        if every earlier position had been accepted. Position 1's predecessor is the
        anchor token itself, exactly as at decode time.

        Args:
            hidden: Tensor of shape [batch, blocks, depth, hidden]; the draft hidden
                states of the predicted (non-anchor) block positions.
            logits: Tensor of shape [batch, blocks, depth, vocab]; the backbone
                logits at those positions.
            target_ids: Long tensor of shape [batch, blocks, block_size]; the
                ground-truth token at ``anchor + k`` for block position ``k``, so
                ``depth == block_size - 1``.

        Returns:
            Tuple ``(scores, candidate_ids, target_index, has_target)``: ``scores``
            is a Tensor of shape [batch, blocks, depth, candidates];
            ``candidate_ids`` a Long tensor of the same shape with the scored token
            ids; ``target_index`` a Long tensor of shape [batch, blocks, depth]
            with the true token's slot in the candidate list (0 where it is
            absent); and ``has_target`` a Bool tensor of the same shape marking the
            positions where it is present.
        """
        selector = self.draft_model.candidate_selector
        unary, candidate_ids = torch.topk(logits, selector.top_k, dim=-1)
        scores = selector.pair_scores(hidden, unary, candidate_ids, target_ids[:, :, :-1])
        in_candidates = candidate_ids == target_ids[:, :, 1:].unsqueeze(-1)
        return scores, candidate_ids, in_candidates.to(torch.int64).argmax(dim=-1), in_candidates.any(dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> DFlash2StepMetrics:
        """Parallel block-wise training forward with the DFlash 2 path selector.

        Sequence packing (``position_ids`` ``[B, S]`` per-document reset positions,
        ``seq_lens`` ``[B, max_docs]`` document lengths, ``doc_remaining`` ``[B, S]``)
        is handled by the shared DFlash prologue, which keeps every block inside one
        document.

        Args:
            input_ids: Long tensor of shape [batch, sequence]; the context tokens.
            hidden_states: Tensor of shape [batch, sequence, layers * hidden]; the
                captured target-model context features.
            loss_mask: Tensor of shape [batch, sequence]; the supervised-token mask.
            position_ids: Long tensor of shape [batch, sequence] with per-document
                reset positions under packing, or ``None``.
            seq_lens: Long tensor of shape [batch, max_docs] with packed document
                lengths, or ``None`` when unpacked.
            doc_remaining: Long tensor of shape [batch, sequence]; real tokens left
                in each position's document, or ``None`` when unpacked.

        Returns:
            DFlash2StepMetrics for this micro-batch.
        """
        bsz, seq_len = input_ids.shape

        anchor_positions, block_keep_mask, noise_embedding, full_position_ids, dflash_attn_mask, _ = (
            self._prepare_block_inputs(
                input_ids, loss_mask, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
            )
        )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        # A tensor-parallel target's lm_head is column-parallel and returns
        # vocab-sharded (DTensor) logits; gather to a full tensor for the loss.
        logits = _to_full_tensor(self.lm_head(output_hidden))

        n, bs = anchor_positions.size(1), self.block_size
        _, target_ids, block_mask = self._build_block_targets(
            input_ids, loss_mask, anchor_positions, block_keep_mask, seq_len, doc_remaining=doc_remaining
        )

        # Drop block position 0 (the clean anchor token, never a target); the
        # remaining bs-1 positions are what both objectives supervise.
        pred_hidden = output_hidden.view(bsz, n, bs, -1)[:, :, 1:, :]
        pred_logits = logits.view(bsz, n, bs, -1)[:, :, 1:, :]
        pred_targets = target_ids[:, :, 1:]
        pred_mask = block_mask[:, :, 1:]

        loss_fn = self.loss_fn
        assert loss_fn is not None, "loss_fn is always constructed (loss_type='variable_prefix' is rejected)"
        # forward_with_token_nll's own per-token NLL is reused below for the
        # selector's D-PACE weighting, instead of a second full-vocabulary CE
        # pass over pred_logits.
        base_token_nll, loss_out = loss_fn.forward_with_token_nll(
            pred_logits, pred_targets, pred_mask, num_tokens=None, total_blocks=self.num_anchors
        )

        scores, candidate_ids, target_index, has_target = self._selector_scores(pred_hidden, pred_logits, target_ids)
        selector_mask = pred_mask * has_target.to(pred_mask.dtype)
        selector_nll = F.cross_entropy(
            scores.reshape(-1, scores.shape[-1]).float(), target_index.reshape(-1), reduction="none"
        ).view_as(selector_mask)
        # See the module docstring for why the schedule mask stays pred_mask and
        # has_target only narrows via value_mask.
        selector_loss = loss_fn.weighted_mean(
            selector_nll, base_token_nll, pred_mask, value_mask=selector_mask, total_blocks=self.num_anchors
        )
        loss = loss_out.total_loss + self.selector_loss_weight * selector_loss

        with torch.no_grad():
            eval_mask = pred_mask.bool()
            valid_tokens = pred_mask.sum()
            selected_ids = candidate_ids.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
            correct_tokens = ((selected_ids == pred_targets) & eval_mask).sum()
            accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(selected_ids, pred_targets, eval_mask)
            base_ids = pred_logits.argmax(dim=-1)
            base_correct_tokens = ((base_ids == pred_targets) & eval_mask).sum()
            base_accept_len, base_accept_len_sum, _ = compute_acceptance_stats(base_ids, pred_targets, eval_mask)
            denominator = valid_tokens.clamp_min(1)

        return DFlash2StepMetrics(
            loss=loss,
            loss_weight=loss_out.loss_denominator.detach(),
            accuracy=(correct_tokens / denominator).detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct_tokens.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
            base_loss=loss_out.total_loss.detach(),
            selector_loss=selector_loss.detach(),
            base_accuracy=(base_correct_tokens / denominator).detach(),
            base_correct_tokens=base_correct_tokens.detach(),
            base_accept_len=base_accept_len.detach(),
            base_accept_len_sum=base_accept_len_sum.detach(),
            candidate_recall=(selector_mask.sum() / denominator).detach(),
        )
