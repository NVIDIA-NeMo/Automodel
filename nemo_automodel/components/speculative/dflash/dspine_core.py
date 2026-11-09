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
#
# Adapted from SpecForge (MIT License):
#   https://github.com/sgl-project/SpecForge
#   revision 2dadea10e317842af21c79e2097dc05abe1e2b5a (head of PR #1038)
#   specforge/algorithms/dspine/model.py (OnlineDSpineModel objective)
#   specforge/modeling/draft/dspine.py (DSpineConfig.schedule)
# SpecForge upstream copyright notice: Copyright (c) 2025 sgl-project.
# Licensed under the MIT License - https://github.com/sgl-project/SpecForge/blob/2dadea10e317842af21c79e2097dc05abe1e2b5a/LICENSE

"""DSpine online training wrapper.

DSpine (https://github.com/Tencent/DSpine) keeps DFlash's anchor sampling and
parallel block draft, switches the in-block attention to causal, and adds an
adjacent causal injection after every draft layer (see
``nemo_automodel.components.speculative.dflash.draft_qwen3_dspine``). The
objective follows the corrected SpecForge DSpine objective (sgl-project/SpecForge
#1038, from the DSpine authors). Over the supervised block positions
``k = 1..block_size-1`` with weights ``w_k = exp(-(k-1)/gamma)``, where the
supervision stops at a block's first unsupervised token::

    backbone   = a * sum(w * CE(y)) + (1-a) * sum(w * L1(p_draft, p_target))
    refinement = (a * mean_w(CE_topk) + (1-a) * mean_w(L1_topk)) * num_supervised
    alignment  = mean_layers(sum(w * (1 - cos(code_l, E[y]))))
    loss       = (backbone + refinement + beta_t * stopgrad(CE_bb) * alignment) / batch

* ``L1`` is the full-vocabulary L1 distance between the draft's and the frozen
  target's next-token distributions (the target logits at ``anchor + k - 1``).
* The refinement terms rescore each position's top-``head_topk`` backbone
  candidates after redoing the last write with the *true* previous token -- the
  exact-sender refinement served at decode time -- and supervise them with the
  label's candidate slot (only where the label is a candidate) and the target
  distribution renormalized over the candidates (only where the target puts more
  than ``candidate_mass_threshold`` of its mass on them).
* The alignment term pulls every layer's post-injection code toward the fixed
  codebook code of the token that position predicts, scaled by the detached
  backbone CE ``CE_bb = sum(w * CE) / sum(w)`` and a weight ``beta_t`` that warms
  up linearly (:class:`DSpineSchedule`).
* A training curriculum replaces, per block and with a probability that decays
  to zero over training, the predecessor codes of the first few layers with the
  true previous tokens' codes.

One deliberate difference from SpecForge: ``CE_bb`` here is this micro-batch's
own value, whereas SpecForge all-reduces it over the data-parallel ranks and the
whole gradient-accumulation window (with a no-grad prepass and RNG replay). It
only sets the detached scale of the alignment regularizer. Reducing it over the
data-parallel ranks inside the forward would add a collective that a rank whose
micro-batch has no valid anchor never reaches (it raises before the loss), while
the others wait in it; the recipe only synchronizes that skip after the forward.
Matching SpecForge exactly would need its prepass over the whole window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.speculative.dflash.core import (
    DFlashTrainerModule,
    _to_full_tensor,
    compute_acceptance_stats,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3_dspine import Qwen3DSpineDraftModel


@dataclass(frozen=True)
class DSpineSchedule:
    """Step-dependent DSpine weights: the alignment warmup and the teacher-forcing curriculum.

    Attributes:
        alignment_weight: Final weight ``beta`` of the alignment term.
        alignment_warmup_steps: Optimizer steps over which ``beta_t`` ramps
            linearly from 0 to ``alignment_weight``; ``0`` applies it from step 0.
        teacher_forcing_probability: Per-block teacher-forcing probability at the
            start of training.
        teacher_forcing_hold_fraction: Training fraction for which that
            probability is held.
        teacher_forcing_zero_fraction: Training fraction at which it has decayed
            linearly to 0; it stays 0 afterwards.
        total_steps: Total optimizer steps of the run.
    """

    alignment_weight: float
    alignment_warmup_steps: int
    teacher_forcing_probability: float
    teacher_forcing_hold_fraction: float
    teacher_forcing_zero_fraction: float
    total_steps: int

    def __post_init__(self) -> None:
        if self.alignment_weight < 0:
            raise ValueError(f"alignment_weight must be >= 0, got {self.alignment_weight}.")
        if self.alignment_warmup_steps < 0:
            raise ValueError(f"alignment_warmup_steps must be >= 0, got {self.alignment_warmup_steps}.")
        if not 0 <= self.teacher_forcing_probability <= 1:
            raise ValueError(f"teacher_forcing_probability must be in [0, 1], got {self.teacher_forcing_probability}.")
        if not 0 <= self.teacher_forcing_hold_fraction < self.teacher_forcing_zero_fraction <= 1:
            raise ValueError(
                "teacher forcing needs 0 <= teacher_forcing_hold_fraction < teacher_forcing_zero_fraction <= 1, got "
                f"{self.teacher_forcing_hold_fraction} and {self.teacher_forcing_zero_fraction}."
            )
        if self.total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {self.total_steps}.")

    def alignment_beta(self, step: int) -> float:
        """Return the alignment weight ``beta_t`` at optimizer step ``step``."""
        if self.alignment_warmup_steps == 0:
            return self.alignment_weight
        return self.alignment_weight * min(max(step, 0) / self.alignment_warmup_steps, 1.0)

    def teacher_forcing(self, step: int) -> float:
        """Return the per-block teacher-forcing probability at optimizer step ``step``."""
        progress = max(step, 0) / self.total_steps
        fraction = (self.teacher_forcing_zero_fraction - progress) / (
            self.teacher_forcing_zero_fraction - self.teacher_forcing_hold_fraction
        )
        return self.teacher_forcing_probability * min(max(fraction, 0.0), 1.0)


@dataclass
class DSpineStepMetrics:
    """Per-step training outputs for the DSpine draft.

    ``loss`` / ``loss_weight`` / ``accuracy`` / ``valid_tokens`` /
    ``correct_tokens`` / ``accept_len*`` / ``valid_blocks`` mirror
    ``DFlashStepMetrics`` so the shared DFlash training loop consumes them
    unchanged. The primary accuracy and acceptance-length fields describe the
    *refined* path -- what DSpine emits at decode time -- and the ``base_*``
    fields the backbone's own top-1 picks, on the same denominator. Because the
    refinement is teacher-forced on the true previous token, the refined
    acceptance length equals the serial decode walk's: the walk only departs from
    the teacher-forced path after its first mistake, which ends the accepted
    prefix anyway.

    Attributes:
        loss: Scalar tensor containing the differentiable training loss (the
            objective's numerator divided by the micro-batch's sequence count).
        loss_weight: Scalar tensor containing that sequence count.
        accuracy: Scalar tensor containing refined-path greedy token accuracy.
        valid_tokens: Scalar tensor containing the supervised-token count.
        correct_tokens: Scalar tensor containing the refined-path correct-token count.
        accept_len: Scalar tensor containing refined-path mean acceptance length.
        accept_len_sum: Scalar tensor containing refined-path additive acceptance length.
        valid_blocks: Scalar tensor containing the number of evaluated draft blocks.
        weight_sum: Scalar tensor containing the decay-weight sum ``sum(w)``.
        backbone_ce_sum: Scalar tensor containing ``sum(w * CE)`` of the backbone.
        backbone_l1_sum: Scalar tensor containing ``sum(w * L1)`` of the backbone.
        refine_ce: Scalar tensor containing the candidate-weighted refinement CE.
        refine_ce_weight: Scalar tensor containing its weight (``sum(w)`` over
            positions whose label is a candidate).
        refine_l1: Scalar tensor containing the candidate-weighted refinement L1.
        refine_l1_weight: Scalar tensor containing its weight (``sum(w)`` over
            positions whose target mass on the candidates passes the threshold).
        alignment_sum: Scalar tensor containing the layer-averaged ``sum(w * (1 - cos))``.
        candidate_hits: Scalar tensor containing the supervised positions whose
            label is among the backbone's top-``head_topk`` candidates.
        base_correct_tokens: Scalar tensor containing backbone correct-token count.
        base_accept_len: Scalar tensor containing backbone mean acceptance length.
        base_accept_len_sum: Scalar tensor containing backbone additive acceptance length.
    """

    loss: torch.Tensor
    loss_weight: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    correct_tokens: torch.Tensor
    accept_len: torch.Tensor
    accept_len_sum: torch.Tensor
    valid_blocks: torch.Tensor
    weight_sum: torch.Tensor
    backbone_ce_sum: torch.Tensor
    backbone_l1_sum: torch.Tensor
    refine_ce: torch.Tensor
    refine_ce_weight: torch.Tensor
    refine_l1: torch.Tensor
    refine_l1_weight: torch.Tensor
    alignment_sum: torch.Tensor
    candidate_hits: torch.Tensor
    base_correct_tokens: torch.Tensor
    base_accept_len: torch.Tensor
    base_accept_len_sum: torch.Tensor


def _teacher_l1(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """L1 distance between the softmax distributions of two logit tensors, in float32.

    Args:
        student_logits: Tensor of shape [..., vocab]; differentiable.
        teacher_logits: Tensor of shape [..., vocab]; treated as a constant.

    Returns:
        Tensor of shape [...].
    """
    teacher = teacher_logits.detach().float().softmax(dim=-1)
    return (student_logits.float().softmax(dim=-1) - teacher).abs().sum(dim=-1)


class DSpineTrainerModule(DFlashTrainerModule):
    """DSpine online training wrapper: backbone CE + L1, refinement, and code alignment."""

    def __init__(
        self,
        draft_model: Qwen3DSpineDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: float | None = 7.0,
        sliding_window: int | None = None,
        *,
        distribution_ce_weight: float = 0.1,
        candidate_mass_threshold: float = 1e-4,
        objective_chunk_blocks: int = 128,
    ):
        if not isinstance(draft_model, Qwen3DSpineDraftModel):
            raise TypeError(f"DSpineTrainerModule requires a Qwen3DSpineDraftModel, got {type(draft_model).__name__}.")
        if not draft_model.transfer_space_ready:
            raise ValueError(
                "The DSpine draft's codebook and readouts are uninitialized: call "
                "draft_model.initialize_transfer_space(target_head_weight) or load a trained checkpoint first."
            )
        if block_size != draft_model.block_size:
            raise ValueError(
                f"DSpine training block_size={block_size} must match the draft's block_size={draft_model.block_size}."
            )
        if not 0 <= distribution_ce_weight <= 1:
            raise ValueError(f"distribution_ce_weight must be in [0, 1], got {distribution_ce_weight}.")
        if not 0 <= candidate_mass_threshold < 1:
            raise ValueError(f"candidate_mass_threshold must be in [0, 1), got {candidate_mass_threshold}.")
        if objective_chunk_blocks < 1:
            raise ValueError(f"objective_chunk_blocks must be >= 1, got {objective_chunk_blocks}.")
        if getattr(target_lm_head, "bias", None) is not None:
            raise ValueError(
                "DSpine scores candidates with raw head rows; a target lm_head with a bias is unsupported."
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
            sliding_window=sliding_window,
        )
        self.distribution_ce_weight = float(distribution_ce_weight)
        self.candidate_mass_threshold = float(candidate_mass_threshold)
        self.objective_chunk_blocks = int(objective_chunk_blocks)

    def _objective_chunk(
        self,
        hidden: torch.Tensor,
        pre_injection: torch.Tensor,
        layer_features: torch.Tensor,
        labels: torch.Tensor,
        predecessors: torch.Tensor,
        teacher_rows: torch.Tensor,
        weights: torch.Tensor,
        supervised: torch.Tensor,
        target_logits: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Compute the objective's sums over one chunk of draft blocks.

        Args:
            hidden: Tensor of shape [blocks, depth, hidden]; final-normed draft
                hidden states of the predicted positions (``depth = block_size - 1``).
            pre_injection: Tensor of shape [blocks, depth, hidden]; their last-layer
                outputs before the last injection.
            layer_features: Tensor of shape [blocks, layers, depth, code]; their
                post-injection codes at every layer.
            labels: Long tensor of shape [blocks, depth]; the true token each
                position predicts.
            predecessors: Long tensor of shape [blocks, depth]; the true previous
                token (the anchor for the first position).
            teacher_rows: Long tensor of shape [blocks, depth]; each position's row
                in the flattened ``target_logits``.
            weights: Tensor of shape [blocks, depth]; the decay weights, zero on
                unsupervised positions.
            supervised: Tensor of shape [blocks, depth]; 0/1 supervision mask.
            target_logits: Tensor of shape [batch * sequence, vocab]; the frozen
                target's logits, flattened over batch and sequence.
            head_weight: Tensor of shape [vocab, hidden]; the frozen target head.

        Returns:
            Tuple of scalar tensors ``(ce_sum, l1_sum, refine_ce_sum,
            refine_ce_weight, refine_l1_sum, refine_l1_weight, alignment_sum,
            candidate_hits)`` followed by two long tensors of shape [blocks, depth]:
            the backbone's and the refined path's greedy picks.
        """
        vocab = head_weight.shape[0]
        logits = F.linear(hidden, head_weight).float()
        with torch.no_grad():
            teacher = target_logits[teacher_rows].float()
        ce = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none").view_as(labels)
        l1 = _teacher_l1(logits, teacher)

        top_k = self.draft_model.head_topk
        candidates = logits.detach().topk(top_k, dim=-1).indices
        refined = self.draft_model.refine(pre_injection, predecessors)
        scores = torch.einsum("bdh,bdkh->bdk", refined, F.embedding(candidates, head_weight)).float()
        covered = candidates == labels.unsqueeze(-1)
        hit = covered.any(dim=-1)
        refine_ce = F.cross_entropy(
            scores.reshape(-1, top_k), covered.long().argmax(dim=-1).reshape(-1), reduction="none"
        ).view_as(labels)
        teacher_candidates = teacher.gather(-1, candidates)
        with torch.no_grad():
            # The eligibility gate reads the target's full-vocabulary mass on the
            # candidates, before they are renormalized among themselves.
            eligible = teacher.softmax(dim=-1).gather(-1, candidates).sum(dim=-1) > self.candidate_mass_threshold
        refine_l1 = _teacher_l1(scores, teacher_candidates)

        codebook = self.draft_model.selected_prefix_bridge.codebook
        reference = F.embedding(labels, codebook).float().unsqueeze(1)
        alignment = 1.0 - F.cosine_similarity(layer_features.float(), reference, dim=-1)

        with torch.no_grad():
            hit_weight = (hit * weights).sum()
            eligible_weight = (eligible * weights).sum()
            candidate_hits = (hit * supervised).sum()
            base_ids = logits.argmax(dim=-1)
            refined_ids = candidates.gather(-1, scores.argmax(dim=-1, keepdim=True)).squeeze(-1)
        return (
            (ce * weights).sum(),
            (l1 * weights).sum(),
            (refine_ce * hit * weights).sum(),
            hit_weight,
            (refine_l1 * eligible * weights).sum(),
            eligible_weight,
            (alignment * weights.unsqueeze(1)).sum() / layer_features.shape[1],
            candidate_hits,
            base_ids,
            refined_ids,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_logits: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
        *,
        alignment_beta: float = 0.0,
        teacher_forcing: float = 0.0,
    ) -> DSpineStepMetrics:
        """Parallel block-wise DSpine training forward.

        Sequence packing (``position_ids`` / ``seq_lens`` / ``doc_remaining``) is
        handled by the shared DFlash prologue, which keeps every block inside one
        document; the in-block mask is causal.

        Args:
            input_ids: Long tensor of shape [batch, sequence]; the context tokens,
                also the supervision labels.
            hidden_states: Tensor of shape [batch, sequence, layers * hidden]; the
                captured target-model context features.
            loss_mask: Tensor of shape [batch, sequence]; the supervised-token mask.
            target_logits: Tensor of shape [batch, sequence, vocab]; the frozen
                target's logits, where position ``t`` predicts token ``t + 1``.
                May be vocab-sharded (DTensor) under tensor parallelism.
            position_ids: Long tensor of shape [batch, sequence] with per-document
                reset positions under packing, or ``None``.
            seq_lens: Long tensor of shape [batch, max_docs] with packed document
                lengths, or ``None`` when unpacked.
            doc_remaining: Long tensor of shape [batch, sequence]; real tokens left
                in each position's document, or ``None`` when unpacked.
            alignment_beta: Current alignment weight ``beta_t``
                (:meth:`DSpineSchedule.alignment_beta`).
            teacher_forcing: Current per-block teacher-forcing probability
                (:meth:`DSpineSchedule.teacher_forcing`); only applied in training mode.

        Returns:
            DSpineStepMetrics for this micro-batch.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchors, block_keep_mask, noise_embedding, full_position_ids, attn_mask, _ = self._prepare_block_inputs(
            input_ids,
            loss_mask,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
            causal=True,
        )
        label_indices, block_ids, block_mask = self._build_block_targets(
            input_ids, loss_mask, anchors, block_keep_mask, seq_len, doc_remaining=doc_remaining
        )
        replace_mask = None
        if self.training and teacher_forcing > 0:
            replace_mask = (torch.rand(anchors.shape, device=device) < teacher_forcing) & block_keep_mask

        output = self.draft_model(
            position_ids=full_position_ids,
            attention_mask=attn_mask,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            block_token_ids=block_ids,
            replace_mask=replace_mask,
        )

        n, bs = anchors.size(1), self.block_size
        depth = bs - 1
        # Supervise each block's contiguous supervised prefix only (SpecForge's
        # cumprod): a position past the first unsupervised token is never reached
        # by an accepted draft.
        supervised = block_mask[:, :, 1:].cumprod(dim=-1)
        weights = supervised
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            decay = torch.exp(-torch.arange(depth, device=device, dtype=torch.float32) / self.loss_decay_gamma)
            weights = supervised * decay
        # Position k is predicted by the target's logits at anchor + k - 1.
        teacher_pos = label_indices[:, :, :-1].clamp(max=seq_len - 1)
        teacher_rows = teacher_pos + (torch.arange(bsz, device=device) * seq_len).view(-1, 1, 1)

        num_blocks = bsz * n
        hidden = output.hidden_states.view(bsz, n, bs, -1)[:, :, 1:].reshape(num_blocks, depth, -1)
        pre_injection = output.pre_injection.view(bsz, n, bs, -1)[:, :, 1:].reshape(num_blocks, depth, -1)
        num_layers = output.layer_features.shape[0]
        layer_features = (
            output.layer_features.view(num_layers, bsz, n, bs, -1)[:, :, :, 1:]
            .permute(1, 2, 0, 3, 4)
            .reshape(num_blocks, num_layers, depth, -1)
        )
        chunk_args = (
            hidden,
            pre_injection,
            layer_features,
            block_ids[:, :, 1:].reshape(num_blocks, depth),
            block_ids[:, :, :-1].reshape(num_blocks, depth),
            teacher_rows.reshape(num_blocks, depth),
            weights.reshape(num_blocks, depth),
            supervised.reshape(num_blocks, depth),
        )
        flat_target_logits = _to_full_tensor(target_logits).reshape(bsz * seq_len, -1)
        head_weight = _to_full_tensor(self.lm_head.weight)

        sums = None
        base_ids, refined_ids = [], []
        for start in range(0, num_blocks, self.objective_chunk_blocks):
            chunk = tuple(arg[start : start + self.objective_chunk_blocks] for arg in chunk_args)
            if torch.is_grad_enabled():
                # Recompute the chunk's full-vocabulary logits in backward instead of
                # keeping them for every block at once.
                result = checkpoint(self._objective_chunk, *chunk, flat_target_logits, head_weight, use_reentrant=False)
            else:
                result = self._objective_chunk(*chunk, flat_target_logits, head_weight)
            sums = result[:8] if sums is None else tuple(a + b for a, b in zip(sums, result[:8]))
            base_ids.append(result[8])
            refined_ids.append(result[9])
        ce_sum, l1_sum, refine_ce_sum, refine_ce_weight, refine_l1_sum, refine_l1_weight, alignment_sum, hits = sums

        alpha = self.distribution_ce_weight
        weight_sum = weights.sum()
        valid_tokens = supervised.sum()
        refine_ce = torch.where(
            refine_ce_weight > 0, refine_ce_sum / refine_ce_weight.clamp_min(1e-30), refine_ce_sum * 0.0
        )
        refine_l1 = torch.where(
            refine_l1_weight > 0, refine_l1_sum / refine_l1_weight.clamp_min(1e-30), refine_l1_sum * 0.0
        )
        refinement = (alpha * refine_ce + (1.0 - alpha) * refine_l1) * valid_tokens
        alignment_scale = alignment_beta * (ce_sum.detach() / weight_sum.clamp_min(1e-30))
        numerator = alpha * ce_sum + (1.0 - alpha) * l1_sum + refinement + alignment_scale * alignment_sum
        sequences = torch.tensor(float(bsz), device=device)
        loss = numerator / sequences

        with torch.no_grad():
            eval_mask = supervised.bool()
            labels = block_ids[:, :, 1:]
            base = torch.cat(base_ids).view(bsz, n, depth)
            refined = torch.cat(refined_ids).view(bsz, n, depth)
            correct_tokens = ((refined == labels) & eval_mask).sum()
            accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(refined, labels, eval_mask)
            base_correct_tokens = ((base == labels) & eval_mask).sum()
            base_accept_len, base_accept_len_sum, _ = compute_acceptance_stats(base, labels, eval_mask)

        return DSpineStepMetrics(
            loss=loss,
            loss_weight=sequences,
            accuracy=(correct_tokens / valid_tokens.clamp_min(1)).detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct_tokens.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
            weight_sum=weight_sum.detach(),
            backbone_ce_sum=ce_sum.detach(),
            backbone_l1_sum=l1_sum.detach(),
            refine_ce=refine_ce.detach(),
            refine_ce_weight=refine_ce_weight.detach(),
            refine_l1=refine_l1.detach(),
            refine_l1_weight=refine_l1_weight.detach(),
            alignment_sum=alignment_sum.detach(),
            candidate_hits=hits.detach(),
            base_correct_tokens=base_correct_tokens.detach(),
            base_accept_len=base_accept_len.detach(),
            base_accept_len_sum=base_accept_len_sum.detach(),
        )
