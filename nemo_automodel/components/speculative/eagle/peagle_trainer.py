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

"""P-EAGLE (parallel-drafting EAGLE-3) training logic.

Split out of ``core.py`` so the P-EAGLE trainer evolves independently of the
EAGLE-3 test-time-training trainer. The shared step-metrics container
(:class:`Eagle3StepMetrics`) still lives in ``core.py`` and is imported here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.core import Eagle3StepMetrics


def _kl_div_loss(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Per-position KL(target || draft) over the draft vocabulary.

    Matches speculators' ``kl_div_loss``: ``log_softmax`` the draft logits,
    ``softmax`` the target logits, and sum the elementwise KL over the vocab
    axis. Shapes ``[*, draft_vocab]`` -> ``[*]``.
    """
    log_p = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    target_p = torch.nn.functional.softmax(target_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(log_p, target_p, reduction="none", log_target=False).sum(dim=-1)


class _PeaglePlan:
    """Sequence-partitioning plan: one ``unit`` per non-empty ``(row, segment)``.

    A plain (non-container) class on purpose: ``DistributedDataParallel`` only
    scatters tensors and built-in containers across its inputs, so passing this
    object through ``forward(peagle_segment=(plan, i))`` leaves its tensors intact
    (a ``dict``/``tuple`` of tensors would be sliced along dim 0 by DDP's scatter).
    """

    __slots__ = ("units", "total_den")

    def __init__(self, units: list[tuple], total_den: torch.Tensor):
        self.units = units
        self.total_den = total_den


class PEagleTrainerModule(nn.Module):
    """Draft-side P-EAGLE (parallel-drafting EAGLE-3) trainer module.

    Faithful port of speculators' P-EAGLE
    (https://github.com/vllm-project/speculators/pull/480): the draft predicts
    all ``num_depths`` tokens in a *single* parallel forward over a flat,
    COD-subsampled sequence -- it does NOT run EAGLE-3's autoregressive TTT
    recurrence.

    Per training sequence:

    1. **COD sampling** (:func:`generate_cod_sample_indices`) draws
       ``(anchor_pos, depth)``: depth 0 keeps every position, depth ``d`` keeps a
       geometrically decaying ``down_sample_ratio**d`` fraction. The reference
       position of each element is ``anchor_pos + depth``.
    2. **Flat input assembly.** All depths are concatenated into one
       ``[1, total_sampled]`` sequence. Depth-0 slots take the real token id and
       the ``fc``-projected target aux hidden state; depth >= 1 slots take the
       masked ``mask_token_id`` and the single learnable ``mask_hidden``
       placeholder (projected through the same ``fc``).
    3. **COD flex attention.** A single ``flex_attention`` forward with the
       :func:`create_peagle_mask_mod` block mask: each element attends to the
       causal depth-0 context of its document plus earlier-or-equal depths of
       its own rollout. This is exactly what vLLM's parallel-drafting runtime
       sees at inference.
    4. **Count-normalized KL loss.** ``KL(target || draft)`` over the draft vocab
       at every supervised sampled position, normalized by a single total token
       count -- deeper depths (fewer COD positions) naturally contribute less
       gradient. No ``0.8**d`` schedule.

    Batches with ``batch_size > 1`` are processed row-by-row (speculators is
    batch-size-1); per-row losses are accumulated with a shared denominator so
    the normalization stays count-based across the whole batch.

    **Sequence partitioning (``sequence_partitions > 1``).** The flat COD
    forward attends over ``n * sum(r**d)`` positions, so its peak attention /
    activation memory grows with the context length and OOMs on long sequences.
    P-EAGLE's Algorithm 1 (arXiv:2602.01469) splits each sequence into ``S``
    segments by dependency lineage (:func:`assign_cod_segments`) and runs a
    *separate* forward+backward per segment so only one segment's activations are
    resident at a time. The partition is exact: each segment additionally reads
    every depth-0 position up to its right boundary as key/value context (causal
    completion), so a segment's queries see exactly the key/value set they would
    in the single flat forward -- the gradients accumulated across segments equal
    the single-forward gradient.

    The split is *caller-driven* so the gradient sync stays correct under DDP:
    :meth:`build_peagle_plan` (no-grad) assigns COD elements to segments, then the
    recipe runs one ``forward(..., peagle_segment=(plan, i))`` per segment and
    owns the ``backward()``. Doing the per-segment backward here (inside a single
    ``forward``) would bypass ``DistributedDataParallel``'s reducer -- its grad
    all-reduce hooks only fire for backwards over the tensor ``DDP.forward``
    returns -- and silently desynchronize ranks. ``sequence_partitions == 1`` and
    eval take the single flat ``forward`` unchanged.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        selected_token_ids: torch.Tensor,
        selected_token_mask: torch.Tensor,
        num_depths: int,
        mask_token_id: int,
        down_sample_ratio: float = 0.7,
        down_sample_ratio_min: float = 0.2,
        sequence_partitions: int = 1,
    ):
        super().__init__()
        if not isinstance(num_depths, int) or num_depths < 1:
            raise ValueError(
                f"PEagleTrainerModule requires num_depths to be an integer >= 1 "
                f"(the draft must produce at least one token), got num_depths={num_depths!r}."
            )
        if not isinstance(sequence_partitions, int) or sequence_partitions < 1:
            raise ValueError(
                f"PEagleTrainerModule requires sequence_partitions to be an integer >= 1, "
                f"got sequence_partitions={sequence_partitions!r}."
            )
        if getattr(draft_model, "mask_hidden", None) is None:
            raise ValueError(
                "PEagleTrainerModule requires the draft model to expose a learnable 'mask_hidden' "
                "parameter; build the draft with config.parallel_drafting=True."
            )
        self.draft_model = draft_model
        self.register_buffer("selected_token_ids", selected_token_ids, persistent=True)
        self.register_buffer("selected_token_mask", selected_token_mask, persistent=True)
        self.num_depths = num_depths
        self.mask_token_id = int(mask_token_id)
        self.down_sample_ratio = float(down_sample_ratio)
        self.down_sample_ratio_min = float(down_sample_ratio_min)
        self.sequence_partitions = int(sequence_partitions)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        peagle_segment: tuple | None = None,
    ) -> Eagle3StepMetrics:
        """Run the P-EAGLE parallel-drafting loss for one batch.

        ``attention_mask`` supplies the per-row valid length so padded positions
        are excluded from attention (document mask) and from supervision.

        ``peagle_segment`` selects the sequence-partitioning path: when it is a
        ``(plan, index)`` pair (built by :meth:`build_peagle_plan`) this computes
        the loss for that one segment only -- the recipe calls this once per
        segment and owns the ``backward()`` so DDP's gradient sync stays correct.
        When ``None`` (``sequence_partitions == 1`` and eval) a single flat
        forward over the whole COD sequence returns a grad-carrying loss.
        """
        if peagle_segment is not None:
            return self._forward_peagle_segment(
                input_ids, attention_mask, aux_hidden_states, target_logits, peagle_segment
            )

        from nemo_automodel.components.speculative.eagle.peagle_data import generate_cod_sample_indices

        batch_size, seq_len = input_ids.shape
        ref_dtype = self.draft_model.masked_projected_hidden().dtype
        loss_num = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        loss_den = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        running_correct = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        running_valid = torch.zeros((), device=input_ids.device, dtype=torch.float32)

        for b in range(batch_size):
            row_loss_mask = loss_mask[b : b + 1].long()  # [1, seq_len]
            anchor_pos, depth = generate_cod_sample_indices(
                seq_length=seq_len,
                loss_mask=row_loss_mask,
                num_depths=self.num_depths,
                down_sample_ratio=self.down_sample_ratio,
                down_sample_ratio_min=self.down_sample_ratio_min,
            )
            orig_positions = anchor_pos + depth
            row_length = attention_mask[b].sum().clamp_min(1).reshape(1).to(orig_positions.device)
            num, den, correct, valid = self._peagle_position_loss(
                input_ids[b],
                aux_hidden_states[b : b + 1],
                target_logits[b],
                anchor_pos,
                depth,
                orig_positions,
                row_loss_mask[0, orig_positions].bool(),  # supervision = loss mask at all sampled positions
                row_length,
                seq_len,
            )
            loss_num = loss_num + num
            loss_den = loss_den + den
            running_correct = running_correct + correct
            running_valid = running_valid + valid

        avg_loss = loss_num / loss_den.clamp_min(1e-5)
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=avg_loss.to(ref_dtype), accuracy=accuracy, valid_tokens=running_valid)

    def _peagle_position_loss(
        self,
        input_ids_row: torch.Tensor,  # [seq_len]
        aux_row: torch.Tensor,  # [1, seq_len, num_aux * H]
        target_logits_row: torch.Tensor,  # [seq_len, target_vocab]
        anchor_pos: torch.Tensor,  # [n]
        depth: torch.Tensor,  # [n]
        orig_positions: torch.Tensor,  # [n]
        loss_positions: torch.Tensor,  # [n] bool -- positions to charge loss/accuracy on
        row_length: torch.Tensor,  # [1] valid document length
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draft forward + count-normalized KL for one row's COD elements.

        Shared by the single flat forward (one call per row, all sampled
        positions charged) and the partitioned forward (one call per segment,
        only the segment's owned/supervised positions charged -- the rest ride
        along as key/value context). Returns ``(loss_num, loss_den, correct,
        valid)`` as float scalars, where the loss is ``Σ KL`` over ``loss_positions``
        and ``loss_den`` is their count; the caller normalizes.
        """
        draft = self.draft_model
        is_depth0 = depth == 0
        mask_hidden_proj = draft.masked_projected_hidden()  # [1, H]

        # Flat input ids: real token at depth 0, masked token elsewhere.
        flat_ids = torch.where(
            is_depth0,
            input_ids_row[orig_positions],
            torch.full_like(orig_positions, self.mask_token_id),
        ).unsqueeze(0)
        # Flat projected hidden: real aux at depth 0, mask_hidden elsewhere. Only
        # the referenced positions are projected (cheap) and ``torch.where`` keeps
        # ``mask_hidden`` in the autograd graph at every call.
        real_proj = draft.project_hidden_states(aux_row[:, orig_positions])[0]  # [n, H]
        flat_hidden = torch.where(
            is_depth0.unsqueeze(-1),
            real_proj,
            mask_hidden_proj.expand(orig_positions.shape[0], -1),
        ).unsqueeze(0)

        block_mask = draft.build_peagle_block_mask(
            anchor_pos=anchor_pos, depth=depth, lengths=row_length, total_seq_len=seq_len
        )
        hidden = draft.forward_peagle(
            sampled_input_ids=flat_ids,
            sampled_projected_hidden=flat_hidden,
            position_ids=orig_positions.unsqueeze(0),
            block_mask=block_mask,
        )
        logits = draft.compute_logits(hidden)[0]  # [n, draft_vocab]

        # Gather target logits to the draft vocab. This index_select equals
        # speculators' draft-vocab ``verifier_lm_head`` (= target lm_head restricted
        # to the t2d rows). Supervision is the loss mask only -- speculators does
        # not drop positions whose full-vocab argmax falls outside the draft vocab,
        # so neither do we.
        target_sel = target_logits_row[orig_positions]  # [n, target_vocab]
        draft_target_logits = target_sel.index_select(dim=-1, index=self.selected_token_ids)  # [n, draft_vocab]

        elementwise = _kl_div_loss(logits, draft_target_logits)  # [n]
        mask_f = loss_positions.to(elementwise.dtype)
        loss_num = (elementwise * mask_f).sum()
        loss_den = mask_f.sum()
        correct = ((logits.argmax(dim=-1) == draft_target_logits.argmax(dim=-1)) & loss_positions).sum()
        return loss_num, loss_den, correct.to(loss_num.dtype), loss_den

    @torch.no_grad()
    def build_peagle_plan(self, loss_mask: torch.Tensor) -> "_PeaglePlan":
        """Assign COD elements to segments for the sequence-partitioning path.

        Samples COD once per row (the indices must be reused across the segment
        forwards), runs Algorithm 1 assignment (:func:`assign_cod_segments`) plus
        causal completion, and emits one ``unit`` per non-empty ``(row, segment)``
        as ``(b, anchor, depth, orig_positions, loss_positions)``. ``loss_positions``
        marks the segment's *owned* supervised elements (charged loss); the other
        elements are depth-0 causal-completion context (key/value only). The shared
        ``total_den`` is the batch's total supervised-token count, so each segment's
        ``loss / total_den`` sums to the single-forward loss.
        """
        from nemo_automodel.components.speculative.eagle.peagle_data import (
            assign_cod_segments,
            generate_cod_sample_indices,
        )

        num_segments = self.sequence_partitions
        batch_size, seq_len = loss_mask.shape
        units: list[tuple] = []
        total_valid = torch.zeros((), device=loss_mask.device, dtype=torch.long)
        for b in range(batch_size):
            row_loss_mask = loss_mask[b : b + 1].long()
            anchor_pos, depth = generate_cod_sample_indices(
                seq_length=seq_len,
                loss_mask=row_loss_mask,
                num_depths=self.num_depths,
                down_sample_ratio=self.down_sample_ratio,
                down_sample_ratio_min=self.down_sample_ratio_min,
            )
            orig_positions = anchor_pos + depth
            sampled_loss_mask = row_loss_mask[0, orig_positions].bool()
            seg_assign = assign_cod_segments(anchor_pos, depth, seq_len, num_segments)
            is_depth0 = depth == 0
            total_valid = total_valid + sampled_loss_mask.sum()
            for s in range(num_segments):
                assigned = seg_assign == s
                # Segment s holds its assigned elements plus the causal completion:
                # every depth-0 position up to its right boundary (``seg <= s``).
                in_seg = assigned | (is_depth0 & (seg_assign <= s))
                seg_idx = torch.where(in_seg)[0]
                if seg_idx.numel() == 0:
                    continue
                loss_positions = assigned[seg_idx] & sampled_loss_mask[seg_idx]
                units.append((b, anchor_pos[seg_idx], depth[seg_idx], orig_positions[seg_idx], loss_positions))
        total_den = total_valid.clamp_min(1).to(torch.float32)
        return _PeaglePlan(units, total_den)

    def _forward_peagle_segment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        peagle_segment: tuple,
    ) -> Eagle3StepMetrics:
        """Loss for one segment of a :meth:`build_peagle_plan` plan.

        The recipe drives this once per ``plan.units`` entry and back-propagates
        each result, so each segment owns a self-contained autograd graph that is
        freed before the next -- and the backward flows through ``DDP.forward`` so
        gradients all-reduce correctly. ``metrics.loss`` is the segment's share of
        the count-normalized batch loss (``loss / total_den``); summing it over the
        plan reproduces the single flat forward's loss.
        """
        plan, index = peagle_segment
        b, anchor_pos, depth, orig_positions, loss_positions = plan.units[index]
        seq_len = input_ids.shape[1]
        row_length = attention_mask[b].sum().clamp_min(1).reshape(1).to(orig_positions.device)
        num, _den, correct, valid = self._peagle_position_loss(
            input_ids[b],
            aux_hidden_states[b : b + 1],
            target_logits[b],
            anchor_pos,
            depth,
            orig_positions,
            loss_positions,
            row_length,
            seq_len,
        )
        loss = (num / plan.total_den).to(self.draft_model.masked_projected_hidden().dtype)
        accuracy = correct / valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=loss, accuracy=accuracy, valid_tokens=valid)
