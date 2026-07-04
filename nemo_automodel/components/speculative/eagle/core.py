# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Core EAGLE-3 training logic for the minimal Llama MVP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail."""
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


def _cp_shift_left(tensor: torch.Tensor, cp_group) -> torch.Tensor:
    """Left-shift a context-parallel-sharded sequence tensor by one position.

    Each rank holds a contiguous ``S/cp`` shard. A plain left-shift would zero-fill
    the boundary, but the token that rolls into rank ``r``'s tail lives at rank
    ``r+1``'s head. Shift locally, then P2P the neighbour's (current) head into the
    tail; the last rank keeps the zero fill. Applied in lockstep each TTT step this
    reproduces the global shift exactly. Used for labels / input_ids only (no grad).
    """
    shifted = torch.cat((tensor[:, 1:], torch.zeros_like(tensor[:, :1])), dim=1)
    world = dist.get_world_size(cp_group)
    if world == 1:
        return shifted
    rank = dist.get_rank(cp_group)
    head = tensor[:, :1].contiguous()
    recv = torch.empty_like(head)
    send_to = dist.get_global_rank(cp_group, (rank - 1) % world)  # my head -> rank-1
    recv_from = dist.get_global_rank(cp_group, (rank + 1) % world)  # rank+1's head -> my tail
    for req in dist.batch_isend_irecv(
        [dist.P2POp(dist.isend, head, send_to, group=cp_group), dist.P2POp(dist.irecv, recv, recv_from, group=cp_group)]
    ):
        req.wait()
    if rank != world - 1:  # the last rank has no successor -> keep the zero fill
        shifted = torch.cat((shifted[:, :-1], recv), dim=1)
    return shifted


def _cp_shift_left_zigzag(tensor: torch.Tensor, cp_group) -> torch.Tensor:
    """Left-shift a ZIG-ZAG-sharded sequence tensor by one position.

    Under zig-zag sharding rank ``r`` holds two non-contiguous global chunks --
    the early chunk ``r`` and the late chunk ``2*cp-1-r`` -- laid out locally as
    ``[early | late]`` (each of length ``c = local_len/2``). A global left-shift
    therefore rolls each half locally and needs two boundary tokens:

    * the early half's tail (global end of chunk ``r``) is the head of chunk
      ``r+1`` -- rank ``r+1``'s early head -- except on the last rank, where chunk
      ``r+1`` is that rank's OWN late chunk, so the fill is local.
    * the late half's tail (global end of chunk ``2*cp-1-r``) is the head of chunk
      ``2*cp-r`` -- rank ``r-1``'s late head -- except on rank 0, whose late tail is
      the global last position and stays zero-filled.

    Two ring P2P exchanges (early head -> ``r-1``, late head -> ``r+1``) cover both.
    Used for labels / input_ids only (no grad). Applied in lockstep each TTT step
    this reproduces the global shift exactly. See :func:`_cp_shift_left` for the
    contiguous-shard analogue.
    """
    c = tensor.shape[1] // 2
    early, late = tensor[:, :c], tensor[:, c:]
    zero = torch.zeros_like(tensor[:, :1])
    early_shift = torch.cat((early[:, 1:], zero), dim=1)
    late_shift = torch.cat((late[:, 1:], zero.clone()), dim=1)

    world = dist.get_world_size(cp_group)
    if world == 1:
        # Rank 0 owns [chunk 0 | chunk 1] = the whole sequence in order: the early
        # tail is the late head, the late tail is the global end (zero).
        early_shift = torch.cat((early_shift[:, :-1], late[:, :1]), dim=1)
        return torch.cat((early_shift, late_shift), dim=1)

    rank = dist.get_rank(cp_group)
    early_head = early[:, :1].contiguous()
    late_head = late[:, :1].contiguous()
    recv_early = torch.empty_like(early_head)  # rank+1's early head -> my early tail
    recv_late = torch.empty_like(late_head)  # rank-1's late head -> my late tail
    prev = dist.get_global_rank(cp_group, (rank - 1) % world)
    nxt = dist.get_global_rank(cp_group, (rank + 1) % world)
    for req in dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, early_head, prev, group=cp_group),
            dist.P2POp(dist.irecv, recv_early, nxt, group=cp_group),
            dist.P2POp(dist.isend, late_head, nxt, group=cp_group),
            dist.P2POp(dist.irecv, recv_late, prev, group=cp_group),
        ]
    ):
        req.wait()

    early_tail = recv_early if rank < world - 1 else late[:, :1]
    early_shift = torch.cat((early_shift[:, :-1], early_tail), dim=1)
    if rank >= 1:  # rank 0's late tail is the global end -> keep the zero fill
        late_shift = torch.cat((late_shift[:, :-1], recv_late), dim=1)
    return torch.cat((early_shift, late_shift), dim=1)


class _CpAllReduceSum(torch.autograd.Function):
    """Differentiable sum-all-reduce across the cp group.

    Forward sums per-rank inputs into a replicated total. Each rank uses that total
    identically, so the loss gradient w.r.t. this rank's input is just the incoming
    grad (coefficient 1) -- hence the identity backward.
    """

    @staticmethod
    def forward(ctx, x, cp_group):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=cp_group)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _cp_global_step_loss(step_loss: torch.Tensor, position_mask: torch.Tensor, cp_group) -> torch.Tensor:
    """Renormalize a per-shard masked-mean loss over the full cp sequence.

    ``step_loss`` is the mean over this rank's LOCAL supervised positions. Recover the
    local sum (mean * count), sum both across cp (the sum differentiably), and divide
    by the global count -- so the loss value equals the full-sequence loss and the
    backprop'd gradient is the global gradient.
    """
    count = position_mask.sum()
    local_sum = step_loss * count
    total_sum = _CpAllReduceSum.apply(local_sum, cp_group)
    total_count = count.detach().clone()
    dist.all_reduce(total_count, op=dist.ReduceOp.SUM, group=cp_group)
    return total_sum / total_count.clamp_min(1.0)


def _compute_target_distribution(
    target_logits: torch.Tensor,
    selected_token_ids: torch.Tensor,
    selected_token_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project target logits into draft vocabulary space and build supervision mask."""
    target_top_ids = target_logits.argmax(dim=-1)
    position_mask = (selected_token_mask[target_top_ids] & loss_mask.bool()).unsqueeze(-1)
    draft_target_logits = target_logits.index_select(dim=-1, index=selected_token_ids.to(target_logits.device))
    target_probs = torch.softmax(draft_target_logits.float(), dim=-1).detach()
    return target_probs, position_mask


@dataclass
class Eagle3StepMetrics:
    """Aggregated metrics from one EAGLE-3 training step."""

    loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor


class Eagle3TrainerModule(nn.Module):
    """Draft-side EAGLE-3 trainer module with test-time-training unroll."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        selected_token_ids: torch.Tensor,
        selected_token_mask: torch.Tensor,
        ttt_steps: int,
        cp_group=None,
        cp_zigzag: bool = False,
    ):
        super().__init__()
        # The forward pass weighs each TTT step by ``0.8 ** i`` and divides
        # the running loss by ``sum_{i=0}^{ttt_steps-1} 0.8 ** i``. With
        # ``ttt_steps <= 0`` the loop never runs and the divisor is zero,
        # which would silently produce a NaN loss instead of an actionable
        # error. Catch the misconfiguration here so it surfaces during
        # recipe setup rather than mid-training.
        if not isinstance(ttt_steps, int) or ttt_steps < 1:
            raise ValueError(
                f"Eagle3TrainerModule requires ttt_steps to be an integer >= 1 "
                f"(the draft must run at least one forward step to produce a "
                f"loss), got ttt_steps={ttt_steps!r}."
            )
        self.draft_model = draft_model
        self.register_buffer("selected_token_ids", selected_token_ids, persistent=True)
        self.register_buffer("selected_token_mask", selected_token_mask, persistent=True)
        self.ttt_steps = ttt_steps
        # cp process group when the draft runs sequence-sharded under context
        # parallelism; the per-step left-shift then rolls tokens across rank
        # boundaries via P2P instead of zero-filling. None for the single-rank path.
        self.cp_group = cp_group
        # Whether the cp sharding is the load-balanced zig-zag layout (selects the
        # matching two-neighbour boundary shift) vs the contiguous layout.
        self.cp_zigzag = cp_zigzag

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor | None = None,
        *,
        target_probs: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> Eagle3StepMetrics:
        """Run the EAGLE-3 unrolled draft loss for one batch.

        The attention layer is driven through a shared ``cache_hidden``
        list so each TTT step can attend to the K/V branches produced by
        every previous step at the same position. This matches the
        SpecForge ``llama3_eagle.py`` recurrence; without it, multi-step
        TTT would degenerate into ``ttt_steps`` independent single-step
        passes and the draft would never learn the multi-step
        distribution it sees at deployment time.

        ``attention_mask`` is held constant across TTT steps -- only
        ``input_ids`` / ``loss_mask`` / ``position_mask`` /
        ``target_probs`` roll forward by one position per step.

        Packing: ``position_ids`` / ``seq_lens`` make the draft's Block-1 attention
        document-level block-causal, and ``doc_remaining`` gates supervision per
        step (slot ``t`` valid at step ``k`` only while ``k < doc_remaining[t]``),
        masking every cross-document TTT prediction.

        Two supervision sources are accepted: the live path passes the
        target's full-vocab ``target_logits`` and the draft distribution is
        derived here; the offline-cache path (``precompute_eagle3``) passes the
        already-derived ``target_probs`` (over the draft vocab) and
        ``position_mask`` directly, so the full-vocab logits never have to be
        stored. Provide exactly one of the two.
        """
        precomputed = target_probs is not None and position_mask is not None
        if target_logits is not None and precomputed:
            raise ValueError(
                "Eagle3TrainerModule.forward got both target_logits and precomputed "
                "(target_probs, position_mask); pass exactly one supervision source."
            )
        hidden_states = self.draft_model.project_hidden_states(aux_hidden_states)
        if not precomputed:
            if target_logits is None:
                raise ValueError(
                    "Eagle3TrainerModule.forward requires either target_logits (live path) or both "
                    "target_probs and position_mask (offline-cache path); got neither."
                )
            target_probs, position_mask = _compute_target_distribution(
                target_logits=target_logits,
                selected_token_ids=self.selected_token_ids,
                selected_token_mask=self.selected_token_mask,
                loss_mask=loss_mask,
            )

        running_loss = hidden_states.new_zeros(())
        running_correct = hidden_states.new_zeros(())
        running_valid = hidden_states.new_zeros(())

        cur_input_ids = input_ids
        cur_position_mask = position_mask
        cur_target_probs = target_probs
        cur_hidden_states = hidden_states

        # EAGLE-3 TTT KV cache: a pair of lists [K_list, V_list] that the
        # attention layer appends to on every step. Re-created per batch.
        cache_hidden: list[list[torch.Tensor]] = [[], []]

        # Weighted average across TTT steps: step ``i`` is weighted by
        # ``0.8 ** i`` and the sum is divided by the total weight. This
        # keeps the EAGLE-3 / SpecForge decay schedule (earlier steps
        # dominate, later steps still contribute a smaller signal) while
        # making the loss magnitude *invariant* to the choice of
        # ``ttt_steps`` and the decay constant -- a proper weighted mean
        # always lands in the same ``~ln(draft_vocab_size)`` range at
        # init, and the optimizer LR does not need to be rescaled when
        # the TTT schedule changes. SpecForge omits this normalization;
        # we keep it deliberately so config knobs stay decoupled from LR.
        weight_sum = sum(0.8**i for i in range(self.ttt_steps))
        for step_idx in range(self.ttt_steps):
            cur_hidden_states = self.draft_model(
                input_ids=cur_input_ids,
                projected_hidden_states=cur_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                seq_lens=seq_lens,
            )
            logits = self.draft_model.compute_logits(cur_hidden_states)

            # Packing: drop supervision whose step_idx-ahead target crosses this
            # slot's document boundary. Gate recomputed per step (depends on step_idx).
            step_position_mask = cur_position_mask
            if doc_remaining is not None:
                in_doc = (step_idx < doc_remaining).unsqueeze(-1)
                step_position_mask = cur_position_mask & in_doc

            step_loss = masked_soft_cross_entropy(
                logits=logits,
                target_probs=cur_target_probs,
                position_mask=step_position_mask,
            )
            # Under CP each rank sees only its sequence shard; renormalize the step
            # loss over the full cp sequence so the value and gradient match the
            # single-GPU run (the draft grads are then summed across cp by the recipe).
            if self.cp_group is not None:
                step_loss = _cp_global_step_loss(step_loss, step_position_mask, self.cp_group)
            running_loss = running_loss + step_loss * (0.8**step_idx)

            valid_mask = step_position_mask.squeeze(-1).bool()
            correct = (logits.argmax(dim=-1) == cur_target_probs.argmax(dim=-1)) & valid_mask
            running_correct = running_correct + correct.sum()
            running_valid = running_valid + valid_mask.sum()

            if step_idx + 1 < self.ttt_steps:
                if self.cp_group is None:
                    shift = _shift_left_with_zero
                elif self.cp_zigzag:
                    shift = lambda t: _cp_shift_left_zigzag(t, self.cp_group)  # noqa: E731
                else:
                    shift = lambda t: _cp_shift_left(t, self.cp_group)  # noqa: E731
                cur_input_ids = shift(cur_input_ids)
                cur_position_mask = shift(cur_position_mask)
                cur_target_probs = shift(cur_target_probs)

        avg_loss = running_loss / weight_sum
        if self.cp_group is not None:
            # Sum the accuracy counters over the cp shards for a full-sequence metric.
            running_correct = running_correct.clone()
            running_valid = running_valid.clone()
            dist.all_reduce(running_correct, op=dist.ReduceOp.SUM, group=self.cp_group)
            dist.all_reduce(running_valid, op=dist.ReduceOp.SUM, group=self.cp_group)
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=avg_loss, accuracy=accuracy, valid_tokens=running_valid)
