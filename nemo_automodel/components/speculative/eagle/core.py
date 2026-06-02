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
import torch.nn as nn

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail."""
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
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
        """
        hidden_states = self.draft_model.project_hidden_states(aux_hidden_states)
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
                cache_hidden=cache_hidden,
            )
            logits = self.draft_model.compute_logits(cur_hidden_states)
            step_loss = masked_soft_cross_entropy(
                logits=logits,
                target_probs=cur_target_probs,
                position_mask=cur_position_mask,
            )
            running_loss = running_loss + step_loss * (0.8**step_idx)

            valid_mask = cur_position_mask.squeeze(-1).bool()
            correct = (logits.argmax(dim=-1) == cur_target_probs.argmax(dim=-1)) & valid_mask
            running_correct = running_correct + correct.sum()
            running_valid = running_valid + valid_mask.sum()

            if step_idx + 1 < self.ttt_steps:
                cur_input_ids = _shift_left_with_zero(cur_input_ids)
                cur_position_mask = _shift_left_with_zero(cur_position_mask)
                cur_target_probs = _shift_left_with_zero(cur_target_probs)

        avg_loss = running_loss / weight_sum
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=avg_loss, accuracy=accuracy, valid_tokens=running_valid)


def _kl_div_loss(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Per-position KL(target || draft) over the draft vocabulary.

    Matches speculators' ``kl_div_loss``: ``log_softmax`` the draft logits,
    ``softmax`` the target logits, and sum the elementwise KL over the vocab
    axis. Shapes ``[*, draft_vocab]`` -> ``[*]``.
    """
    log_p = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    target_p = torch.nn.functional.softmax(target_logits.float(), dim=-1)
    return torch.nn.functional.kl_div(log_p, target_p, reduction="none", log_target=False).sum(dim=-1)


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
    ):
        super().__init__()
        if not isinstance(num_depths, int) or num_depths < 1:
            raise ValueError(
                f"PEagleTrainerModule requires num_depths to be an integer >= 1 "
                f"(the draft must produce at least one token), got num_depths={num_depths!r}."
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Eagle3StepMetrics:
        """Run the P-EAGLE parallel-drafting loss for one batch.

        ``attention_mask`` supplies the per-row valid length so padded positions
        are excluded from attention (document mask) and from supervision.
        """
        from nemo_automodel.components.speculative.eagle.peagle_data import generate_cod_sample_indices

        draft = self.draft_model
        batch_size, seq_len = input_ids.shape
        mask_hidden_proj = draft.masked_projected_hidden()  # [1, H]

        loss_num = mask_hidden_proj.new_zeros(())
        loss_den = mask_hidden_proj.new_zeros(())
        running_correct = mask_hidden_proj.new_zeros(())
        running_valid = mask_hidden_proj.new_zeros(())

        for b in range(batch_size):
            row_loss_mask = loss_mask[b : b + 1].long()  # [1, seq_len]
            anchor_pos, depth = generate_cod_sample_indices(
                seq_length=seq_len,
                loss_mask=row_loss_mask,
                num_depths=self.num_depths,
                down_sample_ratio=self.down_sample_ratio,
                down_sample_ratio_min=self.down_sample_ratio_min,
            )
            orig_positions = anchor_pos + depth  # [total_sampled]
            is_depth0 = depth == 0  # [total_sampled]

            # Flat input ids: real token at depth 0, masked token elsewhere.
            sampled_ids = torch.where(
                is_depth0,
                input_ids[b, orig_positions],
                torch.full_like(orig_positions, self.mask_token_id),
            ).unsqueeze(0)  # [1, total_sampled]

            # Flat projected hidden: real aux at depth 0, mask_hidden elsewhere.
            real_proj = draft.project_hidden_states(aux_hidden_states[b : b + 1])[0]  # [seq_len, H]
            sampled_hidden = torch.where(
                is_depth0.unsqueeze(-1),
                real_proj[orig_positions],
                mask_hidden_proj.expand(orig_positions.shape[0], -1),
            ).unsqueeze(0)  # [1, total_sampled, H]

            position_ids = orig_positions.unsqueeze(0)  # [1, total_sampled]
            lengths = attention_mask[b].sum().clamp_min(1).reshape(1).to(orig_positions.device)
            block_mask = draft.build_peagle_block_mask(
                anchor_pos=anchor_pos, depth=depth, lengths=lengths, total_seq_len=seq_len
            )

            hidden = draft.forward_peagle(
                sampled_input_ids=sampled_ids,
                sampled_projected_hidden=sampled_hidden,
                position_ids=position_ids,
                block_mask=block_mask,
            )
            logits = draft.compute_logits(hidden)[0]  # [total_sampled, draft_vocab]

            # Project target logits to the draft vocab at the sampled positions
            # and drop positions whose target top-1 falls outside the draft vocab
            # (the shared draft-vocab-shrink rule, applied per sampled element).
            target_sel = target_logits[b, orig_positions]  # [total_sampled, target_vocab]
            target_top = target_sel.argmax(dim=-1)
            in_vocab = self.selected_token_mask[target_top]
            sampled_loss_mask = row_loss_mask[0, orig_positions].bool() & in_vocab  # [total_sampled]
            draft_target_logits = target_sel.index_select(
                dim=-1, index=self.selected_token_ids.to(target_sel.device)
            )  # [total_sampled, draft_vocab]

            elementwise = _kl_div_loss(logits, draft_target_logits)  # [total_sampled]
            mask_f = sampled_loss_mask.to(elementwise.dtype)
            loss_num = loss_num + (elementwise * mask_f).sum()
            loss_den = loss_den + mask_f.sum()

            correct = (logits.argmax(dim=-1) == draft_target_logits.argmax(dim=-1)) & sampled_loss_mask
            running_correct = running_correct + correct.sum()
            running_valid = running_valid + sampled_loss_mask.sum()

        avg_loss = loss_num / loss_den.clamp_min(1e-5)
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=avg_loss.to(mask_hidden_proj.dtype), accuracy=accuracy, valid_tokens=running_valid)
