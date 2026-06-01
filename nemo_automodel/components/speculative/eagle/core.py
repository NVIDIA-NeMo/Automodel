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


class PEagleTrainerModule(nn.Module):
    """Draft-side P-EAGLE (parallel-drafting EAGLE-3) trainer module.

    P-EAGLE replaces EAGLE-3's autoregressive test-time-training recurrence
    with *parallel* multi-token prediction: at every base position the draft
    predicts the next ``num_draft_tokens`` tokens in one shot, conditioning
    depths ``>= 1`` on fixed learnable placeholders instead of the previous
    step's own output. For depth ``d``:

    * depth 0 (next-token prediction) consumes the real token embedding and
      the projected target auxiliary hidden states -- identical to
      ``Eagle3TrainerModule`` step 0;
    * depths ``1..K-1`` (multi-token prediction) consume the *masked* token
      ``ptd_token_id`` and the single learnable ``mask_hidden`` placeholder
      (projected through ``model.fc``), with **no** recurrence -- the inputs
      are identical across all masked depths and only the supervision target
      rolls forward.

    The cross-depth attention is the same EAGLE-3 ``cache_hidden``
    diagonal-extension pattern: depth ``d`` attends to the causal real prefix
    (the ``T x T`` block over depth-0 keys) plus the same base position at
    every earlier depth, with RoPE phase ``position + d``. This is exactly
    what vLLM's parallel-drafting runtime sees at inference -- one base
    position is expanded per decode step and the KV cache holds committed
    depth-0 tokens only -- so training through ``cache_hidden`` is numerically
    faithful to deployment despite running K sequential forwards here (the
    single-forward parallelism is an inference-time property).

    The draft-vocab projection, ``0.8 ** d`` weighted-mean loss schedule, and
    accuracy metrics are shared verbatim with :class:`Eagle3TrainerModule`.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        selected_token_ids: torch.Tensor,
        selected_token_mask: torch.Tensor,
        num_draft_tokens: int,
        ptd_token_id: int,
    ):
        super().__init__()
        if not isinstance(num_draft_tokens, int) or num_draft_tokens < 1:
            raise ValueError(
                f"PEagleTrainerModule requires num_draft_tokens to be an integer >= 1 "
                f"(the draft must produce at least one token), got num_draft_tokens={num_draft_tokens!r}."
            )
        if getattr(draft_model, "mask_hidden", None) is None:
            raise ValueError(
                "PEagleTrainerModule requires the draft model to expose a learnable 'mask_hidden' "
                "parameter; build the draft with config.parallel_drafting=True."
            )
        self.draft_model = draft_model
        self.register_buffer("selected_token_ids", selected_token_ids, persistent=True)
        self.register_buffer("selected_token_mask", selected_token_mask, persistent=True)
        self.num_draft_tokens = num_draft_tokens
        self.ptd_token_id = int(ptd_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Eagle3StepMetrics:
        """Run the P-EAGLE parallel-drafting loss for one batch.

        ``attention_mask`` is held constant across depths -- only the
        supervision (``loss_mask`` / ``position_mask`` / ``target_probs``)
        rolls forward by one position per depth, mirroring
        ``Eagle3TrainerModule``. Unlike EAGLE-3 TTT, the per-depth *inputs* at
        depths ``>= 1`` are the fixed masked token / ``mask_hidden`` placeholder
        rather than the previous depth's output.
        """
        hidden_states = self.draft_model.project_hidden_states(aux_hidden_states)
        # Single projected ``mask_hidden`` placeholder, broadcast across the
        # batch / sequence and shared by every masked depth. Computed once: it
        # is independent of the base position, so all masked depths backprop
        # into the same parameter through this one projection. It shares the
        # ``project_hidden_states`` (``fc``) output dtype with ``hidden_states``,
        # so no cast is needed.
        mask_hidden = self.draft_model.masked_projected_hidden()
        masked_hidden_states = mask_hidden.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        masked_input_ids = torch.full_like(input_ids, self.ptd_token_id)

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
        cur_hidden_states = hidden_states
        cur_position_mask = position_mask
        cur_target_probs = target_probs

        # Shared EAGLE-3 TTT KV cache [K_list, V_list]; the attention layer
        # appends each depth's K/V so depth ``d`` attends to the same base
        # position at depths ``0..d-1``. Re-created per batch.
        cache_hidden: list[list[torch.Tensor]] = [[], []]

        weight_sum = sum(0.8**i for i in range(self.num_draft_tokens))
        for step_idx in range(self.num_draft_tokens):
            step_hidden = self.draft_model(
                input_ids=cur_input_ids,
                projected_hidden_states=cur_hidden_states,
                attention_mask=attention_mask,
                cache_hidden=cache_hidden,
            )
            logits = self.draft_model.compute_logits(step_hidden)
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

            if step_idx + 1 < self.num_draft_tokens:
                # Masked depths share fixed inputs (no recurrence): the
                # ``ptd_token_id`` token embedding and the projected
                # ``mask_hidden`` placeholder. Only supervision rolls forward.
                cur_input_ids = masked_input_ids
                cur_hidden_states = masked_hidden_states
                cur_position_mask = _shift_left_with_zero(cur_position_mask)
                cur_target_probs = _shift_left_with_zero(cur_target_probs)

        avg_loss = running_loss / weight_sum
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=avg_loss, accuracy=accuracy, valid_tokens=running_valid)
