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
        cur_loss_mask = loss_mask
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
                cur_loss_mask = _shift_left_with_zero(cur_loss_mask)
                cur_position_mask = _shift_left_with_zero(cur_position_mask)
                cur_target_probs = _shift_left_with_zero(cur_target_probs)

        avg_loss = running_loss / weight_sum
        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=avg_loss, accuracy=accuracy, valid_tokens=running_valid)
