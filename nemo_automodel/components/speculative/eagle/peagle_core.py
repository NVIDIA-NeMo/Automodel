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

"""Parallel-drafting trainer module for P-EAGLE (arXiv:2602.01469).

This is the parallel analogue of :class:`Eagle3TrainerModule`. Where EAGLE-3
unrolls the draft autoregressively (test-time training over a shared KV cache),
P-EAGLE predicts ``num_depths`` tokens in a single forward: COD sampling packs
all depths of the sequence onto one axis, depth>=1 positions take the learnable
``mask_hidden`` / ``mask_token_id`` inputs, a FlexAttention block mask enforces
causality, and the soft-CE loss is summed over the packed positions with
per-depth acceptance accuracy reported separately.

The constructor and ``forward`` signatures match :class:`Eagle3TrainerModule` so
the EAGLE-3 recipe's ``_forward_batch`` drives either module unchanged. Batch
size is 1 (one sequence is packed across depths per step).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy
from nemo_automodel.components.speculative.eagle.core import (
    Eagle3StepMetrics,
    _compute_target_distribution,
)
from nemo_automodel.components.speculative.eagle.peagle_attention import create_peagle_mask_mod
from nemo_automodel.components.speculative.eagle.peagle_data import generate_cod_sample_indices


class PEagleTrainerModule(nn.Module):
    """Draft-side P-EAGLE trainer with parallel multi-token prediction."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        selected_token_ids: torch.Tensor,
        selected_token_mask: torch.Tensor,
        num_depths: int = 8,
        down_sample_ratio: float = 0.7,
        down_sample_ratio_min: float = 0.2,
        mask_token_id: int = 0,
    ):
        super().__init__()
        if not isinstance(num_depths, int) or num_depths < 1:
            raise ValueError(f"PEagleTrainerModule requires num_depths >= 1, got {num_depths!r}.")
        self.draft_model = draft_model
        self.register_buffer("selected_token_ids", selected_token_ids, persistent=True)
        self.register_buffer("selected_token_mask", selected_token_mask, persistent=True)
        self.num_depths = num_depths
        self.down_sample_ratio = down_sample_ratio
        self.down_sample_ratio_min = down_sample_ratio_min
        self.mask_token_id = int(mask_token_id)

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
    ) -> Eagle3StepMetrics:
        """Run one parallel P-EAGLE step over a single packed sequence.

        Accepts the same two supervision sources as :class:`Eagle3TrainerModule`:
        the live path passes the target's full-vocab ``target_logits``; the
        offline-cache path passes the already-derived ``target_probs`` /
        ``position_mask`` (over the full sequence). Exactly one must be provided.
        """
        precomputed = target_probs is not None and position_mask is not None
        if (target_logits is not None) == precomputed:
            raise ValueError(
                "PEagleTrainerModule.forward expects exactly one supervision source: "
                "either target_logits, or both target_probs and position_mask."
            )

        device = aux_hidden_states.device
        seq_length = input_ids.shape[1]

        # Full-sequence target distribution over the draft vocab + supervised mask.
        if not precomputed:
            target_probs, position_mask = _compute_target_distribution(
                target_logits=target_logits,
                selected_token_ids=self.selected_token_ids,
                selected_token_mask=self.selected_token_mask,
                loss_mask=loss_mask,
            )

        # COD sampling -> packed (anchor_pos, depth); orig = where each element supervises.
        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=seq_length,
            loss_mask=loss_mask,
            num_depths=self.num_depths,
            down_sample_ratio=self.down_sample_ratio,
            down_sample_ratio_min=self.down_sample_ratio_min,
        )
        orig_positions = anchor_pos + depth
        total_sampled = orig_positions.shape[0]
        is_depth_0 = depth == 0

        # Packed token ids: real tokens at depth 0, mask token elsewhere.
        sampled_ids = torch.where(
            is_depth_0,
            input_ids[0, orig_positions],
            torch.full_like(orig_positions, self.mask_token_id, dtype=input_ids.dtype),
        ).unsqueeze(0)

        # Packed target hidden: real aux state at depth 0, shared mask_hidden elsewhere.
        mask_hidden = self.draft_model.mask_hidden.to(device=device, dtype=aux_hidden_states.dtype)
        sampled_hidden = torch.where(
            is_depth_0.unsqueeze(-1),
            aux_hidden_states[0, orig_positions],
            mask_hidden.squeeze(0).expand(total_sampled, -1),
        ).unsqueeze(0)
        projected = self.draft_model.project_hidden_states(sampled_hidden)

        position_ids = orig_positions.unsqueeze(0)
        lengths = attention_mask[0].sum().to(torch.long).reshape(1)
        block_mask = create_block_mask(
            create_peagle_mask_mod(anchor_pos, depth, lengths, seq_length),
            B=None,
            H=None,
            Q_LEN=total_sampled,
            KV_LEN=total_sampled,
            device=device,
        )

        hidden = self.draft_model(
            input_ids=sampled_ids,
            projected_hidden_states=projected,
            position_ids=position_ids,
            block_mask=block_mask,
        )
        logits = self.draft_model.compute_logits(hidden)

        sampled_target_probs = target_probs[:, orig_positions, :]
        sampled_position_mask = position_mask[:, orig_positions, :]
        loss = masked_soft_cross_entropy(
            logits=logits,
            target_probs=sampled_target_probs,
            position_mask=sampled_position_mask,
        )

        with torch.no_grad():
            valid = sampled_position_mask.squeeze(0).squeeze(-1).bool()
            correct = (logits.argmax(dim=-1) == sampled_target_probs.argmax(dim=-1)).squeeze(0) & valid
            running_correct = correct.sum()
            running_valid = valid.sum()

        accuracy = running_correct / running_valid.clamp_min(1.0)
        return Eagle3StepMetrics(loss=loss, accuracy=accuracy, valid_tokens=running_valid.to(loss.dtype))
