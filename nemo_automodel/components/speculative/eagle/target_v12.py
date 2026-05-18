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

"""Target-model wrapper for EAGLE-1 / EAGLE-2 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _shift_left_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Shift a batched sequence tensor left and zero-fill the tail."""
    tail = torch.zeros_like(tensor[:, :1])
    return torch.cat((tensor[:, 1:], tail), dim=1)


@dataclass
class EagleTargetBatch:
    """Target-model outputs needed by the EAGLE-1 / EAGLE-2 trainer."""

    input_hidden_states: torch.Tensor
    target_hidden_states: torch.Tensor
    target_logits: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class HFEagleTargetModel:
    """Thin wrapper that exposes hidden-state supervision from a causal LM."""

    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the target model input embeddings."""
        return self.model.get_input_embeddings()

    def get_lm_head(self) -> nn.Module:
        """Return the target model lm_head."""
        return self.model.lm_head

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> EagleTargetBatch:
        """Run the target transformer and prepare shifted supervision tensors."""
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        return EagleTargetBatch(
            input_hidden_states=hidden_states,
            target_hidden_states=_shift_left_with_zero(hidden_states),
            target_logits=_shift_left_with_zero(logits),
            input_ids=_shift_left_with_zero(input_ids),
            attention_mask=attention_mask,
            loss_mask=_shift_left_with_zero(loss_mask),
        )
