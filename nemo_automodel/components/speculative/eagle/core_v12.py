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

"""Core EAGLE-1 / EAGLE-2 draft-training logic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy


@dataclass
class EagleStepMetrics:
    """Aggregated metrics from one EAGLE-1 / EAGLE-2 training step."""

    loss: torch.Tensor
    hidden_loss: torch.Tensor
    token_loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor


class EagleTrainerModule(nn.Module):
    """Draft-side trainer for EAGLE-1 / EAGLE-2 hidden-state prediction."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        target_lm_head: nn.Module,
        hidden_loss_weight: float = 1.0,
        token_loss_weight: float = 0.1,
        feature_noise: float = 0.0,
    ):
        super().__init__()
        self.draft_model = draft_model
        # Keep a non-registered reference so we do not duplicate the target lm_head
        # inside DDP/state_dict, while still using its frozen weights for gradients
        # w.r.t. the predicted hidden states.
        object.__setattr__(self, "_target_lm_head", target_lm_head)
        self.hidden_loss_weight = hidden_loss_weight
        self.token_loss_weight = token_loss_weight
        # EAGLE feature-noise data augmentation. The original paper adds noise
        # sampled from U(-0.1, 0.1) to the target features fed to the draft during
        # training, to mitigate the error accumulation that compounds across the
        # autoregressive drafting steps. ``feature_noise`` is the (symmetric)
        # magnitude; 0 disables it. EAGLE (Li et al., 2024), arXiv:2401.15077:
        # "we employ data augmentation by adding random noise sampled from a
        # uniform distribution U(-0.1, 0.1) to features of the target LLM during
        # training."
        self.feature_noise = feature_noise
        self.hidden_loss_fn = nn.SmoothL1Loss(reduction="none")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project predicted hidden states through the frozen target lm_head."""
        weight = self._target_lm_head.weight
        # The target may be FSDP2-sharded (the EAGLE-1/2 recipe's optional
        # ``distributed:`` path, used for targets that do not fit on one GPU),
        # which makes ``weight`` a ``DTensor``. The draft runs under DDP, so
        # ``hidden_states`` is a plain tensor and ``F.linear`` cannot mix the
        # two. Gather the frozen weight to a local full tensor first -- mirrors
        # ``copy_embeddings_from_target``. No-op when the target is unsharded.
        if hasattr(weight, "full_tensor"):
            weight = weight.full_tensor()
        return F.linear(hidden_states, weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        input_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> EagleStepMetrics:
        """Run one EAGLE-1 / EAGLE-2 training step.

        Per-token tensors are ``[B, T]`` (``position_ids`` / ``doc_remaining``
        included) except the ``[B, T, H]`` hidden states and ``[B, T, V]``
        ``target_logits``; ``seq_lens`` is ``[B, max_docs]``. When packing is on,
        ``position_ids`` / ``seq_lens`` make the draft block-causal and per-document,
        and ``doc_remaining`` (real tokens after each slot within its document) gates
        supervision: a document's last real token (``doc_remaining == 0``) is dropped
        because the wrapper's global left-shift makes its target the next document's
        first token.
        """
        if self.training and self.feature_noise > 0:
            # EAGLE feature-noise augmentation (see __init__): add
            # U(-feature_noise, feature_noise) to the draft's input features.
            # ``rand_like`` is in [0, 1), so ``(rand_like - 0.5) * 2 * fn`` is
            # uniform on (-fn, fn). Only the draft *input* is perturbed; the
            # SmoothL1 regression target ``target_hidden_states`` stays clean.
            # No-op in eval (``self.training`` is False).
            input_hidden_states = input_hidden_states + (torch.rand_like(input_hidden_states) - 0.5) * (
                2.0 * self.feature_noise
            )
        predicted_hidden_states = self.draft_model(
            input_ids=input_ids,
            target_hidden_states=input_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            seq_lens=seq_lens,
        )
        predicted_logits = self.compute_logits(predicted_hidden_states)

        valid_mask = loss_mask.bool()
        if doc_remaining is not None:
            # Packing: drop each document's last real token (doc_remaining == 0),
            # whose left-shifted supervision target belongs to the next document.
            valid_mask = valid_mask & (doc_remaining > 0)
        position_mask = valid_mask.unsqueeze(-1)
        valid_tokens = valid_mask.sum()

        hidden_loss = self.hidden_loss_fn(predicted_hidden_states, target_hidden_states).mean(dim=-1)
        hidden_loss = hidden_loss[valid_mask].mean() if valid_mask.any() else hidden_loss.new_zeros(())

        target_probs = torch.softmax(target_logits.float(), dim=-1).detach()
        token_loss = masked_soft_cross_entropy(
            logits=predicted_logits,
            target_probs=target_probs,
            position_mask=position_mask,
        )
        loss = self.hidden_loss_weight * hidden_loss + self.token_loss_weight * token_loss

        target_token_ids = target_logits.argmax(dim=-1)
        predicted_token_ids = predicted_logits.argmax(dim=-1)
        correct = (predicted_token_ids == target_token_ids) & valid_mask
        accuracy = correct.sum() / valid_tokens.clamp_min(1)

        return EagleStepMetrics(
            loss=loss,
            hidden_loss=hidden_loss,
            token_loss=token_loss,
            accuracy=accuracy,
            valid_tokens=valid_tokens,
        )
