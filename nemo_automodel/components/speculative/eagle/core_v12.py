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

from nemo_automodel.components.loss.listmle import listmle_loss
from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy


@dataclass
class FeatureNoiseConfig:
    """Train-only uniform noise added to the target features fed to the draft.

    EAGLE's reference implementation draws the perturbation as
    ``(rand_like(x) - 0.5) * std * reference_seq_len / T``, where ``T`` is the
    effective (unpadded) sequence length of the batch. The half-width is therefore
    ``std / 2 * reference_seq_len / T``: it is calibrated at
    ``reference_seq_len`` and shrinks as the sequence grows, so a longer
    context is not perturbed proportionally harder. ViSpec inherits this
    unchanged and enables it in both of its stages.

    ``reference_seq_len=None`` drops the scaling and applies a fixed
    ``U(-std/2, std/2)`` at every length. That is the EAGLE paper's wording
    (``U(-0.1, 0.1)``, i.e. ``std=0.2``) and what the EAGLE-1/2 recipe uses; it
    is 8x the reference half-width once ``T`` reaches 4096, which is why the
    ViSpec stages take the scaled form.

    Attributes:
        std: Width of the underlying uniform draw before scaling. The reference
            uses 0.2, giving a half-width of 0.1 at ``reference_seq_len``.
        reference_seq_len: Sequence length the half-width is calibrated at, or
            ``None`` for a fixed half-width that does not track ``T``.
    """

    std: float = 0.2
    reference_seq_len: int | None = 512

    @classmethod
    def fixed(cls, half_width: float) -> "FeatureNoiseConfig":
        """Build an unscaled ``U(-half_width, half_width)`` draw."""
        return cls(std=2.0 * half_width, reference_seq_len=None)

    def half_width(self, seq_len: int) -> float:
        """Return the symmetric noise half-width for a sequence of ``seq_len``."""
        if self.reference_seq_len is None:
            return 0.5 * self.std
        return 0.5 * self.std * self.reference_seq_len / seq_len

    def apply(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return ``features`` perturbed by the uniform draw.

        Args:
            features: Tensor of shape [batch, sequence, hidden] containing target
                features handed to the draft.
            attention_mask: Tensor of shape [batch, sequence]. Its longest
                unpadded row sets the scale, so padding does not shrink the
                perturbation.

        Returns:
            Tensor of shape [batch, sequence, hidden]. This is a new tensor;
            ``features`` is not modified in place.
        """
        effective_seq_len = max(1, int(attention_mask.sum(dim=-1).max().item()))
        half_width = self.half_width(effective_seq_len)
        if half_width <= 0:
            return features
        # One RNG kernel into a fresh buffer, then one add, rather than the
        # four elementwise passes a ``rand_like``-based expression allocates.
        return torch.empty_like(features).uniform_(-half_width, half_width).add_(features)


@dataclass
class EagleStepMetrics:
    """Aggregated metrics from one EAGLE-1 / EAGLE-2 training step."""

    loss: torch.Tensor
    hidden_loss: torch.Tensor
    token_loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    rank_loss: torch.Tensor | None = None


class EagleTrainerModule(nn.Module):
    """Draft-side trainer for EAGLE-1 / EAGLE-2 hidden-state prediction."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        target_lm_head: nn.Module,
        hidden_loss_weight: float = 1.0,
        token_loss_weight: float = 0.1,
        feature_noise_config: FeatureNoiseConfig | None = None,
        rank_loss_weight: float = 0.0,
        rank_loss_topk: int = 10,
    ):
        super().__init__()
        self.draft_model = draft_model
        # Keep a non-registered reference so we do not duplicate the target lm_head
        # inside DDP/state_dict, while still using its frozen weights for gradients
        # w.r.t. the predicted hidden states.
        object.__setattr__(self, "_target_lm_head", target_lm_head)
        self.hidden_loss_weight = hidden_loss_weight
        self.token_loss_weight = token_loss_weight
        # Optional Plackett-Luce ranking term over the target's top-k tokens.
        # EAGLE-1/2 does not use it, so it is off by default and the objective is
        # unchanged at 0.0. ViSpec's stage 1 does: its reference implementation
        # computes ``v_w * vloss + p_w * (ploss + 0.1 * rloss)`` with ``v_w=1.0``
        # and ``p_w=0.1``, so the ranking term carries an effective weight of
        # 0.1 * 0.1 = 0.01 in the total. ``rank_loss_weight`` is that effective
        # weight, applied directly rather than nested, so the three terms read
        # off the config independently.
        self.rank_loss_weight = rank_loss_weight
        self.rank_loss_topk = rank_loss_topk
        # EAGLE feature-noise data augmentation, mitigating the error that
        # compounds across the autoregressive drafting steps. EAGLE (Li et al.,
        # 2024), arXiv:2401.15077: "we employ data augmentation by adding random
        # noise sampled from a uniform distribution U(-0.1, 0.1) to features of
        # the target LLM during training." ``None`` disables it; the width rule
        # (fixed or sequence-scaled) lives on :class:`FeatureNoiseConfig`.
        self.feature_noise_config = feature_noise_config
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
        if self.training and self.feature_noise_config is not None:
            # EAGLE feature-noise augmentation (see __init__). Only the draft
            # *input* is perturbed; the SmoothL1 regression target
            # ``target_hidden_states`` stays clean. No-op in eval
            # (``self.training`` is False), matching the reference, which
            # attaches the transform to the train dataset only.
            input_hidden_states = self.feature_noise_config.apply(input_hidden_states, attention_mask)
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

        rank_loss = None
        if self.rank_loss_weight > 0.0:
            # Ranking is only defined where there is supervision; an empty
            # selection would make topk/logcumsumexp return NaN.
            if valid_mask.any():
                rank_loss = listmle_loss(predicted_logits[valid_mask], target_probs[valid_mask], self.rank_loss_topk)
            else:
                rank_loss = predicted_logits.sum() * 0.0
            loss = loss + self.rank_loss_weight * rank_loss

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
            rank_loss=rank_loss,
        )
