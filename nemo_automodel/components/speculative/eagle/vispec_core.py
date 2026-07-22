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

"""ViSpec stage-2 draft-training objective (multi-token rollout + distribution losses).

Two things separate this from the EAGLE-1/2 objective in ``core_v12.py``:

* **Self-rollout supervision.** After the first draft pass, the draft's *own*
  predicted hidden states are shifted right by one and fed back as its input
  features, ``mtp_steps`` times. Every rollout is supervised against the same
  target distribution. Training only on the target's fresh hidden states lets
  the draft lean on information it will not have at drafting depth > 1; rolling
  its own output back in removes that shortcut.
* **Distribution losses instead of hidden-state regression.** ViSpec drops the
  SmoothL1 hidden-state term entirely and supervises the *distribution*: an L1
  distance between the draft's and the target's full-vocab probabilities, plus a
  ListMLE ranking term over the target's top-k tokens.

Reference implementation: ``vispec/train/main_mtp.py`` in
https://github.com/KangJialiang/ViSpec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.loss.listmle import listmle_loss


@dataclass
class VispecStepMetrics:
    """Aggregated metrics from one ViSpec training step."""

    loss: torch.Tensor
    prob_loss: torch.Tensor
    rank_loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor


class VispecTrainerModule(nn.Module):
    """Draft-side trainer for ViSpec stage-2 (vision-aware) training.

    Args:
        draft_model: The ViSpec draft model being trained.
        target_lm_head: The frozen target ``lm_head``, used to turn predicted
            hidden states into logits. Held off the module registry so it is
            not duplicated into the draft's state dict / DDP buckets.
        prob_loss_weight: Weight on the probability-L1 term (ViSpec: 10.0).
        rank_loss_weight: Weight on the ListMLE term (ViSpec: 0.1).
        rank_loss_topk: Number of target tokens the ranking term covers.
        mtp_steps: Number of self-rollout passes after the first draft pass.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        target_lm_head: nn.Module,
        prob_loss_weight: float = 10.0,
        rank_loss_weight: float = 0.1,
        rank_loss_topk: int = 10,
        mtp_steps: int = 1,
    ):
        super().__init__()
        self.draft_model = draft_model
        object.__setattr__(self, "_target_lm_head", target_lm_head)
        object.__setattr__(self, "_cached_lm_head_weight", None)
        self.prob_loss_weight = prob_loss_weight
        self.rank_loss_weight = rank_loss_weight
        self.rank_loss_topk = rank_loss_topk
        if mtp_steps < 0:
            raise ValueError(f"mtp_steps must be >= 0, got {mtp_steps}")
        self.mtp_steps = mtp_steps

    def _lm_head_weight(self) -> torch.Tensor:
        """Return the frozen target lm_head weight as a plain local tensor.

        An FSDP2-sharded target exposes a DTensor weight; the draft runs under
        DDP with plain tensors, so it has to be gathered before ``F.linear``.
        The target is frozen, so the gathered result is cached: resolving it
        inside the rollout loop would repeat a full ``[vocab, hidden]``
        all-gather once per rollout, per micro-batch.
        """
        weight = self._cached_lm_head_weight
        if weight is None:
            weight = self._target_lm_head.weight
            if hasattr(weight, "full_tensor"):
                weight = weight.full_tensor()
            object.__setattr__(self, "_cached_lm_head_weight", weight)
        return weight

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the frozen target ``lm_head``.

        Args:
            hidden_states: Tensor of shape [..., hidden], arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., vocab].
        """
        return F.linear(hidden_states, self._lm_head_weight())

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        input_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> VispecStepMetrics:
        """Run one ViSpec training step (first pass plus ``mtp_steps`` self-rollouts).

        Args:
            inputs_embeds: Tensor of shape [1, sequence, hidden] -- target
                embedding-layer output, shifted left by one position.
            attention_mask: Tensor of shape [1, sequence]; 1 for real tokens.
            loss_mask: Tensor of shape [1, sequence]; 1 at supervised positions.
            input_hidden_states: Tensor of shape [1, sequence, hidden] -- the
                target's last hidden state, not shifted.
            target_logits: Tensor of shape [1, sequence, vocab] -- the target's
                logits, shifted left by one position.
            image_mask: Bool tensor of shape [1, sequence] aligned with
                ``inputs_embeds``.

        Returns:
            VispecStepMetrics with scalar ``loss``/``prob_loss``/``rank_loss``/
            ``accuracy`` and the ``valid_tokens`` count the losses averaged over.
        """
        predicted_hidden_states = self.draft_model(
            inputs_embeds=inputs_embeds,
            target_hidden_states=input_hidden_states,
            attention_mask=attention_mask,
            image_mask=image_mask,
        )
        rollouts = [predicted_hidden_states]
        for _ in range(self.mtp_steps):
            # Feed the draft its own prediction for the previous position: the
            # first position keeps the real target feature, the rest shift right.
            rolled_input = torch.cat((input_hidden_states[:, :1], rollouts[-1][:, :-1]), dim=1)
            rollouts.append(
                self.draft_model(
                    inputs_embeds=inputs_embeds,
                    target_hidden_states=rolled_input,
                    attention_mask=attention_mask,
                    image_mask=image_mask,
                )
            )

        valid_mask = loss_mask.bool()
        valid_tokens = valid_mask.sum()
        # Select the supervised positions before the lm_head projection: every
        # consumer below is masked anyway, so this is the same arithmetic as the
        # reference at a fraction of the vocab-sized memory -- which the rollouts
        # would otherwise multiply by ``mtp_steps + 1``.
        target_probs = torch.softmax(target_logits[valid_mask].float(), dim=-1).detach()

        # The reference averages one loss over the concatenated rollouts. Every
        # rollout contributes the same number of supervised positions, so the
        # mean of the per-rollout means is that same value, without holding a
        # ``mtp_steps + 1``-fold copy of the target distribution.
        prob_losses = []
        rank_losses = []
        accuracy = valid_tokens.new_zeros((), dtype=torch.float32)
        for index, rollout in enumerate(rollouts):
            predicted_logits = self.compute_logits(rollout[valid_mask])
            predicted_probs = torch.softmax(predicted_logits, dim=-1, dtype=torch.float32)
            prob_losses.append(torch.abs(predicted_probs - target_probs).sum(dim=-1).mean())
            rank_losses.append(listmle_loss(predicted_logits, target_probs, self.rank_loss_topk))
            if index == 0:
                # Accuracy of the first (non-rollout) pass, over supervised positions.
                correct = predicted_logits.argmax(dim=-1) == target_probs.argmax(dim=-1)
                accuracy = correct.sum() / valid_tokens.clamp_min(1)

        prob_loss = torch.stack(prob_losses).mean()
        rank_loss = torch.stack(rank_losses).mean()
        loss = self.prob_loss_weight * prob_loss + self.rank_loss_weight * rank_loss

        if valid_tokens == 0:
            # Nothing supervised in this micro-batch: ``mean()`` over the empty
            # selection is NaN. Fall back to a zero that still runs through the
            # draft, so every parameter receives a (zero) gradient and DDP's
            # ``find_unused_parameters=False`` bucket check stays satisfied.
            loss = prob_loss = rank_loss = torch.stack([r.sum() for r in rollouts]).sum() * 0.0

        return VispecStepMetrics(
            loss=loss,
            prob_loss=prob_loss,
            rank_loss=rank_loss,
            accuracy=accuracy,
            valid_tokens=valid_tokens,
        )
