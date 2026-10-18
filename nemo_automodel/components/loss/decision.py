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

"""Losses for closed-set decision-model post-training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class RLCRLossOutput:
    """Scalar loss terms from calibration-aware policy continuation."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    supervised_loss: torch.Tensor
    mean_reward: torch.Tensor
    brier: torch.Tensor


def group_masked_bidirectional_infonce(
    state_projections: torch.Tensor,
    action_projections: torch.Tensor,
    logit_scale: torch.Tensor,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the CLM bidirectional in-batch InfoNCE objective.

    Args:
        state_projections: Tensor of shape [batch, projection].
        action_projections: Tensor of shape [batch, projection] with row-aligned positives.
        logit_scale: Scalar Tensor storing the log inverse temperature.
        group_ids: Optional Tensor of shape [batch]. Off-diagonal rows with the same ID are false negatives and masked.

    Returns:
        Scalar Tensor containing the mean of state-to-action and action-to-state cross-entropy.
    """
    if state_projections.ndim != 2 or state_projections.shape != action_projections.shape:
        raise ValueError("state_projections and action_projections must have equal shape [batch, projection]")
    if state_projections.shape[0] < 2:
        raise ValueError("bidirectional InfoNCE requires at least two pairs")
    states = functional.normalize(state_projections, dim=-1)
    actions = functional.normalize(action_projections, dim=-1)
    logits = logit_scale.clamp(max=math.log(100.0)).exp() * (states @ actions.transpose(0, 1))
    labels = torch.arange(logits.shape[0], device=logits.device)
    if group_ids is not None:
        if group_ids.shape != (logits.shape[0],):
            raise ValueError("group_ids must have shape [batch]")
        false_negatives = group_ids[:, None] == group_ids[None, :]
        false_negatives.fill_diagonal_(False)
        logits = logits.masked_fill(false_negatives, float("-inf"))
    return 0.5 * (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels))


def typed_question_cross_entropy(
    record_logits: Sequence[Sequence[torch.Tensor]],
    targets: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Average cross-entropy across variable-size questions.

    Args:
        record_logits: Nested records/questions; each Tensor has shape [options].
        targets: Nested records/questions containing the zero-based correct option index.

    Returns:
        Scalar Tensor averaged uniformly over questions.
    """
    if len(record_logits) != len(targets):
        raise ValueError("record_logits and targets must contain the same number of records")
    losses: list[torch.Tensor] = []
    for logits_per_record, targets_per_record in zip(record_logits, targets):
        if len(logits_per_record) != len(targets_per_record):
            raise ValueError("each record must contain one target per question")
        for logits, target in zip(logits_per_record, targets_per_record):
            if logits.ndim != 1:
                raise ValueError("each question logit Tensor must have shape [options]")
            target_tensor = torch.tensor([target], dtype=torch.long, device=logits.device)
            losses.append(functional.cross_entropy(logits.unsqueeze(0), target_tensor))
    if not losses:
        raise ValueError("at least one labeled question is required")
    return torch.stack(losses).mean()


def rlcr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_size: int,
    exploration_std: float,
    policy_weight: float,
    supervised_weight: float,
    generator: torch.Generator | None = None,
) -> RLCRLossOutput:
    """Apply JevAny-style RLCR to closed-set decision logits.

    This is an experimental adaptation, not an upstream CLEF or CLM recipe.
    It samples Gaussian logit proposals, uses correctness minus squared
    confidence error as reward, subtracts a per-example group baseline, and
    retains a supervised cross-entropy anchor.

    Args:
        logits: Tensor of shape [batch, options].
        targets: Tensor of shape [batch] with zero-based correct option indices.
        group_size: Number of perturbed proposals sampled for each decision.
        exploration_std: Positive Gaussian standard deviation in logit space.
        policy_weight: Multiplier for the score-function policy term.
        supervised_weight: Multiplier for the cross-entropy anchor.
        generator: Optional random generator controlling proposal sampling.

    Returns:
        Named scalar loss terms. Every Tensor has shape [].
    """
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("logits must have shape [batch, options] and targets shape [batch]")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    if exploration_std <= 0:
        raise ValueError("exploration_std must be positive")
    noise = (
        torch.randn(
            logits.shape[0],
            group_size,
            logits.shape[1],
            device=logits.device,
            dtype=logits.dtype,
            generator=generator,
        )
        * exploration_std
    )
    proposals = (logits[:, None, :] + noise).detach()
    proposal_probabilities = proposals.softmax(dim=-1)
    choices = proposal_probabilities.argmax(dim=-1)
    confidence = proposal_probabilities.gather(-1, choices.unsqueeze(-1)).squeeze(-1)
    correctness = choices.eq(targets[:, None]).to(confidence.dtype)
    rewards = correctness - (confidence - correctness).square()
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    location_log_probability = -((proposals - logits[:, None, :]).square().sum(dim=-1)) / (2.0 * exploration_std**2)
    policy_loss = -(advantages.detach() * location_log_probability).mean()
    supervised_loss = functional.cross_entropy(logits, targets)
    probabilities = logits.softmax(dim=-1)
    one_hot = functional.one_hot(targets, num_classes=logits.shape[-1]).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    loss = policy_weight * policy_loss + supervised_weight * supervised_loss
    return RLCRLossOutput(loss, policy_loss, supervised_loss, rewards.mean(), brier)
