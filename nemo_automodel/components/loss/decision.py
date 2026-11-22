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

"""Post-training objectives for decision models (per-question distributions over a finite option set).

Every question of every record is one *decision*: the model emits one logit per allowed option and the answer
distribution is a softmax over them.  Questions have different numbers of options, so a batch is represented as a
padded ``[questions, options]`` logit matrix plus a boolean option mask, where ``questions`` counts every question of
every record in the batch and ``options`` is the largest option count among them.

The configured decision objective supports two stages:

* **SFT** - label-smoothed cross-entropy on the gold option, plus a Brier term (calibration) and a ranked-probability
  score (squared EMD) for ordinal ``score`` questions. The optional ranked-probability term applies only to
  ordinal questions.

* **RL** - the decision is a one-step contextual bandit, so the policy gradient can be computed two ways:

  - ``expected_reward``: maximise ``E_{a~pi}[r(a)] = sum_a pi(a) r(a)`` exactly.  With a finite option set this is the
    exact policy gradient without option-sampling variance ``sum_a pi(a) (r(a) - E[r]) grad log pi(a)``.
  - ``grpo``: sample ``num_samples`` options per question, use the group-mean (optionally std-normalised) reward as
    the baseline and apply a grouped REINFORCE surrogate for single-step decisions.

  Both add a KL penalty to a frozen reference policy (normally the SFT model) and an optional proper-scoring anchor
  (log score of the gold option). These objectives do not guarantee calibration.

The same module owns the tutorial-specific group-masked bidirectional InfoNCE
objective and experimental Gaussian-proposal RLCR continuation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.loss.decision_batch import DecisionTargets


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
    states = F.normalize(state_projections, dim=-1)
    actions = F.normalize(action_projections, dim=-1)
    logits = logit_scale.clamp(max=math.log(100.0)).exp() * (states @ actions.transpose(0, 1))
    labels = torch.arange(logits.shape[0], device=logits.device)
    if group_ids is not None:
        if group_ids.shape != (logits.shape[0],):
            raise ValueError("group_ids must have shape [batch]")
        false_negatives = group_ids[:, None] == group_ids[None, :]
        false_negatives.fill_diagonal_(False)
        logits = logits.masked_fill(false_negatives, float("-inf"))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


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
    supervised_loss = F.cross_entropy(logits, targets)
    probabilities = logits.softmax(dim=-1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).to(probabilities.dtype)
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    loss = policy_weight * policy_loss + supervised_weight * supervised_loss
    return RLCRLossOutput(loss, policy_loss, supervised_loss, rewards.mean(), brier)


IGNORE_INDEX = -100
SCORE_TYPE = 2


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over valid options.

    Args:
        logits: Tensor of shape [questions, options]; padding entries may hold any value (including ``-inf``).
        mask: Bool tensor of shape [questions, options]; True marks a real option.

    Returns:
        Float32 tensor of shape [questions, options].  Padding entries are exactly 0 (not ``-inf``) so that products
        with probabilities stay finite.
    """
    logits = logits.float().masked_fill(~mask, float("-inf"))
    return torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reduce eligible per-question values without inventing a zero-weight loss.

    Args:
        values: Tensor of shape [questions].
        weights: Tensor of shape [questions], zero where supervision is absent.

    Returns:
        Scalar tensor; a zero-weight batch stays connected to the input graph.
    """
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


@dataclass
class DecisionLossConfig:
    """Hyper-parameters of the decision objectives (see the module docstring).

    Attributes:
        mode: ``"sft"`` for supervised fine-tuning or ``"rl"`` for reward-driven post-training.
        label_smoothing: SFT label smoothing over the valid options of each question.
        brier_weight: SFT weight of the Brier score.
        ordinal_weight: SFT weight of the ranked probability score on ``score`` questions.
        rl_objective: ``"expected_reward"`` (exact policy gradient) or ``"grpo"`` (sampled, group baseline).
        num_samples: GRPO samples per question.
        sample_temperature: GRPO sampling temperature.
        normalize_advantages: Divide GRPO advantages by the group reward std.
        kl_coef: Weight of ``KL(pi || pi_ref)`` (needs reference log-probabilities).
        anchor_weight: Weight of the supervised gold-option log-score anchor during RL.
        question_weights: Optional ``{question_id: weight}``.
    """

    mode: Literal["sft", "rl"] = "sft"
    label_smoothing: float = 0.05
    brier_weight: float = 1.0
    ordinal_weight: float = 0.5
    rl_objective: Literal["expected_reward", "grpo"] = "expected_reward"
    num_samples: int = 8
    sample_temperature: float = 1.0
    normalize_advantages: bool = True
    kl_coef: float = 0.05
    anchor_weight: float = 0.1
    question_weights: dict[str, float] = field(default_factory=dict)

    def build(self) -> "DecisionLoss":
        """Return the loss module for this configuration."""
        if self.mode not in ("sft", "rl"):
            raise ValueError(f"mode must be 'sft' or 'rl', got {self.mode!r}")
        if self.rl_objective not in ("expected_reward", "grpo"):
            raise ValueError(f"rl_objective must be 'expected_reward' or 'grpo', got {self.rl_objective!r}")
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be between zero and one")
        if self.rl_objective == "grpo" and self.num_samples < 2:
            raise ValueError("GRPO requires at least two samples for its group baseline")
        if self.sample_temperature <= 0:
            raise ValueError("sample_temperature must be positive")
        return DecisionLoss(self)


class DecisionLoss(nn.Module):
    """Decision objective; construct it with :meth:`DecisionLossConfig.build`."""

    def __init__(self, config: DecisionLossConfig):
        super().__init__()
        self.config = config

    def _effective_weights(self, targets: DecisionTargets) -> torch.Tensor:
        """Return per-question objective weights.

        Args:
            targets: Supervision tensors with layouts documented by DecisionTargets.

        Returns:
            Tensor of shape [questions], zero where required feedback is unavailable.
        """
        eligible = targets.labels != IGNORE_INDEX
        if self.config.mode == "rl" and targets.rewards is not None:
            defined = targets.reward_mask if targets.reward_mask is not None else torch.ones_like(eligible)
            eligible = eligible | defined
        weights = (
            targets.weights.float() if targets.weights is not None else torch.ones_like(eligible, dtype=torch.float32)
        )
        return weights * eligible

    def normalization_weight(self, targets: DecisionTargets) -> torch.Tensor:
        """Return the denominator for composing microbatch or data-parallel losses.

        Args:
            targets: Supervision tensors with layouts documented by DecisionTargets.

        Returns:
            Scalar tensor containing the sum of eligible per-question weights. A
            zero local weight remains valid when another rank has feedback.
        """
        return self._effective_weights(targets).sum()

    def forward(
        self, logits: torch.Tensor, targets: DecisionTargets, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the configured objective.

        Args:
            logits: Tensor of shape [questions, options]; padding entries (``targets.option_mask`` False) are ignored.
            targets: Supervision tensors (see :class:`DecisionTargets`) on the same device as ``logits``.
            generator: Optional RNG for GRPO sampling.

        Returns:
            ``(loss, metrics)``: ``loss`` is a scalar weighted mean over questions; ``metrics`` maps names to detached
            scalar tensors (``nll``, ``brier``, ``accuracy``, ``entropy`` and, for RL, ``expected_reward``/``kl``).
        """
        cfg = self.config
        mask = targets.option_mask
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("logits must have nonempty shape [questions, options]")
        if mask.shape != logits.shape or mask.dtype != torch.bool:
            raise ValueError("option_mask must be boolean with the same [questions, options] shape as logits")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every question must contain at least one valid option")
        if targets.labels.shape != (logits.shape[0],) or targets.question_types.shape != (logits.shape[0],):
            raise ValueError("labels and question_types must have shape [questions]")
        logp = masked_log_softmax(logits, mask)
        probs = logp.exp() * mask
        width = logp.shape[-1]
        labelled = (targets.labels != IGNORE_INDEX).float()
        weights = self._effective_weights(targets)

        onehot = F.one_hot(targets.labels.clamp_min(0), width).float() * mask * labelled[:, None]
        n_valid = mask.sum(-1).clamp_min(1)
        nll = -(logp * onehot).sum(-1)
        brier = ((probs - onehot) ** 2).sum(-1)
        metrics: dict[str, torch.Tensor] = {}

        if cfg.mode == "sft":
            smooth = -(logp * mask).sum(-1) / n_valid
            ce = (1 - cfg.label_smoothing) * nll + cfg.label_smoothing * smooth
            rps = ((probs.cumsum(-1) - onehot.cumsum(-1)) ** 2 * mask).sum(-1) / (n_valid - 1).clamp_min(1)
            rps = rps * (targets.question_types == SCORE_TYPE).float()
            per_question = ce + cfg.brier_weight * brier + cfg.ordinal_weight * rps
            loss = _masked_mean(per_question, weights)
        else:
            rewards = self._rewards(targets, onehot) * mask
            expected = (probs * rewards).sum(-1)
            if cfg.rl_objective == "expected_reward":
                policy = -expected
            else:
                with torch.no_grad():
                    sample_logits = logits.detach().float().masked_fill(~mask, float("-inf")) / cfg.sample_temperature
                    actions = torch.multinomial(
                        torch.softmax(sample_logits, -1), cfg.num_samples, replacement=True, generator=generator
                    )
                    sampled = rewards.gather(-1, actions)
                    advantages = sampled - sampled.mean(-1, keepdim=True)
                    if cfg.normalize_advantages:
                        advantages = advantages / (sampled.std(-1, keepdim=True) + 1e-6)
                # Score the same tempered policy that generated the actions.
                sampled_logp = masked_log_softmax(logits / cfg.sample_temperature, mask)
                policy = -(advantages * sampled_logp.gather(-1, actions)).mean(-1)
            per_question = policy + cfg.anchor_weight * nll * labelled
            if targets.ref_logprobs is not None and cfg.kl_coef > 0:
                kl = (probs * (logp - targets.ref_logprobs.float())).masked_fill(~mask, 0.0).sum(-1)
                per_question = per_question + cfg.kl_coef * kl
                metrics["kl"] = _masked_mean(kl, weights).detach()
            loss = _masked_mean(per_question, weights)
            metrics["expected_reward"] = _masked_mean(expected, weights).detach()

        metrics["normalization_weight"] = weights.sum().detach()
        with torch.no_grad():
            predictions = logp.masked_fill(~mask, float("-inf")).argmax(-1)
            metrics["nll"] = _masked_mean(nll, labelled)
            metrics["brier"] = _masked_mean(brier, labelled)
            metrics["accuracy"] = _masked_mean((predictions == targets.labels).float(), labelled)
            metrics["entropy"] = (-(probs * logp).sum(-1)).mean()
        return loss, metrics

    @staticmethod
    def _rewards(targets: DecisionTargets, onehot: torch.Tensor) -> torch.Tensor:
        """Per-option rewards.

        Args:
            targets: Supervision tensors.
            onehot: Float tensor of shape [questions, options] with the gold option one-hot (fallback reward).

        Returns:
            Float tensor of shape [questions, options].
        """
        if targets.rewards is None:
            return onehot
        defined = targets.reward_mask if targets.reward_mask is not None else torch.ones_like(targets.labels).bool()
        return torch.where(defined[:, None], targets.rewards.float(), onehot)
