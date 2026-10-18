# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

Two stages are supported:

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
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from nemo_automodel.shared.decision_records import QuestionTarget

IGNORE_INDEX = -100
SCORE_TYPE = 2


@dataclass
class DecisionTargets:
    """Padded supervision for ``questions`` decisions with at most ``options`` options each.

    Attributes:
        labels: Tensor of shape [questions] with the gold option index, or ``-100`` when unlabelled.
        option_mask: Bool tensor of shape [questions, options]; True marks a real (non-padding) option.
        question_types: Tensor of shape [questions]; 0 = noul, 1 = choice, 2 = score.
        rewards: Optional float tensor of shape [questions, options] with the reward of choosing each option.
        reward_mask: Optional bool tensor of shape [questions]; True where ``rewards`` is defined.  Other rows use
            ``1[option == label]`` as their reward.
        ref_logprobs: Optional float tensor of shape [questions, options] with the reference policy's
            log-probabilities (padding entries are ignored).
        weights: Optional float tensor of shape [questions] with per-question loss weights.
    """

    labels: torch.Tensor
    option_mask: torch.Tensor
    question_types: torch.Tensor
    rewards: torch.Tensor | None = None
    reward_mask: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "DecisionTargets":
        """Return a copy with every tensor moved to ``device``."""
        return DecisionTargets(
            **{
                f.name: None if getattr(self, f.name) is None else getattr(self, f.name).to(device)
                for f in fields(self)
            }
        )


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


def pad_logits(per_question: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ragged per-question logits into a padded matrix.

    Args:
        per_question: Sequence of 1-D tensors, the i-th of shape [options_i].

    Returns:
        ``(logits, mask)``: ``logits`` has shape [questions, max_options] with ``-inf`` padding and keeps the autograd
        graph of the inputs; ``mask`` is a bool tensor of the same shape.
    """
    width = max(t.numel() for t in per_question)
    padded = [F.pad(t, (0, width - t.numel()), value=float("-inf")) for t in per_question]
    mask = torch.zeros((len(per_question), width), dtype=torch.bool, device=per_question[0].device)
    for i, t in enumerate(per_question):
        mask[i, : t.numel()] = True
    return torch.stack(padded), mask


def _pad_rows(rows: Sequence[Sequence[float] | None], width: int) -> torch.Tensor:
    out = torch.zeros((len(rows), width), dtype=torch.float32)
    for i, row in enumerate(rows):
        if row is not None:
            out[i, : len(row)] = torch.as_tensor(row, dtype=torch.float32)
    return out


def build_targets(
    questions: Sequence["QuestionTarget"],
    question_weights: dict[str, float] | None = None,
    ref_logprobs: Sequence[Sequence[float]] | None = None,
) -> DecisionTargets:
    """Pad a flat list of question targets (all questions of all records in a batch) into tensors.

    Args:
        questions: Question targets in the same order as the model's flattened per-question logits.
        question_weights: Optional ``{question_id: weight}``; missing questions get weight 1.
        ref_logprobs: Optional reference log-probabilities, one sequence of length ``options_i`` per question.

    Returns:
        A :class:`DecisionTargets` on CPU.
    """
    width = max(len(q.option_ids) for q in questions)
    mask = torch.zeros((len(questions), width), dtype=torch.bool)
    for i, q in enumerate(questions):
        mask[i, : len(q.option_ids)] = True
    has_rewards = any(q.rewards is not None for q in questions)
    question_weights = question_weights or {}
    return DecisionTargets(
        labels=torch.tensor([q.label for q in questions], dtype=torch.long),
        option_mask=mask,
        question_types=torch.tensor([q.question_type for q in questions], dtype=torch.long),
        rewards=_pad_rows([q.rewards for q in questions], width) if has_rewards else None,
        reward_mask=torch.tensor([q.rewards is not None for q in questions]) if has_rewards else None,
        ref_logprobs=_pad_rows(ref_logprobs, width) if ref_logprobs is not None else None,
        weights=torch.tensor([float(question_weights.get(q.question_id, 1.0)) for q in questions]),
    )


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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
