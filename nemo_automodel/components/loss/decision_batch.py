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

"""Validate identified question scores against supervision before padding once.

Model adapters attach identities in their actual output order. This boundary can
reject mislabeled ordering and widths; it cannot infer identities from anonymous
logit values or detect a caller attaching incorrect metadata to those values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch.nn import functional


class DecisionTargetProtocol(Protocol):
    """Read-only supervision required by this component."""

    @property
    def question_id(self) -> str:
        """Question identity within its record."""
        ...

    @property
    def option_ids(self) -> tuple[str, ...]:
        """Option identities in model-output order."""
        ...

    @property
    def label(self) -> int:
        """Gold option index, or -100 when unlabelled."""
        ...

    @property
    def rewards(self) -> tuple[float, ...] | None:
        """Optional utilities aligned with the option identities."""
        ...

    @property
    def question_type(self) -> int:
        """Loss question type: 0=noul, 1=choice, 2=score."""
        ...


@dataclass(frozen=True)
class QuestionLogits:
    """Scores carrying their model-output identity through batch assembly.

    Attributes:
        record_id: Input record identifier.
        question_id: Question identifier within the input record.
        option_ids: Allowed option identifiers in the score tensor's order.
        values: Floating tensor of shape [options], containing raw logits or,
            only when used as a reference policy, log-probabilities.
    """

    record_id: str
    question_id: str
    option_ids: tuple[str, ...]
    values: torch.Tensor


@dataclass
class DecisionTargets:
    """Padded supervision and the original record/question/option identities.

    Attributes:
        labels: Long tensor of shape [questions]; gold option index or -100.
        option_mask: Bool tensor of shape [questions, max_options]; True for a
            real option. The assembly boundary uses this same mask for logits.
        question_types: Long tensor of shape [questions]; 0=noul, 1=choice, 2=score.
        rewards: Optional float tensor of shape [questions, max_options].
        reward_mask: Optional bool tensor of shape [questions]; explicit rewards
            are available on True rows, with labelled one-hot rewards otherwise.
        ref_logprobs: Optional float tensor of shape [questions, max_options].
        weights: Optional float tensor of shape [questions].
        record_ids: Input record identifiers in record-major order.
        questions: Original targets grouped by record, retaining option order.
            Direct tensor-only construction is valid for isolated loss math;
            assembly requires identities prepared by ``from_questions``.
    """

    labels: torch.Tensor
    option_mask: torch.Tensor
    question_types: torch.Tensor
    rewards: torch.Tensor | None = None
    reward_mask: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    record_ids: tuple[str, ...] = ()
    questions: tuple[tuple[DecisionTargetProtocol, ...], ...] = ()

    @classmethod
    def from_questions(
        cls,
        *,
        record_ids: Sequence[str],
        questions: Sequence[Sequence[DecisionTargetProtocol]],
        question_weights: Mapping[str, float] | None = None,
        reference_logprobs: Sequence[Sequence[QuestionLogits]] | None = None,
        device: torch.device | str | None = None,
    ) -> DecisionTargets:
        """Prepare supervision before model forwards, preserving identities.

        This preparation allows a distributed optimizer window to reduce its
        supervision denominator without retaining every microbatch's graph.

        Args:
            record_ids: Input record IDs, aligned with the outer question axis.
            questions: Targets in each record's expected question/option order.
            question_weights: Optional per-question weights; missing IDs use 1.
            reference_logprobs: Optional identified reference tensors of shape
                [options] per question, in the same nested record/question order.
                They must be finite; stored copies are detached from autograd.
            device: Device for the supervision tensors. Defaults to CPU.

        Returns:
            Supervision with layouts documented by ``DecisionTargets``. Padding
            entries in rewards/reference tensors are zero; option_mask is False.
        """
        if not record_ids or len(record_ids) != len(questions):
            raise ValueError("record_ids and questions must contain the same nonzero number of records")
        nested = tuple(tuple(record) for record in questions)
        flat = []
        weights = []
        for record_id, record in zip(record_ids, nested):
            if not record_id or not record:
                raise ValueError("each record must have an ID and at least one question")
            if len({q.question_id for q in record}) != len(record):
                raise ValueError(f"record {record_id!r} contains duplicate question IDs")
            for question in record:
                options = question.option_ids
                if not question.question_id or not options or len(set(options)) != len(options):
                    raise ValueError(f"record {record_id!r} needs nonempty question IDs and unique nonempty options")
                if any(not option for option in options):
                    raise ValueError(f"question {question.question_id!r} contains an empty option ID")
                if question.label != -100 and not 0 <= question.label < len(options):
                    raise ValueError(f"question {question.question_id!r} has an out-of-range label")
                if question.question_type not in (0, 1, 2):
                    raise ValueError(f"question {question.question_id!r} has an unsupported question type")
                if question.rewards is not None and (
                    len(question.rewards) != len(options) or not all(math.isfinite(x) for x in question.rewards)
                ):
                    raise ValueError(f"question {question.question_id!r} needs one finite reward per option")
                weight = float((question_weights or {}).get(question.question_id, 1.0))
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"question {question.question_id!r} needs a finite nonnegative weight")
                flat.append(question)
                weights.append(weight)
        width = max(len(q.option_ids) for q in flat)
        option_mask = torch.zeros((len(flat), width), dtype=torch.bool, device=device)
        for index, question in enumerate(flat):
            option_mask[index, : len(question.option_ids)] = True
        has_rewards = any(q.rewards is not None for q in flat)
        rewards = torch.zeros((len(flat), width), device=device) if has_rewards else None
        if rewards is not None:
            for index, question in enumerate(flat):
                if question.rewards is not None:
                    rewards[index, : len(question.option_ids)] = torch.tensor(question.rewards, device=device)
        result = cls(
            labels=torch.tensor([q.label for q in flat], dtype=torch.long, device=device),
            option_mask=option_mask,
            question_types=torch.tensor([q.question_type for q in flat], dtype=torch.long, device=device),
            rewards=rewards,
            reward_mask=torch.tensor([q.rewards is not None for q in flat], dtype=torch.bool, device=device)
            if has_rewards
            else None,
            weights=torch.tensor(weights, dtype=torch.float32, device=device),
            record_ids=tuple(record_ids),
            questions=nested,
        )
        if reference_logprobs is not None:
            references = _validated_rows(reference_logprobs, result)
            result.ref_logprobs = torch.stack(
                [
                    functional.pad(
                        row.detach().to(device=result.labels.device, dtype=torch.float32), (0, width - row.numel())
                    )
                    for row in references
                ]
            )
            if not bool(torch.isfinite(result.ref_logprobs).all()):
                raise ValueError("reference_logprobs must contain finite log-probabilities")
        return result

    def to(self, device: torch.device | str) -> DecisionTargets:
        """Return supervision on ``device``, retaining immutable identity metadata.

        Args:
            device: Destination for every tensor-bearing field described above.

        Returns:
            Targets with the same tensor shapes as this instance. Unchanged-device
            transfers may alias input tensors, following ``torch.Tensor.to``.
        """
        return replace(
            self,
            labels=self.labels.to(device),
            option_mask=self.option_mask.to(device),
            question_types=self.question_types.to(device),
            rewards=None if self.rewards is None else self.rewards.to(device),
            reward_mask=None if self.reward_mask is None else self.reward_mask.to(device),
            ref_logprobs=None if self.ref_logprobs is None else self.ref_logprobs.to(device),
            weights=None if self.weights is None else self.weights.to(device),
        )


@dataclass(frozen=True)
class DecisionBatch:
    """Validated, record-major inputs to the decision objective.

    Attributes:
        logits: Tensor of shape [questions, max_options] with -inf padding and
            autograd connected to every input question's scores.
        targets: Supervision tensors with ``DecisionTargets`` layouts; its
            option_mask is the single authoritative mask for these logits.
    """

    logits: torch.Tensor
    targets: DecisionTargets


def _validated_rows(scores: Sequence[Sequence[QuestionLogits]], targets: DecisionTargets) -> list[torch.Tensor]:
    """Check identified ragged rows against the prepared supervision order.

    Args:
        scores: Nested record/question score tensors of shape [options] each.
        targets: Prepared supervision with the ``DecisionTargets`` tensor layouts.

    Returns:
        Original score tensors of shape [options], flattened in record-major
        order without copying or changing their autograd relationships.
    """
    if not targets.questions or len(scores) != len(targets.questions):
        raise ValueError("scores must match the prepared number of records")
    rows = []
    for record_id, questions, record_scores in zip(targets.record_ids, targets.questions, scores):
        if len(record_scores) != len(questions):
            raise ValueError(f"record {record_id!r} has a mismatched question count")
        for question, score in zip(questions, record_scores):
            expected = (record_id, question.question_id, question.option_ids)
            actual = (score.record_id, score.question_id, score.option_ids)
            if actual != expected:
                raise ValueError(
                    f"record/question/option identity or order mismatch for {record_id!r}/{question.question_id!r}"
                )
            if score.values.ndim != 1 or score.values.shape[0] != len(question.option_ids):
                raise ValueError(
                    f"question {question.question_id!r} logits must have shape [options] matching option IDs"
                )
            if not score.values.is_floating_point():
                raise ValueError(f"question {question.question_id!r} needs floating-point scores")
            rows.append(score.values)
    return rows


def assemble_decision_batch(*, logits: Sequence[Sequence[QuestionLogits]], targets: DecisionTargets) -> DecisionBatch:
    """Validate identified scores and pad them using the supervision's mask.

    Args:
        logits: Per-record/question identified tensors of shape [options]. All
            rows must have the same dtype/device and follow the target identities.
        targets: Supervision with ``DecisionTargets`` layouts, prepared through
            ``from_questions`` and on the same device as logits.

    Returns:
        ``DecisionBatch`` with record-major logits [questions, max_options] and
        the original targets. The output keeps autograd connections to input
        rows; padding is -inf and uses exactly ``targets.option_mask``.
    """
    rows = _validated_rows(logits, targets)
    if any(row.device != rows[0].device or row.dtype != rows[0].dtype for row in rows):
        raise ValueError("all question logits must have the same dtype and device")
    if targets.option_mask.device != rows[0].device:
        raise ValueError("targets and question logits must be on the same device")
    option_counts = [len(q.option_ids) for record in targets.questions for q in record]
    width = max(option_counts)
    if targets.option_mask.dtype != torch.bool or targets.option_mask.shape != (len(rows), width):
        raise ValueError("option_mask shape or dtype no longer matches the prepared question identities")
    # Validate the existing mask; do not create a second independently padded mask.
    counts = torch.tensor(option_counts, device=targets.option_mask.device)
    prefix = (targets.option_mask[:, 1:] <= targets.option_mask[:, :-1]).all()
    if not bool(prefix & (targets.option_mask.sum(-1) == counts).all()):
        raise ValueError("option_mask entries no longer match the prepared question identities")
    padded = torch.stack([functional.pad(row, (0, width - row.numel())) for row in rows])
    return DecisionBatch(padded.masked_fill(~targets.option_mask, float("-inf")), targets)
