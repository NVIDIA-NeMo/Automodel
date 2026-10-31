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

"""Consumers accept structurally typed targets without a dataset dependency."""

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.datasets.decision.records import QuestionTarget
from nemo_automodel.components.eval.decision.metrics import DecisionEvaluator, EvaluationTargetProtocol
from nemo_automodel.components.loss.decision_batch import DecisionTargetProtocol, DecisionTargets


@dataclass(frozen=True)
class ExternalTarget:
    """A caller-owned implementation with no protocol inheritance."""

    question_id: str = "pick"
    question_type: int = 1
    option_ids: tuple[str, ...] = ("b", "a")
    label: int = 1
    rewards: tuple[float, ...] | None = (-0.5, 0.75)


def test_independent_target_works_for_loss_and_evaluation() -> None:
    target = ExternalTarget()
    loss_target: DecisionTargetProtocol = target
    evaluation_target: EvaluationTargetProtocol = target
    batch = DecisionTargets.from_questions(record_ids=["r"], questions=[[loss_target]])
    assert batch.labels.tolist() == [1]
    assert batch.question_types.tolist() == [1]
    torch.testing.assert_close(batch.rewards, torch.tensor([[-0.5, 0.75]]))
    evaluator = DecisionEvaluator()
    evaluator.add_batch([{"id": "r"}], [[evaluation_target]], [[[0.2, 0.8]]])
    summary = evaluator.summary()["overall"]
    assert summary["accuracy"] == 1.0
    assert summary["reward"] == 0.75
    assert summary["expected_reward"] == pytest.approx(0.5)


def test_dataset_target_satisfies_both_consumer_contracts() -> None:
    target = QuestionTarget("flag", 0, ("false", "true"), 1, None)
    loss_target: DecisionTargetProtocol = target
    evaluation_target: EvaluationTargetProtocol = target
    batch = DecisionTargets.from_questions(record_ids=["r"], questions=[[loss_target]])
    assert batch.questions == ((target,),)
    evaluator = DecisionEvaluator()
    evaluator.add_batch([{"id": "r"}], [[evaluation_target]], [[[0.0, 1.0]]])
    assert evaluator.summary()["overall"]["accuracy"] == 1.0
