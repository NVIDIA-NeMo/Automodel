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

"""Observable identity, padding and objective contracts for decision batches."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional

from nemo_automodel.components.datasets.decision.records import QuestionTarget
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import DecisionTargets, QuestionLogits, assemble_decision_batch


def _questions():
    return [
        [QuestionTarget("pick", 1, ("a", "b"), 0, None), QuestionTarget("score", 2, ("0", "1", "2", "3"), 2, None)],
        [QuestionTarget("pick", 1, ("x", "y", "z"), 1, None)],
    ]


def _scores(questions):
    return [
        [
            QuestionLogits(
                record_id,
                question.question_id,
                question.option_ids,
                torch.linspace(-0.5, 1.0, len(question.option_ids), requires_grad=True),
            )
            for question in record
        ]
        for record_id, record in zip(("first", "second"), questions, strict=True)
    ]


def test_unequal_option_counts_use_one_mask_and_question_mean_ce():
    questions = _questions()
    scores = _scores(questions)
    targets = DecisionTargets.from_questions(record_ids=["first", "second"], questions=questions)
    batch = assemble_decision_batch(logits=scores, targets=targets)
    assert batch.targets is targets
    assert batch.targets.option_mask.tolist() == [[True, True, False, False], [True] * 4, [True, True, True, False]]
    assert batch.logits.shape == (3, 4)
    assert torch.isneginf(batch.logits[~targets.option_mask]).all()
    criterion = DecisionLossConfig(label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0).build()
    actual, _ = criterion(batch.logits, batch.targets)
    # Independent unpadded reference: one CE per real question, equal question weight.
    expected = torch.stack(
        [
            functional.cross_entropy(score.values[None], torch.tensor([target.label]))
            for record_scores, record_targets in zip(scores, questions, strict=True)
            for score, target in zip(record_scores, record_targets, strict=True)
        ]
    ).mean()
    torch.testing.assert_close(actual, expected)
    rows = [score.values for record in scores for score in record]
    actual_grads = torch.autograd.grad(actual, rows, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, rows)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "misalignment", ["record_count", "question_count", "record_id", "question_id", "option_order", "width", "rank"]
)
def test_batch_rejects_misaligned_scores(misalignment):
    questions = _questions()
    scores = _scores(questions)
    targets = DecisionTargets.from_questions(record_ids=["first", "second"], questions=questions)
    if misalignment == "record_count":
        scores = scores[:1]
    elif misalignment == "question_count":
        scores[0] = scores[0][:1]
    elif misalignment == "record_id":
        scores[0][0] = replace(scores[0][0], record_id="second")
    elif misalignment == "question_id":
        scores[0] = scores[0][::-1]
    elif misalignment == "option_order":
        scores[0][0] = replace(scores[0][0], option_ids=("b", "a"))
    elif misalignment == "width":
        scores[0][0] = replace(scores[0][0], values=torch.zeros(3))
    else:
        scores[0][0] = replace(scores[0][0], values=torch.zeros(1, 2))
    with pytest.raises(ValueError, match="records|count|identity|shape"):
        assemble_decision_batch(logits=scores, targets=targets)


def test_same_schema_records_cannot_silently_swap_identified_scores():
    question = QuestionTarget("pick", 1, ("a", "b"), 0, None)
    targets = DecisionTargets.from_questions(record_ids=["first", "second"], questions=[[question], [question]])
    scores = _scores([[question], [question]])
    with pytest.raises(ValueError, match="identity"):
        assemble_decision_batch(logits=scores[::-1], targets=targets)


def test_reference_policy_uses_the_same_identities_and_detached_padding():
    questions = _questions()
    scores = _scores(questions)
    references = [[replace(score, values=score.values.log_softmax(-1)) for score in record] for record in scores]
    targets = DecisionTargets.from_questions(
        record_ids=["first", "second"], questions=questions, reference_logprobs=references
    )
    assert targets.ref_logprobs.shape == (3, 4)
    assert not targets.ref_logprobs.requires_grad
    assert (targets.ref_logprobs[~targets.option_mask] == 0).all()
    assembled = assemble_decision_batch(logits=scores, targets=targets)
    criterion = DecisionLossConfig(mode="rl", anchor_weight=0.0).build()
    _, metrics = criterion(assembled.logits, assembled.targets)
    torch.testing.assert_close(metrics["kl"], torch.tensor(0.0), atol=1e-7, rtol=0)
    references[0][0] = replace(references[0][0], option_ids=("b", "a"))
    with pytest.raises(ValueError, match="identity"):
        DecisionTargets.from_questions(
            record_ids=["first", "second"], questions=questions, reference_logprobs=references
        )


def test_unlabelled_and_reward_only_rows_keep_existing_weighted_rl_semantics():
    questions = [
        [
            QuestionTarget("missing", 1, ("a", "b"), -100, None),
            QuestionTarget("reward", 1, ("a", "b", "c"), -100, (0.0, 1.0, -1.0)),
        ]
    ]
    scores = [
        [
            QuestionLogits("record", "missing", ("a", "b"), torch.tensor([0.5, -0.5], requires_grad=True)),
            QuestionLogits("record", "reward", ("a", "b", "c"), torch.tensor([0.2, 0.5, -0.1], requires_grad=True)),
        ]
    ]
    targets = DecisionTargets.from_questions(
        record_ids=["record"], questions=questions, question_weights={"missing": 9.0, "reward": 2.0}
    )
    assembled = assemble_decision_batch(logits=scores, targets=targets)
    criterion = DecisionLossConfig(mode="rl", kl_coef=0.0, anchor_weight=0.0).build()
    assert criterion.normalization_weight(targets).item() == 2.0
    loss, _ = criterion(assembled.logits, assembled.targets)
    expected = -(scores[0][1].values.softmax(-1) * torch.tensor([0.0, 1.0, -1.0])).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(scores[0][0].values.grad, torch.zeros(2))
    assert torch.isfinite(scores[0][1].values.grad).all()
    assert DecisionLossConfig(mode="sft").build().normalization_weight(targets).item() == 0.0


def test_supervision_prepass_does_not_require_model_logits():
    targets = DecisionTargets.from_questions(record_ids=["first", "second"], questions=_questions())
    criterion = DecisionLossConfig().build()
    assert criterion.normalization_weight(targets).item() == 3.0
    moved = targets.to("cpu")
    assert moved.record_ids == targets.record_ids
    assert moved.questions == targets.questions
    assemble_decision_batch(logits=_scores(_questions()), targets=moved)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("label", 2, "label"),
        ("option_ids", ("a", "a"), "unique"),
        ("rewards", (1.0,), "reward"),
        ("rewards", (1.0, float("nan")), "reward"),
    ],
)
def test_target_preparation_rejects_invalid_option_contract(field, value, match):
    question = replace(QuestionTarget("pick", 1, ("a", "b"), 0, None), **{field: value})
    with pytest.raises(ValueError, match=match):
        DecisionTargets.from_questions(record_ids=["record"], questions=[[question]])


def test_batch_rejects_same_shape_mask_corruption():
    questions = _questions()
    targets = DecisionTargets.from_questions(record_ids=["first", "second"], questions=questions)
    targets.option_mask[0] = torch.tensor([False, True, True, False])
    with pytest.raises(ValueError, match="option_mask entries"):
        assemble_decision_batch(logits=_scores(questions), targets=targets)
