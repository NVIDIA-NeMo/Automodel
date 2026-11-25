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

"""Soft labels through the shared decision records, supervision tensors and SFT objective."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.decision.records import QuestionTarget, question_targets, validate_record
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import DecisionTargets

PLAIN_CE = {"label_smoothing": 0.0, "brier_weight": 0.0, "ordinal_weight": 0.0}


def make_record(**overrides):
    record = {
        "id": "r1",
        "state": {"ticket": "refund"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which queue?",
                "criteria": {"billing": "Refunds", "technical": "Bugs", "sales": "Contracts"},
            },
            "page": {"type": "noul", "instructions": "Page on-call?"},
        },
        "answers": {"route": "billing"},
        "soft_labels": {"route": {"billing": 3, "sales": 1}},
    }
    return {**record, **overrides}


def targets_for(*questions: QuestionTarget) -> DecisionTargets:
    return DecisionTargets.from_questions(record_ids=["r"], questions=[list(questions)])


def loss_of(logits: torch.Tensor, targets: DecisionTargets, **config) -> torch.Tensor:
    return DecisionLossConfig(mode="sft", **config).build()(logits, targets)[0]


# ------------------------------------------------------------------------------------------------- records
def test_soft_labels_align_with_the_model_order_and_normalize_over_all_options():
    record = make_record()
    validate_record(record)
    route, page = question_targets(record, {"route": ["sales", "technical", "billing"], "page": ["true", "false"]})
    assert route.label == 2 and route.soft_labels == pytest.approx((0.25, 0.0, 0.75))
    assert page.label == -100 and page.soft_labels is None


@pytest.mark.parametrize(
    "soft_labels,match",
    [
        ({"unknown": {"billing": 1.0}}, "soft labels for unknown question"),
        ({"route": {"refunds": 1.0}}, "soft labels must weight options"),
        ({"route": {}}, "soft labels must weight options"),
        ({"route": {"billing": -1.0, "sales": 2.0}}, "nonnegative"),
        ({"route": {"billing": 0.0}}, "positive sum"),
        ({"route": {"billing": float("nan")}}, "finite"),
        ({"route": {"billing": "high"}}, "finite nonnegative numbers"),
    ],
)
def test_malformed_soft_labels_are_rejected_before_training(soft_labels, match):
    with pytest.raises(ValueError, match=match):
        validate_record(make_record(soft_labels=soft_labels))


# ------------------------------------------------------------------------------------------------- supervision
def test_supervision_pads_soft_labels_and_marks_the_rows_that_have_them():
    soft = QuestionTarget("route", 1, ("a", "b", "c"), 0, None, (0.5, 0.25, 0.25))
    hard = QuestionTarget("page", 0, ("true", "false"), 1, None)
    targets = targets_for(soft, hard)
    torch.testing.assert_close(targets.soft_labels, torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.0, 0.0]]))
    assert targets.soft_label_mask.tolist() == [True, False]
    moved = targets.to("cpu")
    assert moved.soft_labels is not None and moved.soft_label_mask.tolist() == [True, False]
    assert targets_for(hard).soft_labels is None and targets_for(hard).soft_label_mask is None


@pytest.mark.parametrize("soft", [(0.5, 0.5), (0.5, 0.25, 0.5), (1.5, -0.25, -0.25), (float("nan"), 0.5, 0.5)])
def test_supervision_rejects_soft_labels_that_are_not_a_distribution_over_the_options(soft):
    with pytest.raises(ValueError, match="soft labels"):
        targets_for(QuestionTarget("route", 1, ("a", "b", "c"), 0, None, soft))


# ------------------------------------------------------------------------------------------------- objective
def test_soft_labels_replace_the_gold_target_and_give_the_soft_cross_entropy_gradient():
    soft = torch.tensor([0.7, 0.3, 0.0])
    targets = targets_for(QuestionTarget("route", 1, ("a", "b", "c"), 1, None, tuple(soft.tolist())))
    logits = torch.tensor([[0.1, 0.2, -0.3]], requires_grad=True)
    loss = loss_of(logits, targets, **PLAIN_CE)
    torch.testing.assert_close(loss, -(soft * logits.detach().log_softmax(-1)).sum())
    loss.backward()
    torch.testing.assert_close(logits.grad[0], logits.detach()[0].softmax(-1) - soft)


def test_without_soft_labels_the_objective_is_gold_cross_entropy():
    targets = targets_for(QuestionTarget("route", 1, ("a", "b", "c"), 2, None))
    logits = torch.tensor([[0.5, -1.0, 2.0]], requires_grad=True)
    loss = loss_of(logits, targets, **PLAIN_CE)
    torch.testing.assert_close(loss, F.cross_entropy(logits.detach(), torch.tensor([2])))
    loss.backward()
    torch.testing.assert_close(logits.grad[0], logits.detach()[0].softmax(-1) - F.one_hot(torch.tensor(2), 3).float())


def test_kl_weight_keeps_the_gold_target_and_adds_kl_to_the_soft_labels():
    soft = torch.tensor([0.5, 0.5, 0.0])
    with_soft = QuestionTarget("route", 1, ("a", "b", "c"), 0, None, tuple(soft.tolist()))
    without_soft = QuestionTarget("page", 1, ("a", "b", "c"), 0, None)
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    log_probs = logits.log_softmax(-1)
    expected = -log_probs[0, 0] + 0.5 * F.kl_div(log_probs[0], soft, reduction="sum")
    torch.testing.assert_close(loss_of(logits, targets_for(with_soft), soft_label_kl_weight=0.5, **PLAIN_CE), expected)
    torch.testing.assert_close(
        loss_of(logits, targets_for(without_soft), soft_label_kl_weight=0.5, **PLAIN_CE), -log_probs[0, 0]
    )


def test_gold_mask_ignores_soft_labels_whose_top_option_is_not_gold():
    disagreeing = QuestionTarget("route", 1, ("a", "b", "c"), 0, None, (0.1, 0.9, 0.0))
    agreeing = replace(disagreeing, soft_labels=(0.6, 0.4, 0.0))
    logits = torch.tensor([[0.3, -0.2, 0.1]])
    gold_ce = F.cross_entropy(logits, torch.tensor([0]))
    torch.testing.assert_close(
        loss_of(logits, targets_for(disagreeing), soft_label_gold_mask=True, **PLAIN_CE), gold_ce
    )
    soft_ce = -(torch.tensor([0.6, 0.4, 0.0]) * logits.log_softmax(-1)).sum()
    torch.testing.assert_close(loss_of(logits, targets_for(agreeing), soft_label_gold_mask=True, **PLAIN_CE), soft_ce)


def test_soft_labels_without_gold_supervise_only_when_they_replace_the_target():
    soft_only = QuestionTarget("route", 1, ("a", "b"), -100, None, (0.25, 0.75))
    targets = targets_for(soft_only)
    assert float(DecisionLossConfig(mode="sft").build().normalization_weight(targets)) == 1.0
    for config in ({"soft_label_kl_weight": 0.5}, {"soft_label_gold_mask": True}):
        assert float(DecisionLossConfig(mode="sft", **config).build().normalization_weight(targets)) == 0.0
    assert float(DecisionLossConfig(mode="rl").build().normalization_weight(targets)) == 0.0


def test_brier_and_ranked_probability_terms_use_the_soft_target():
    soft = torch.tensor([0.1, 0.2, 0.3, 0.4])
    targets = targets_for(QuestionTarget("severity", 2, ("0", "1", "2", "3"), 3, None, tuple(soft.tolist())))
    logits = soft.log()[None]  # the policy equals the soft target, so Brier and RPS against it vanish
    loss = loss_of(logits, targets, label_smoothing=0.0, brier_weight=1.0, ordinal_weight=0.5)
    torch.testing.assert_close(loss, -(soft * soft.log()).sum())


def test_mixed_batches_average_soft_and_gold_questions_by_weight():
    soft = QuestionTarget("route", 1, ("a", "b", "c"), 1, None, (0.2, 0.8, 0.0))
    hard = QuestionTarget("page", 1, ("a", "b", "c"), 2, None)
    logits = torch.tensor([[0.4, 0.1, -0.2], [-0.3, 0.6, 0.2]])
    each = [loss_of(logits[i : i + 1], targets_for(t), **PLAIN_CE) for i, t in enumerate((soft, hard))]
    torch.testing.assert_close(loss_of(logits, targets_for(soft, hard), **PLAIN_CE), (each[0] + each[1]) / 2)


def test_bf16_logits_with_soft_labels_are_scored_in_fp32():
    targets = targets_for(QuestionTarget("route", 1, ("a", "b", "c"), 1, None, (0.0, 0.5, 0.5)))
    loss = loss_of(torch.tensor([[30.0, -30.0, 0.0]], dtype=torch.bfloat16), targets, **PLAIN_CE)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_soft_label_kl_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="soft_label_kl_weight"):
        DecisionLossConfig(soft_label_kl_weight=weight).build()
