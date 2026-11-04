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
"""Unit tests for :mod:`nemo_automodel.components.loss.cross_encoder_distill`."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.cross_encoder_distill import GroupedScoreKLLoss


def _reference_kl(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(teacher || student) over the last dim, averaged over rows."""
    p_t = F.softmax(teacher.double() / temperature, dim=-1)
    log_p_t = F.log_softmax(teacher.double() / temperature, dim=-1)
    log_p_s = F.log_softmax(student.double() / temperature, dim=-1)
    return (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()


def test_matches_hand_computed_kl():
    student = torch.tensor([[0.4, 0.1, 0.0, -0.2], [0.0, 0.3, 0.1, 0.2]])
    teacher = torch.tensor([[0.6, 0.0, 0.1, -0.1], [0.1, 0.5, -0.1, 0.0]])

    loss = GroupedScoreKLLoss(temperature=0.1)(student, teacher)

    assert torch.allclose(loss.double(), _reference_kl(student, teacher, 0.1), atol=1e-6)


def test_identical_scores_are_zero():
    scores = torch.randn(8, 4)

    loss = GroupedScoreKLLoss(temperature=0.1)(scores.clone(), scores.clone())

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_batchmean_averages_over_queries_not_elements():
    student = torch.randn(6, 4)
    teacher = torch.randn(6, 4)
    loss_fn = GroupedScoreKLLoss(temperature=0.2)

    full = loss_fn(student, teacher)
    per_row = torch.stack([loss_fn(student[i : i + 1], teacher[i : i + 1]) for i in range(6)]).mean()

    assert torch.allclose(full, per_row, atol=1e-6)


def test_reference_temperature_applies_exact_squared_ratio():
    student = torch.randn(8, 4)
    teacher = torch.randn(8, 4)

    plain = GroupedScoreKLLoss(temperature=0.1)(student, teacher)
    scaled = GroupedScoreKLLoss(temperature=0.1, reference_temperature=0.02)(student, teacher)

    assert torch.allclose(scaled, plain * (0.1 / 0.02) ** 2, atol=1e-5)


def test_reference_temperature_none_disables_scaling():
    student = torch.randn(8, 4)
    teacher = torch.randn(8, 4)

    assert torch.allclose(
        GroupedScoreKLLoss(temperature=0.1, reference_temperature=None)(student, teacher),
        GroupedScoreKLLoss(temperature=0.1)(student, teacher),
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_raises(temperature):
    with pytest.raises(ValueError, match="temperature must be > 0"):
        GroupedScoreKLLoss(temperature=temperature)


@pytest.mark.parametrize("reference_temperature", [0.0, -0.5, float("nan"), float("inf")])
def test_invalid_reference_temperature_raises(reference_temperature):
    with pytest.raises(ValueError, match="reference_temperature must be > 0"):
        GroupedScoreKLLoss(temperature=0.1, reference_temperature=reference_temperature)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        GroupedScoreKLLoss(temperature=0.1)(torch.randn(4, 4), torch.randn(4, 8))


@pytest.mark.parametrize("shape", [(16,), (4, 4, 1)])
def test_non_2d_input_raises(shape):
    with pytest.raises(ValueError, match="expects 2-D"):
        GroupedScoreKLLoss(temperature=0.1)(torch.randn(*shape), torch.randn(*shape))


def test_bfloat16_inputs_give_finite_fp32_loss():
    student = torch.randn(8, 4, dtype=torch.bfloat16)
    teacher = torch.randn(8, 4, dtype=torch.bfloat16)

    loss = GroupedScoreKLLoss(temperature=0.1, reference_temperature=0.02)(student, teacher)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_gradient_reaches_student_only():
    student = torch.randn(8, 4, requires_grad=True)
    teacher = torch.randn(8, 4, requires_grad=True)

    GroupedScoreKLLoss(temperature=0.1)(student, teacher).backward()

    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None
