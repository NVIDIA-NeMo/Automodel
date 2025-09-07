# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
Unit tests for :pyclass:`nemo_automodel.components.loss.kd_loss.KDLoss`.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.kd_loss import KDLoss

import pytest

def _reference_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    temperature: float = 1.0,
    num_batch_labels: Optional[int] = None,
) -> torch.Tensor:
    """Standalone implementation mirroring :pyfunc:`KDLoss.forward`."""

    # Flatten + mask
    valid_mask = (labels != ignore_index).view(-1)
    s_logits = student_logits.view(-1, student_logits.size(-1))[valid_mask]
    t_logits = teacher_logits.view(-1, teacher_logits.size(-1))[valid_mask]

    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature

    teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
    student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)

    kl_per_token = -(teacher_prob * student_logprob).sum(-1)  # shape: [n_valid]

    if num_batch_labels is not None:
        return kl_per_token.sum() / num_batch_labels
    return kl_per_token.mean()

@pytest.mark.parametrize("temperature,upcast,unsqueeze", [(1.0, True, False), (2.0, False, True)])
def test_kd_loss_basic(temperature, upcast, unsqueeze):
    """Loss matches reference implementation for a simple example."""
    student_logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    teacher_logits = torch.tensor([[1.5, 0.0, -0.5], [0.2, -0.1, 0.0]])
    labels = torch.tensor([0, 1])
    if unsqueeze:
        student_logits = student_logits.unsqueeze(0)
        teacher_logits = teacher_logits.unsqueeze(0)
        labels = labels.unsqueeze(0)

    loss = KDLoss(temperature=temperature, upcast=upcast)(student_logits, teacher_logits, labels)
    ref = _reference_kd_loss(student_logits, teacher_logits, labels, temperature=temperature)

    assert torch.allclose(loss, ref, atol=1e-6), f"Expected {ref}, got {loss}"

def test_kd_loss_basic_no_labels():
    """Loss matches reference implementation for a simple example."""
    student_logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    teacher_logits = torch.tensor([[1.5, 0.0, -0.5], [0.2, -0.1, 0.0]])
    labels = torch.tensor([-100, -100])

    loss = KDLoss()(student_logits, teacher_logits, labels)
    assert loss == 0.0


def test_kd_loss_ignore_index():
    """Tokens with ``ignore_index`` are excluded from the loss computation."""
    student_logits = torch.tensor(
        [[1.0, 0.0], [0.5, -0.5], [2.0, -1.0]], dtype=torch.float32
    )
    teacher_logits = torch.tensor(
        [[0.8, -0.2], [0.4, -0.4], [1.5, -0.5]], dtype=torch.float32
    )
    labels = torch.tensor([0, -100, 1])  # middle element ignored

    kd = KDLoss(ignore_index=-100)
    loss = kd(student_logits, teacher_logits, labels)

    ref = _reference_kd_loss(student_logits, teacher_logits, labels, ignore_index=-100)

    assert torch.allclose(loss, ref, atol=1e-6), f"Expected {ref}, got {loss}"


def test_kd_loss_num_labels():
    """When ``num_batch_labels`` provided, denominator equals the given count."""
    student_logits = torch.tensor([[0.3, 0.7], [1.0, -1.0]])
    teacher_logits = torch.tensor([[0.2, 0.8], [0.9, -0.9]])
    labels = torch.tensor([1, 0])
    num_labels = 10  # arbitrary count (e.g., with gradient accumulation)

    kd = KDLoss()
    loss = kd(student_logits, teacher_logits, labels, num_batch_labels=num_labels)

    ref = _reference_kd_loss(
        student_logits, teacher_logits, labels, num_batch_labels=num_labels
    )

    assert torch.allclose(loss, ref, atol=1e-6), f"Expected {ref}, got {loss}"
