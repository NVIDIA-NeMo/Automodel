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

"""Tests for the experimental soft-MTP (KD self-distillation) branch of ``calculate_mtp_loss``.

Prototype for kashif/experiment/soft-mtp: each MTP depth trains against the
main head's own (detached) distribution via forward-KL (reusing the
already-tested :class:`KDLoss`) instead of the hard label. These tests pin
down the one thing most likely to be silently wrong: whether the teacher
logits are rolled to the *same* position the student's depth-k target
occupies.
"""

from __future__ import annotations

import torch

from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.models.common.mtp import roll_tensor

IGNORE = -100


def test_soft_distillation_requires_both_or_neither():
    labels = torch.randint(0, 8, (1, 6))
    mtp_per_depth_logits = [torch.randn(1, 6, 8, requires_grad=True)]
    teacher_logits = torch.randn(1, 6, 8)

    import pytest

    with pytest.raises(ValueError, match="must be supplied together"):
        calculate_mtp_loss(
            MaskedCrossEntropy(ignore_index=IGNORE),
            mtp_per_depth_logits=mtp_per_depth_logits,
            labels=labels,
            model=None,
            teacher_logits=teacher_logits,
            distillation_loss_fn=None,
        )


def test_soft_distillation_rejects_cp_targets():
    import pytest

    labels = torch.randint(0, 8, (1, 6))
    mtp_per_depth_logits = [torch.randn(1, 6, 8, requires_grad=True)]
    teacher_logits = torch.randn(1, 6, 8)
    targets = [torch.full_like(labels, IGNORE)]

    with pytest.raises(NotImplementedError, match="mtp_per_depth_targets"):
        calculate_mtp_loss(
            MaskedCrossEntropy(ignore_index=IGNORE),
            mtp_per_depth_logits=mtp_per_depth_logits,
            mtp_per_depth_targets=targets,
            labels=labels,
            model=None,
            teacher_logits=teacher_logits,
            distillation_loss_fn=KDLoss(ignore_index=IGNORE),
        )


def test_soft_distillation_basic_shape_and_gradient_flow():
    torch.manual_seed(0)
    batch, seq, vocab = 2, 10, 16
    labels = torch.randint(0, vocab, (batch, seq))
    teacher_logits = torch.randn(batch, seq, vocab)
    mtp_per_depth_logits = [
        torch.randn(batch, seq, vocab, requires_grad=True),
        torch.randn(batch, seq, vocab, requires_grad=True),
    ]

    loss = calculate_mtp_loss(
        MaskedCrossEntropy(ignore_index=IGNORE),
        mtp_per_depth_logits=mtp_per_depth_logits,
        labels=labels,
        model=None,
        scaling_factor=0.1,
        teacher_logits=teacher_logits,
        distillation_loss_fn=KDLoss(ignore_index=IGNORE),
    )

    assert loss.dim() == 0
    assert loss.item() >= 0.0  # KL divergence is non-negative
    loss.backward()
    for depth_logits in mtp_per_depth_logits:
        assert depth_logits.grad is not None
        assert torch.isfinite(depth_logits.grad).all()


def test_soft_distillation_teacher_offset_matches_hard_label_offset():
    """The teacher must be rolled to the exact position depth k's hard label would occupy.

    Build student logits at depth k as an *exact copy* of the teacher rolled by
    -(k+1) along the sequence dim (matching the same roll `calculate_mtp_loss`
    applies to `cur_labels` for the hard path). If the offset is right, every
    valid (non-tail) position has student == teacher exactly, so the per-token
    forward KL must be ~0 there. A wrong offset would make student and teacher
    see different positions and the KL would be far from zero.
    """
    torch.manual_seed(0)
    batch, seq, vocab = 1, 12, 20
    labels = torch.randint(0, vocab, (batch, seq))
    teacher_logits = torch.randn(batch, seq, vocab)

    depths = 3
    student_logits = [roll_tensor(teacher_logits, shifts=-(k + 1), dim=-2).clone() for k in range(depths)]

    loss = calculate_mtp_loss(
        MaskedCrossEntropy(ignore_index=IGNORE),
        mtp_per_depth_logits=student_logits,
        labels=labels,
        model=None,
        scaling_factor=1.0,
        teacher_logits=teacher_logits,
        distillation_loss_fn=KDLoss(ignore_index=IGNORE),
    )

    assert loss.item() < 1e-5, f"expected ~0 KL when student==teacher at the aligned offset, got {loss.item()}"


def test_soft_distillation_wrong_offset_is_not_near_zero():
    """Sanity check for the test above: a deliberately wrong roll offset must NOT be ~0.

    Guards against the aligned-offset test passing for a trivial reason (e.g.
    both tensors collapsing to the same masked-out region).
    """
    torch.manual_seed(1)
    batch, seq, vocab = 1, 12, 20
    labels = torch.randint(0, vocab, (batch, seq))
    teacher_logits = torch.randn(batch, seq, vocab)

    depths = 2
    # Off-by-one: roll by -(k+2) instead of -(k+1).
    student_logits = [roll_tensor(teacher_logits, shifts=-(k + 2), dim=-2).clone() for k in range(depths)]

    loss = calculate_mtp_loss(
        MaskedCrossEntropy(ignore_index=IGNORE),
        mtp_per_depth_logits=student_logits,
        labels=labels,
        model=None,
        scaling_factor=1.0,
        teacher_logits=teacher_logits,
        distillation_loss_fn=KDLoss(ignore_index=IGNORE),
    )

    assert loss.item() > 1e-3, "misaligned student/teacher should not look like zero KL"
