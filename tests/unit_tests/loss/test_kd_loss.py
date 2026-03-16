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
Unit tests for :pyclass:`nemo_automodel.components.loss.kd_loss.KDLoss` and its
tensor-parallel helpers.
"""
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.kd_loss import KDLoss, _infer_tp_group_from_dtensor, _kl_forward_tp

# ---------------------------------------------------------------------------
# Reference implementation (no TP, no T² scaling applied yet)
# ---------------------------------------------------------------------------


def _reference_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    temperature: float = 1.0,
    num_batch_labels: Optional[int] = None,
) -> torch.Tensor:
    """Standalone implementation mirroring :pyfunc:`KDLoss.forward`."""
    valid_mask = (labels != ignore_index).view(-1)
    s_logits = student_logits.view(-1, student_logits.size(-1))[valid_mask]
    t_logits = teacher_logits.view(-1, teacher_logits.size(-1))[valid_mask]

    if temperature != 1.0:
        s_logits = s_logits / temperature
        t_logits = t_logits / temperature

    teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
    student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)

    # T² scaling (Hinton et al., 2015)
    scale = temperature**2
    kl_per_token = -(teacher_prob * student_logprob).sum(-1) * scale  # shape: [n_valid]

    if num_batch_labels is not None:
        return kl_per_token.sum() / num_batch_labels
    return kl_per_token.mean()


# ---------------------------------------------------------------------------
# KDLoss – basic correctness
# ---------------------------------------------------------------------------


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

    loss = KDLoss(temperature=temperature, fp32_upcast=upcast)(student_logits, teacher_logits, labels)
    ref = _reference_kd_loss(student_logits, teacher_logits, labels, temperature=temperature)

    assert torch.allclose(loss, ref, atol=1e-6), f"Expected {ref}, got {loss}"


def test_kd_loss_basic_no_labels():
    """Returns zero when the entire batch is padding."""
    student_logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    teacher_logits = torch.tensor([[1.5, 0.0, -0.5], [0.2, -0.1, 0.0]])
    labels = torch.tensor([-100, -100])

    loss = KDLoss()(student_logits, teacher_logits, labels)
    assert loss == 0.0


def test_kd_loss_ignore_index():
    """Tokens with ``ignore_index`` are excluded from the loss computation."""
    student_logits = torch.tensor([[1.0, 0.0], [0.5, -0.5], [2.0, -1.0]], dtype=torch.float32)
    teacher_logits = torch.tensor([[0.8, -0.2], [0.4, -0.4], [1.5, -0.5]], dtype=torch.float32)
    labels = torch.tensor([0, -100, 1])  # middle element ignored

    loss = KDLoss(ignore_index=-100)(student_logits, teacher_logits, labels)
    ref = _reference_kd_loss(student_logits, teacher_logits, labels, ignore_index=-100)

    assert torch.allclose(loss, ref, atol=1e-6), f"Expected {ref}, got {loss}"


def test_kd_loss_num_labels():
    """When ``num_batch_labels`` is provided, denominator equals the given count."""
    student_logits = torch.tensor([[0.3, 0.7], [1.0, -1.0]])
    teacher_logits = torch.tensor([[0.2, 0.8], [0.9, -0.9]])
    labels = torch.tensor([1, 0])
    num_labels = 10

    loss = KDLoss()(student_logits, teacher_logits, labels, num_batch_labels=num_labels)
    ref = _reference_kd_loss(student_logits, teacher_logits, labels, num_batch_labels=num_labels)

    assert torch.allclose(loss, ref, atol=1e-6), f"Expected {ref}, got {loss}"


# ---------------------------------------------------------------------------
# T² scaling
# ---------------------------------------------------------------------------


def test_kd_loss_temperature_scaling():
    """T² scaling keeps gradient magnitudes consistent across temperatures.

    For any temperature T, ``KDLoss(temperature=T)`` should equal
    ``KDLoss(temperature=1)`` only when the distributions are flat (uniform teacher
    and uniform student), because in that case temperature does not change the
    probabilities and the T² factor is the only difference.

    Here we verify the more directly testable property: the loss computed by
    KDLoss(temperature=T) matches _reference_kd_loss(temperature=T), which
    applies the T² scaling explicitly.
    """
    torch.manual_seed(42)
    student_logits = torch.randn(4, 8)
    teacher_logits = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    temperature = 3.0

    loss = KDLoss(temperature=temperature)(student_logits, teacher_logits, labels)
    ref = _reference_kd_loss(student_logits, teacher_logits, labels, temperature=temperature)

    assert torch.allclose(loss, ref, atol=1e-5), f"Expected {ref.item():.6f}, got {loss.item():.6f}"


def test_kd_loss_temperature_1_no_scaling():
    """With temperature=1 the T² factor is 1 and has no effect."""
    torch.manual_seed(0)
    student_logits = torch.randn(3, 5)
    teacher_logits = torch.randn(3, 5)
    labels = torch.tensor([0, 1, -100])

    loss_t1 = KDLoss(temperature=1.0)(student_logits, teacher_logits, labels)
    ref = _reference_kd_loss(student_logits, teacher_logits, labels, temperature=1.0)

    assert torch.allclose(loss_t1, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# _infer_tp_group_from_dtensor
# ---------------------------------------------------------------------------


def test_infer_tp_group_plain_tensor_returns_none():
    """Plain tensors are not vocab-sharded DTensors; group must be None."""
    t = torch.randn(4, 32)
    assert _infer_tp_group_from_dtensor(t) is None


# ---------------------------------------------------------------------------
# TP path: _kl_forward_tp on a trivial single-process group
#
# With world_size=1 all collectives are identity operations, so _kl_forward_tp
# must produce the same result as the standard non-TP softmax / log-softmax.
# ---------------------------------------------------------------------------


def _init_single_process_group() -> Optional[torch.distributed.ProcessGroup]:
    """Initialise (or reuse) a trivial gloo group for single-process TP tests."""
    if not torch.distributed.is_available():
        return None
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29501",
            rank=0,
            world_size=1,
        )
    return torch.distributed.group.WORLD


@pytest.fixture(scope="module")
def trivial_pg():
    """Module-scoped fixture that returns a single-process gloo ProcessGroup."""
    pg = _init_single_process_group()
    if pg is None:
        pytest.skip("torch.distributed not available")
    return pg


def test_kl_forward_tp_matches_non_tp(trivial_pg):
    """_kl_forward_tp with world_size=1 equals standard log-softmax computation."""
    torch.manual_seed(7)
    t_logits = torch.randn(6, 16, dtype=torch.float32)
    s_logits = torch.randn(6, 16, dtype=torch.float32)

    # Non-TP reference
    teacher_prob = F.softmax(t_logits, dim=-1)
    student_logprob = F.log_softmax(s_logits, dim=-1)
    ref = (teacher_prob * student_logprob).sum(-1)  # negative CE per token

    tp_out = _kl_forward_tp(t_logits, s_logits, trivial_pg)

    assert torch.allclose(tp_out, ref, atol=1e-5), f"max diff: {(tp_out - ref).abs().max().item()}"


def test_kd_loss_tp_path_matches_non_tp(trivial_pg):
    """KDLoss with an explicit tp_group (world_size=1) produces the same loss as without TP."""
    torch.manual_seed(13)
    student_logits = torch.randn(5, 20, dtype=torch.float32)
    teacher_logits = torch.randn(5, 20, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, -100, 4])

    loss_no_tp = KDLoss()(student_logits, teacher_logits, labels)
    loss_tp = KDLoss(tp_group=trivial_pg)(student_logits, teacher_logits, labels)

    assert torch.allclose(loss_no_tp, loss_tp, atol=1e-5), (
        f"Non-TP loss {loss_no_tp.item():.6f} != TP loss {loss_tp.item():.6f}"
    )


def test_kd_loss_tp_path_with_temperature(trivial_pg):
    """TP path with temperature applies T² scaling consistently with the non-TP path."""
    torch.manual_seed(99)
    student_logits = torch.randn(4, 10, dtype=torch.float32)
    teacher_logits = torch.randn(4, 10, dtype=torch.float32)
    labels = torch.tensor([0, 1, -100, 3])
    temperature = 2.0

    loss_no_tp = KDLoss(temperature=temperature)(student_logits, teacher_logits, labels)
    loss_tp = KDLoss(temperature=temperature, tp_group=trivial_pg)(student_logits, teacher_logits, labels)

    assert torch.allclose(loss_no_tp, loss_tp, atol=1e-5), (
        f"Non-TP loss {loss_no_tp.item():.6f} != TP loss {loss_tp.item():.6f}"
    )
