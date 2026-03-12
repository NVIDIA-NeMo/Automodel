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
# Tests for the PP-specific KD wrapper logic
# ---------------------------------------------------------------------------

# Standalone re-implementation of the capture closure and the pp_kd_loss_fn
# wrapper used in KnowledgeDistillationRecipeForNextTokenPrediction, so these
# tests have no dependency on the recipe's distributed infrastructure.


def _make_capture_fn(capture_list):
    """Reproduce _teacher_capture_loss_fn from kd.py."""

    def _teacher_capture_loss_fn(logits, target, **kwargs):
        capture_list[0] = logits.detach().clone()
        return logits.new_tensor(0.0, dtype=logits.dtype)

    return _teacher_capture_loss_fn


class _SimpleRecipe:
    """Minimal stand-in for KnowledgeDistillationRecipeForNextTokenPrediction.

    Only carries the attributes that pp_kd_loss_fn reads, so it can be
    constructed without any distributed setup.
    """

    def __init__(self, kd_ratio, ce_fn, kd_loss_fn):
        self.kd_ratio = kd_ratio
        self._ce_fn = ce_fn  # replaces calculate_loss(self.loss_fn, ...)
        self.kd_loss_fn = kd_loss_fn
        self._ce_loss_buffer = []
        self._kd_loss_buffer = []
        self._current_teacher_logits = None
        self._current_num_label_tokens = None


def _make_pp_kd_loss_fn(recipe):
    """Reproduce _make_pp_kd_loss_wrapper from kd.py, using recipe._ce_fn
    instead of calculate_loss so the test needs no recipe/model import."""

    def pp_kd_loss_fn(logits, target, **kwargs):
        teacher_logits = getattr(recipe, "_current_teacher_logits", None)
        num_label_tokens = getattr(recipe, "_current_num_label_tokens", None)
        if teacher_logits is None:
            raise RuntimeError(
                "KD loss wrapper: _current_teacher_logits not set. "
                "Teacher pipeline eval must run before student step."
            )
        if recipe.kd_ratio >= 1.0:
            ce_loss = logits.new_tensor(0.0, dtype=logits.dtype)
        else:
            ce_loss = recipe._ce_fn(logits=logits, labels=target, num_label_tokens=num_label_tokens)
        kd_loss = recipe.kd_loss_fn(logits, teacher_logits, target, num_batch_labels=num_label_tokens)
        recipe._ce_loss_buffer.append(ce_loss.detach().clone())
        recipe._kd_loss_buffer.append(kd_loss.detach().clone())
        return (1.0 - recipe.kd_ratio) * ce_loss + recipe.kd_ratio * kd_loss

    return pp_kd_loss_fn


def _simple_ce(logits, labels, num_label_tokens=None):
    """Simple cross-entropy summed over valid tokens, divided by num_label_tokens."""
    valid = (labels != -100).view(-1)
    loss_sum = F.cross_entropy(
        logits.view(-1, logits.size(-1))[valid],
        labels.view(-1)[valid],
        reduction="sum",
    )
    if num_label_tokens is not None:
        return loss_sum / num_label_tokens
    return loss_sum / valid.sum().clamp(min=1)


def test_teacher_capture_fn_stores_logits_and_returns_zero():
    """_teacher_capture_loss_fn stores logits in the capture list and returns 0."""
    capture = [None]
    capture_fn = _make_capture_fn(capture)

    logits = torch.randn(3, 8)
    target = torch.zeros(3, dtype=torch.long)

    result = capture_fn(logits, target)

    assert capture[0] is not None, "capture list should be populated"
    assert torch.allclose(capture[0], logits), "captured logits must match input logits"
    assert result.item() == pytest.approx(0.0), "capture fn must return 0.0"
    assert not result.requires_grad, "returned tensor must not require grad"


def test_teacher_capture_fn_overwrites_on_repeated_calls():
    """Each call to the capture fn replaces the previous logits (last-microbatch semantics)."""
    capture = [None]
    capture_fn = _make_capture_fn(capture)

    logits_first = torch.randn(3, 8)
    logits_second = torch.randn(3, 8)
    target = torch.zeros(3, dtype=torch.long)

    capture_fn(logits_first, target)
    capture_fn(logits_second, target)

    assert torch.allclose(capture[0], logits_second), "only last microbatch logits should be retained"


def test_pp_kd_loss_fn_raises_when_teacher_logits_missing():
    """pp_kd_loss_fn raises RuntimeError when _current_teacher_logits is None."""
    recipe = _SimpleRecipe(kd_ratio=0.5, ce_fn=_simple_ce, kd_loss_fn=KDLoss())
    loss_fn = _make_pp_kd_loss_fn(recipe)

    logits = torch.randn(4, 6)
    labels = torch.randint(0, 6, (4,))
    with pytest.raises(RuntimeError, match="_current_teacher_logits not set"):
        loss_fn(logits, labels)


def test_pp_kd_loss_fn_correct_combination():
    """Combined loss equals (1-ratio)*ce + ratio*kd for a standard ratio."""
    torch.manual_seed(0)
    kd_ratio = 0.7
    num_label_tokens = 8

    student_logits = torch.randn(4, 6)
    teacher_logits = torch.randn(4, 6)
    labels = torch.tensor([0, 1, -100, 2])

    kd_fn = KDLoss()
    recipe = _SimpleRecipe(kd_ratio=kd_ratio, ce_fn=_simple_ce, kd_loss_fn=kd_fn)
    recipe._current_teacher_logits = teacher_logits.clone()
    recipe._current_num_label_tokens = num_label_tokens
    loss_fn = _make_pp_kd_loss_fn(recipe)

    combined = loss_fn(student_logits, labels)

    expected_ce = _simple_ce(logits=student_logits, labels=labels, num_label_tokens=num_label_tokens)
    expected_kd = kd_fn(student_logits, teacher_logits, labels, num_batch_labels=num_label_tokens)
    expected = (1.0 - kd_ratio) * expected_ce + kd_ratio * expected_kd

    assert torch.allclose(combined, expected, atol=1e-6), f"Expected {expected.item()}, got {combined.item()}"


def test_pp_kd_loss_fn_kd_ratio_one_zeros_ce():
    """When kd_ratio=1.0, ce_loss is zeroed and combined loss equals kd_loss."""
    torch.manual_seed(1)
    num_label_tokens = 6

    student_logits = torch.randn(3, 5)
    teacher_logits = torch.randn(3, 5)
    labels = torch.tensor([0, 2, 1])

    kd_fn = KDLoss()
    recipe = _SimpleRecipe(kd_ratio=1.0, ce_fn=_simple_ce, kd_loss_fn=kd_fn)
    recipe._current_teacher_logits = teacher_logits.clone()
    recipe._current_num_label_tokens = num_label_tokens
    loss_fn = _make_pp_kd_loss_fn(recipe)

    combined = loss_fn(student_logits, labels)
    expected_kd = kd_fn(student_logits, teacher_logits, labels, num_batch_labels=num_label_tokens)

    assert torch.allclose(combined, expected_kd, atol=1e-6)
    # CE buffer must be zero (CE was skipped).
    assert recipe._ce_loss_buffer[-1].item() == pytest.approx(0.0)


def test_pp_kd_loss_fn_kd_ratio_zero_zeros_kd_contribution():
    """When kd_ratio=0.0, the kd term has zero weight; combined loss equals ce_loss."""
    torch.manual_seed(2)
    num_label_tokens = 4

    student_logits = torch.randn(4, 5)
    teacher_logits = torch.randn(4, 5)
    labels = torch.tensor([0, 1, 2, 3])

    kd_fn = KDLoss()
    recipe = _SimpleRecipe(kd_ratio=0.0, ce_fn=_simple_ce, kd_loss_fn=kd_fn)
    recipe._current_teacher_logits = teacher_logits.clone()
    recipe._current_num_label_tokens = num_label_tokens
    loss_fn = _make_pp_kd_loss_fn(recipe)

    combined = loss_fn(student_logits, labels)
    expected_ce = _simple_ce(logits=student_logits, labels=labels, num_label_tokens=num_label_tokens)

    assert torch.allclose(combined, expected_ce, atol=1e-6)


def test_pp_kd_loss_fn_fills_loss_buffers():
    """After each call pp_kd_loss_fn appends to _ce_loss_buffer and _kd_loss_buffer."""
    torch.manual_seed(3)
    num_label_tokens = 5

    student_logits = torch.randn(3, 4)
    teacher_logits = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])

    recipe = _SimpleRecipe(kd_ratio=0.5, ce_fn=_simple_ce, kd_loss_fn=KDLoss())
    recipe._current_teacher_logits = teacher_logits.clone()
    recipe._current_num_label_tokens = num_label_tokens
    loss_fn = _make_pp_kd_loss_fn(recipe)

    assert len(recipe._ce_loss_buffer) == 0
    assert len(recipe._kd_loss_buffer) == 0

    loss_fn(student_logits, labels)

    assert len(recipe._ce_loss_buffer) == 1
    assert len(recipe._kd_loss_buffer) == 1
    assert recipe._ce_loss_buffer[0].numel() == 1
    assert recipe._kd_loss_buffer[0].numel() == 1

    # Second call accumulates another entry (simulates grad-accumulation microbatches).
    recipe._current_teacher_logits = teacher_logits.clone()
    loss_fn(student_logits, labels)
    assert len(recipe._ce_loss_buffer) == 2
    assert len(recipe._kd_loss_buffer) == 2


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
