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
"""Unit tests for :mod:`nemo_automodel.components.loss.infonce`."""

import pytest
import torch

from nemo_automodel.components.loss.infonce import (
    InfoNCEDistillLoss,
    InfoNCELoss,
    infonce_distill_loss,
    infonce_loss,
)

# Relaxed tolerance: these tests target code coverage / finiteness, not exact numerics.
ATOL = 1e-3


def _grads_finite(*tensors: torch.Tensor) -> bool:
    return all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)


# ---------------------------------------------------------------------------
# infonce_loss
# ---------------------------------------------------------------------------
def test_infonce_loss_basic_in_batch():
    torch.manual_seed(0)
    q = torch.randn(4, 8, requires_grad=True)
    d = torch.randn(4, 8, requires_grad=True)

    loss = infonce_loss(q, d)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_finite(q, d)


def test_infonce_loss_with_hard_negatives():
    torch.manual_seed(0)
    q = torch.randn(3, 8, requires_grad=True)
    d = torch.randn(3, 8, requires_grad=True)
    n = torch.randn(3, 4, 8, requires_grad=True)

    loss = infonce_loss(q, d, hard_negatives=n)

    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_finite(q, d, n)


@pytest.mark.parametrize("direction", ["q2d", "d2q", "symmetric"])
def test_infonce_loss_directions(direction):
    torch.manual_seed(0)
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)
    n = torch.randn(4, 2, 8)

    loss = infonce_loss(q, d, hard_negatives=n, direction=direction)

    assert torch.isfinite(loss)


def test_infonce_loss_no_in_batch_requires_negatives():
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)

    with pytest.raises(ValueError, match="no negatives provided"):
        infonce_loss(q, d, use_in_batch_negatives=False)


def test_infonce_loss_no_in_batch_with_hard_negatives():
    torch.manual_seed(0)
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)
    n = torch.randn(4, 3, 8)

    loss = infonce_loss(q, d, hard_negatives=n, use_in_batch_negatives=False)

    assert torch.isfinite(loss)


def test_infonce_loss_without_normalize():
    torch.manual_seed(0)
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)

    loss = infonce_loss(q, d, normalize=False)

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"queries": torch.randn(2, 3, 4), "documents": torch.randn(2, 3, 4)}, "must be 2-D"),
        ({"queries": torch.randn(4, 8), "documents": torch.randn(4, 16)}, "must equal"),
        ({"queries": torch.randn(4, 8), "documents": torch.randn(4, 8), "direction": "bogus"}, "unknown direction"),
    ],
)
def test_infonce_loss_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        infonce_loss(**kwargs)


def test_infonce_loss_hard_negatives_shape_error():
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)
    n = torch.randn(4, 2, 16)  # wrong hidden dim

    with pytest.raises(ValueError, match="hard_negatives must be"):
        infonce_loss(q, d, hard_negatives=n)


def test_infonce_loss_hard_negatives_mask_shape_error():
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)
    n = torch.randn(4, 2, 8)
    mask = torch.ones(4, 3, dtype=torch.long)  # K mismatch

    with pytest.raises(ValueError, match="hard_negatives_mask must be"):
        infonce_loss(q, d, hard_negatives=n, hard_negatives_mask=mask)


def test_infonce_loss_ragged_mask_ignores_padding():
    """Padded hard negatives (mask == 0) must not change the loss value."""
    torch.manual_seed(0)
    q = torch.randn(1, 8)
    d = torch.randn(1, 8)
    real = torch.randn(1, 2, 8)
    pad = torch.randn(1, 2, 8)  # arbitrary junk in padded slots

    full = torch.cat([real, pad], dim=1)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    loss_full = infonce_loss(q, d, hard_negatives=full, hard_negatives_mask=mask)
    loss_real = infonce_loss(q, d, hard_negatives=real)

    assert torch.isfinite(loss_full)
    assert torch.allclose(loss_full, loss_real, atol=ATOL)


def test_infonce_loss_ragged_mask_finite_gradients():
    torch.manual_seed(0)
    q = torch.randn(3, 8, requires_grad=True)
    d = torch.randn(3, 8, requires_grad=True)
    n = torch.randn(3, 4, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)

    loss = infonce_loss(q, d, hard_negatives=n, hard_negatives_mask=mask)

    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_finite(q, d, n)


# ---------------------------------------------------------------------------
# infonce_distill_loss
# ---------------------------------------------------------------------------
def _distill_inputs(seed=0, batch=4, ds=8, dt=8, k=3, requires_grad=False):
    torch.manual_seed(seed)
    s_q = torch.randn(batch, ds, requires_grad=requires_grad)
    s_d = torch.randn(batch, ds, requires_grad=requires_grad)
    t_q = torch.randn(batch, dt)
    t_d = torch.randn(batch, dt)
    s_n = torch.randn(batch, k, ds, requires_grad=requires_grad)
    t_n = torch.randn(batch, k, dt)
    return s_q, s_d, t_q, t_d, s_n, t_n


@pytest.mark.parametrize("divergence", ["kl", "ce", "mse"])
def test_distill_loss_basic(divergence):
    s_q, s_d, t_q, t_d, s_n, t_n = _distill_inputs(requires_grad=True)

    loss = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
        divergence=divergence,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_finite(s_q, s_d, s_n)


@pytest.mark.parametrize("divergence", ["kl", "ce", "mse"])
def test_distill_loss_different_student_teacher_dims(divergence):
    s_q, s_d, t_q, t_d, s_n, t_n = _distill_inputs(ds=8, dt=16, requires_grad=True)

    loss = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
        divergence=divergence,
    )

    assert torch.isfinite(loss)


@pytest.mark.parametrize("direction", ["q2d", "d2q", "symmetric"])
def test_distill_loss_directions(direction):
    s_q, s_d, t_q, t_d, s_n, t_n = _distill_inputs()

    loss = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
        direction=direction,
    )

    assert torch.isfinite(loss)


@pytest.mark.parametrize("divergence", ["kl", "mse"])
def test_distill_loss_zero_when_student_matches_teacher(divergence):
    """KL(p||p) and MSE(p,p) vanish when student logits equal teacher logits."""
    torch.manual_seed(0)
    emb_q = torch.randn(4, 8)
    emb_d = torch.randn(4, 8)
    neg = torch.randn(4, 3, 8)

    loss = infonce_distill_loss(
        emb_q.clone(),
        emb_d.clone(),
        emb_q.clone(),
        emb_d.clone(),
        student_hard_negatives=neg.clone(),
        teacher_hard_negatives=neg.clone(),
        divergence=divergence,
    )

    assert torch.allclose(loss, torch.zeros(()), atol=ATOL)


def test_distill_loss_teacher_gets_no_gradient():
    s_q, s_d, _, _, s_n, _ = _distill_inputs(requires_grad=True)
    t_q = torch.randn(4, 8, requires_grad=True)
    t_d = torch.randn(4, 8, requires_grad=True)
    t_n = torch.randn(4, 3, 8, requires_grad=True)

    loss = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
        divergence="kl",
    )
    loss.backward()

    assert t_q.grad is None
    assert t_d.grad is None
    assert t_n.grad is None


@pytest.mark.parametrize("divergence", ["kl", "ce", "mse"])
def test_distill_loss_ragged_mask_ignores_padding(divergence):
    """Padded negatives must contribute nothing for every divergence."""
    torch.manual_seed(1)
    s_q = torch.randn(1, 8)
    s_d = torch.randn(1, 8)
    t_q = torch.randn(1, 8)
    t_d = torch.randn(1, 8)
    s_real = torch.randn(1, 2, 8)
    t_real = torch.randn(1, 2, 8)
    s_pad = torch.randn(1, 2, 8)
    t_pad = torch.randn(1, 2, 8)

    s_full = torch.cat([s_real, s_pad], dim=1)
    t_full = torch.cat([t_real, t_pad], dim=1)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    loss_full = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_full,
        teacher_hard_negatives=t_full,
        hard_negatives_mask=mask,
        divergence=divergence,
    )
    loss_real = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_real,
        teacher_hard_negatives=t_real,
        divergence=divergence,
    )

    assert torch.isfinite(loss_full)
    assert torch.allclose(loss_full, loss_real, atol=ATOL)


@pytest.mark.parametrize("divergence", ["kl", "ce", "mse"])
def test_distill_loss_ragged_mask_finite_loss_and_gradients(divergence):
    """Regression: ragged batches must not produce NaN loss or NaN gradients."""
    s_q, s_d, t_q, t_d, s_n, t_n = _distill_inputs(batch=3, k=4, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)

    loss = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
        hard_negatives_mask=mask,
        divergence=divergence,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_finite(s_q, s_d, s_n)


def test_distill_loss_ragged_all_padded_row_is_finite():
    """A row whose hard negatives are all padded still yields a finite loss."""
    s_q, s_d, t_q, t_d, s_n, t_n = _distill_inputs(batch=2, k=3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.long)

    loss = infonce_distill_loss(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
        hard_negatives_mask=mask,
        divergence="kl",
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_finite(s_q, s_d, s_n)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"student_queries": torch.randn(2, 3, 4)}, "student embeddings must be 2-D"),
        ({"teacher_queries": torch.randn(2, 3, 4)}, "teacher embeddings must be 2-D"),
        ({"student_documents": torch.randn(4, 16)}, "student query/doc shapes must match"),
        ({"teacher_documents": torch.randn(4, 16)}, "teacher query/doc shapes must match"),
        ({"teacher_queries": torch.randn(3, 8), "teacher_documents": torch.randn(3, 8)}, "batch sizes must match"),
        ({"direction": "bogus"}, "unknown direction"),
        ({"divergence": "bogus"}, "unknown divergence"),
    ],
)
def test_distill_loss_validation_errors(overrides, match):
    base = dict(
        student_queries=torch.randn(4, 8),
        student_documents=torch.randn(4, 8),
        teacher_queries=torch.randn(4, 8),
        teacher_documents=torch.randn(4, 8),
    )
    base.update(overrides)
    with pytest.raises(ValueError, match=match):
        infonce_distill_loss(**base)


def test_distill_loss_requires_teacher_hard_negatives():
    s_q, s_d, t_q, t_d, s_n, _ = _distill_inputs()

    with pytest.raises(ValueError, match="teacher_hard_negatives required"):
        infonce_distill_loss(s_q, s_d, t_q, t_d, student_hard_negatives=s_n)


def test_distill_loss_no_negatives_error():
    s_q, s_d, t_q, t_d, _, _ = _distill_inputs()

    with pytest.raises(ValueError, match="no negatives available"):
        infonce_distill_loss(s_q, s_d, t_q, t_d, use_in_batch_negatives=False)


# ---------------------------------------------------------------------------
# InfoNCELoss module
# ---------------------------------------------------------------------------
def test_infonce_module_forward():
    torch.manual_seed(0)
    loss_fn = InfoNCELoss()
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)

    loss = loss_fn(q, d)

    assert torch.isfinite(loss)


def test_infonce_module_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        InfoNCELoss(temperature=0.0)


def test_infonce_module_fixed_temperature_buffer():
    loss_fn = InfoNCELoss(temperature=0.07)

    assert loss_fn.log_inv_tau is None
    assert torch.allclose(loss_fn.current_temperature(), torch.tensor(0.07), atol=ATOL)


def test_infonce_module_learnable_temperature():
    loss_fn = InfoNCELoss(temperature=0.05, learnable_temperature=True)

    assert isinstance(loss_fn.log_inv_tau, torch.nn.Parameter)
    assert torch.allclose(loss_fn.current_temperature(), torch.tensor(0.05), atol=ATOL)


# ---------------------------------------------------------------------------
# InfoNCEDistillLoss module
# ---------------------------------------------------------------------------
def test_infonce_distill_module_forward():
    s_q, s_d, t_q, t_d, s_n, t_n = _distill_inputs()
    loss_fn = InfoNCEDistillLoss(divergence="kl")

    loss = loss_fn(
        s_q,
        s_d,
        t_q,
        t_d,
        student_hard_negatives=s_n,
        teacher_hard_negatives=t_n,
    )

    assert torch.isfinite(loss)


def test_infonce_distill_module_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        InfoNCEDistillLoss(temperature=-1.0)
