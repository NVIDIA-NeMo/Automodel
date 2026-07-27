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
"""Unit tests for :mod:`nemo_automodel.components.loss.embedding_distill`."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.embedding_distill import (
    EmbeddingDistillLoss,
    EmbeddingMSELoss,
    ScoreDistillLoss,
    cosine_distance,
    distill_loss_pair,
    mse_loss_pair,
    score_distill_loss,
)

# Relaxed tolerance: these tests target code coverage, not exact numerics.
ATOL = 1e-3


# ---------------------------------------------------------------------------
# cosine_distance
# ---------------------------------------------------------------------------
def test_cosine_distance_identical_is_zero():
    z = torch.randn(4, 8)

    dist = cosine_distance(z.clone(), z.clone())

    assert torch.allclose(dist, torch.zeros(4), atol=ATOL)


def test_cosine_distance_opposite_is_two():
    z = torch.randn(3, 8)

    dist = cosine_distance(z, -z)

    assert torch.allclose(dist, torch.full((3,), 2.0), atol=ATOL)


def test_cosine_distance_orthogonal_is_one():
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    dist = cosine_distance(s, t)

    assert torch.allclose(dist, torch.ones(2), atol=ATOL)


def test_cosine_distance_ignores_projection_scale():
    z_s = torch.randn(4, 8)
    z_t = torch.randn(4, 8)

    base = cosine_distance(z_s, z_t)
    scaled = cosine_distance(z_s * 5.0, z_t)

    assert torch.allclose(base, scaled, atol=ATOL)


# ---------------------------------------------------------------------------
# distill_loss_pair
# ---------------------------------------------------------------------------
def test_distill_loss_pair_mean_and_sum():
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    mean = distill_loss_pair(s_q, t_q, s_d, t_d, reduction="mean")
    total = distill_loss_pair(s_q, t_q, s_d, t_d, reduction="sum")

    assert torch.isfinite(mean)
    assert torch.allclose(total, mean * 4, atol=ATOL)


def test_distill_loss_pair_unknown_reduction():
    z = torch.randn(4, 8)
    with pytest.raises(ValueError, match="Unknown reduction"):
        distill_loss_pair(z, z, z, z, reduction="bogus")


# ---------------------------------------------------------------------------
# mse_loss_pair
# ---------------------------------------------------------------------------
def test_mse_loss_pair_identical_is_zero():
    z = torch.randn(4, 8)

    loss = mse_loss_pair(z.clone(), z.clone(), z.clone(), z.clone())

    assert torch.allclose(loss, torch.zeros(()), atol=ATOL)


@pytest.mark.parametrize("reduction", ["mean", "sum", "batchmean"])
def test_mse_loss_pair_reductions_finite(reduction):
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    loss = mse_loss_pair(s_q, t_q, s_d, t_d, reduction=reduction)

    assert torch.isfinite(loss)


def test_mse_loss_pair_normalize_matches_normalized_inputs():
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    normed = mse_loss_pair(s_q, t_q, s_d, t_d, normalize=True)
    manual = mse_loss_pair(
        F.normalize(s_q, dim=-1),
        F.normalize(t_q, dim=-1),
        F.normalize(s_d, dim=-1),
        F.normalize(t_d, dim=-1),
        normalize=False,
    )

    assert torch.allclose(normed, manual, atol=ATOL)


def test_mse_loss_pair_unknown_reduction():
    z = torch.randn(4, 8)
    with pytest.raises(ValueError, match="Unknown reduction"):
        mse_loss_pair(z, z, z, z, reduction="bogus")


# ---------------------------------------------------------------------------
# score_distill_loss
# ---------------------------------------------------------------------------
def test_score_distill_loss_identical_is_zero():
    torch.manual_seed(0)
    q = torch.randn(4, 8)
    d = torch.randn(4, 8)

    loss = score_distill_loss(q.clone(), q.clone(), d.clone(), d.clone())

    assert torch.allclose(loss, torch.zeros(()), atol=ATOL)


def test_score_distill_loss_finite_and_nonnegative():
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    loss = score_distill_loss(s_q, t_q, s_d, t_d, temperature=0.05)

    assert torch.isfinite(loss)
    assert loss.item() >= -ATOL


# ---------------------------------------------------------------------------
# module wrappers
# ---------------------------------------------------------------------------
def test_embedding_distill_loss_module_matches_function():
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    module = EmbeddingDistillLoss(reduction="sum")
    expected = distill_loss_pair(s_q, t_q, s_d, t_d, reduction="sum")

    assert torch.allclose(module(s_q, t_q, s_d, t_d), expected, atol=ATOL)


def test_embedding_mse_loss_module_matches_function():
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    module = EmbeddingMSELoss(normalize=True, reduction="mean")
    expected = mse_loss_pair(s_q, t_q, s_d, t_d, normalize=True, reduction="mean")

    assert torch.allclose(module(s_q, t_q, s_d, t_d), expected, atol=ATOL)


def test_score_distill_loss_module_matches_function():
    torch.manual_seed(0)
    s_q = torch.randn(4, 8)
    t_q = torch.randn(4, 8)
    s_d = torch.randn(4, 8)
    t_d = torch.randn(4, 8)

    module = ScoreDistillLoss(temperature=0.03)
    expected = score_distill_loss(s_q, t_q, s_d, t_d, temperature=0.03)

    assert torch.allclose(module(s_q, t_q, s_d, t_d), expected, atol=ATOL)
