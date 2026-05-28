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

"""Unit tests for distributed in-batch negative sampling utilities."""

import pytest
import torch

from nemo_automodel.components.models.common.inbatch_neg_utils import (
    dist_gather_tensor,
    mask_gathered_passages_same_doc_as_positive,
)


def _is_masked(x: torch.Tensor) -> bool:
    """True when ``x`` is the dtype's ``-inf`` marker or its ``finfo.min``.

    Pass a tensor (not a Python float) so the dtype is preserved — otherwise
    ``finfo.min`` would be evaluated against the default float dtype and miss
    masked values from lower-precision dtypes (e.g. bfloat16).
    """
    assert torch.is_tensor(x), "pass tensor (not .item()) to preserve dtype"
    return bool(torch.isneginf(x).item() or x.item() <= torch.finfo(x.dtype).min)


def test_dist_gather_tensor_single_rank_is_noop():
    """world_size <= 1 short-circuits to identity (no allocation/copy).

    ``is t`` pins the no-copy contract; downgrading to ``torch.equal`` would
    let a regression silently allocate a needless copy.
    """
    t = torch.randn(4, 8)
    assert dist_gather_tensor(t) is t


def test_dist_gather_tensor_none_returns_none():
    assert dist_gather_tensor(None) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_same_doc_basic(dtype):
    """Duplicate of q0's positive doc id elsewhere in the batch must be masked
    on q0's row, while q0's positive column itself stays untouched."""
    local_bs, n_passages = 2, 2
    scores = torch.zeros(local_bs, local_bs * n_passages, dtype=dtype)
    # Layout: [q0_pos_id, q0_neg_id, q1_pos_id, q1_neg_id]
    # Force q1's neg (col 3) to share q0's pos id (col 0).
    passage_doc_ids = torch.tensor([100, 200, 300, 100], dtype=torch.long)

    mask_gathered_passages_same_doc_as_positive(
        scores,
        passage_doc_ids,
        train_n_passages=n_passages,
        rank=0,
        local_batch_size=local_bs,
    )

    # q0's row: positive (col 0) preserved, duplicate (col 3) masked.
    assert scores[0, 0].item() == 0.0
    assert _is_masked(scores[0, 3])
    # Non-matching cols on q0's row stay 0.
    assert scores[0, 1].item() == 0.0
    assert scores[0, 2].item() == 0.0
    # q1's row: q1's pos id (300) is unique, no masking anywhere.
    assert torch.all(scores[1] == 0.0)


def test_mask_same_doc_no_collisions_is_noop():
    """All unique doc ids -> scores unchanged."""
    local_bs, n_passages = 2, 2
    scores = torch.full((local_bs, local_bs * n_passages), 0.5)
    passage_doc_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    mask_gathered_passages_same_doc_as_positive(
        scores,
        passage_doc_ids,
        train_n_passages=n_passages,
        rank=0,
        local_batch_size=local_bs,
    )

    assert torch.all(scores == 0.5)


def test_mask_does_not_clobber_own_positive():
    """The labeled positive column must never be masked, even if other passages
    share its doc id."""
    local_bs, n_passages = 1, 3
    scores = torch.zeros(1, 3)
    # Col 2 shares q0's positive doc id (col 0).
    passage_doc_ids = torch.tensor([42, 99, 42], dtype=torch.long)

    mask_gathered_passages_same_doc_as_positive(
        scores,
        passage_doc_ids,
        train_n_passages=n_passages,
        rank=0,
        local_batch_size=local_bs,
    )

    assert scores[0, 0].item() == 0.0  # positive preserved
    assert scores[0, 1].item() == 0.0  # different id, untouched
    assert _is_masked(scores[0, 2])  # duplicate masked


def test_mask_respects_rank_offset_local_bs_1():
    """For rank > 0, positives live further into the gathered tensor."""
    local_bs, n_passages, rank = 1, 2, 1
    # Gathered tensor: world_size=2 * local_bs=1 * n_passages=2 = 4 cols.
    scores = torch.zeros(local_bs, 4)
    # Rank 1's positive is at col rank*local_bs*n_passages = 2.
    # Force col 0 (rank 0's positive) to share rank 1's positive doc id.
    passage_doc_ids = torch.tensor([77, 1, 77, 2], dtype=torch.long)

    mask_gathered_passages_same_doc_as_positive(
        scores,
        passage_doc_ids,
        train_n_passages=n_passages,
        rank=rank,
        local_batch_size=local_bs,
    )

    assert _is_masked(scores[0, 0])  # duplicate of rank 1's positive
    assert scores[0, 1].item() == 0.0
    assert scores[0, 2].item() == 0.0  # rank 1's own positive, preserved
    assert scores[0, 3].item() == 0.0


def test_mask_respects_rank_offset_local_bs_2():
    """rank>0 with local_bs>1 — exercises both the rank offset and the
    per-row positive-column broadcast simultaneously."""
    local_bs, n_passages, rank = 2, 2, 1
    # Gathered tensor: world_size=2 * local_bs=2 * n_passages=2 = 8 cols.
    scores = torch.zeros(local_bs, 8)
    # Layout (col index : meaning):
    #   0 r0_q0_pos | 1 r0_q0_neg | 2 r0_q1_pos | 3 r0_q1_neg
    #   4 r1_q0_pos | 5 r1_q0_neg | 6 r1_q1_pos | 7 r1_q1_neg
    # We are rank 1, so our positives live at cols 4 and 6.
    # Make col 0 share r1_q0's pos id (500); make col 7 share r1_q1's pos id (600).
    passage_doc_ids = torch.tensor(
        [500, 11, 600, 13, 500, 15, 600, 600],
        dtype=torch.long,
    )

    mask_gathered_passages_same_doc_as_positive(
        scores,
        passage_doc_ids,
        train_n_passages=n_passages,
        rank=rank,
        local_batch_size=local_bs,
    )

    # Local q0 (row 0): own positive col is 4. Col 0 shares its id 500 -> masked.
    # Cols with id 600 (2, 6, 7) do NOT share q0's id, untouched.
    assert _is_masked(scores[0, 0])
    assert scores[0, 4].item() == 0.0  # own positive preserved
    for c in (1, 2, 3, 5, 6, 7):
        assert scores[0, c].item() == 0.0

    # Local q1 (row 1): own positive col is 6. Cols 2 and 7 share id 600 -> masked.
    # Col 6 itself is the labeled positive and must stay.
    assert _is_masked(scores[1, 2])
    assert _is_masked(scores[1, 7])
    assert scores[1, 6].item() == 0.0  # own positive preserved
    for c in (0, 1, 3, 4, 5):
        assert scores[1, c].item() == 0.0
