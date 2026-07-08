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

"""Unit tests for the CPSharder contract in components/distributed/cp_sharder.py.

Collectives are not exercised here (CPU CI): the tests cover the pure layout
math — local index generation, index-based token-tensor shard, gathered-shard
reordering — and the CPSharder default/override resolution.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from nemo_automodel.components.distributed import cp_sharder as cs


class _FakeMesh:
    """Minimal stand-in for a CP device-mesh slice (no real process group)."""

    def __init__(self, size: int, local_rank: int = 0):
        self._size = size
        self._local_rank = local_rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_group(self):
        return None


@pytest.fixture(autouse=True)
def _force_no_dist(monkeypatch):
    """Pin rank resolution to the fake mesh's local rank."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


# ---------------------------------------------------------------------------
# contiguous_local_indices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size,rank", [(1, 0), (2, 0), (2, 1), (4, 2)])
def test_contiguous_local_indices(cp_size, rank):
    indices = cs.contiguous_local_indices(_FakeMesh(cp_size, rank), 8 * cp_size)
    assert torch.equal(indices, torch.arange(rank * 8, (rank + 1) * 8))


def test_contiguous_local_indices_requires_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        cs.contiguous_local_indices(_FakeMesh(4, 0), 6)


# ---------------------------------------------------------------------------
# shard <-> gather round trip (pure permutation math, no collectives)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [2, 4])
def test_shard_then_reorder_roundtrip_contiguous(cp_size):
    seq = 8 * cp_size
    full = torch.randn(2, seq)
    parts, index_parts = [], []
    for rank in range(cp_size):
        idx = cs.contiguous_local_indices(_FakeMesh(cp_size, rank), seq)
        parts.append(cs.shard_token_tensor_by_indices(full, idx, seq_dim=1))
        index_parts.append(idx)
    rebuilt = cs._reorder_gathered_token_tensor(parts, index_parts, seq_dim=1)
    torch.testing.assert_close(rebuilt, full)


def test_shard_then_reorder_roundtrip_round_robin():
    # A non-contiguous layout (torch CP's 2*cp round-robin chunks) must also
    # round-trip through the same index-based verbs: the indices tensor is the
    # universal layout coordinate system.
    cp_size, chunk = 2, 4
    seq = 2 * cp_size * chunk
    chunks = torch.arange(seq).view(2 * cp_size, chunk)
    # torch CP load balancing: rank r takes chunks (r, 2*cp - r - 1)
    rank_indices = [torch.cat([chunks[r], chunks[2 * cp_size - r - 1]]) for r in range(cp_size)]
    full = torch.randn(1, seq, 3)
    parts = [cs.shard_token_tensor_by_indices(full, idx, seq_dim=1) for idx in rank_indices]
    rebuilt = cs._reorder_gathered_token_tensor(parts, rank_indices, seq_dim=1)
    torch.testing.assert_close(rebuilt, full)


def test_gather_token_tensor_identity_without_cp():
    t = torch.randn(1, 6)
    assert cs.gather_token_tensor_by_indices(_FakeMesh(1), t, torch.arange(6)) is t
    assert cs.gather_token_tensor_by_indices(None, t, torch.arange(6)) is t


# ---------------------------------------------------------------------------
# CPSharder default/override resolution
# ---------------------------------------------------------------------------
def test_sharder_default_shard_token_tensor_uses_indices():
    sharder = cs.CPSharder(
        shard_batch=lambda *a, **k: (contextlib.nullcontext, {}),
        local_token_global_indices=cs.contiguous_local_indices,
        layout="contiguous",
    )
    full = torch.randn(1, 8)
    local = sharder.shard_token_tensor(_FakeMesh(2, 1), full, seq_dim=1)
    torch.testing.assert_close(local, full[:, 4:])


# ---------------------------------------------------------------------------
# shard_batch_contiguous: pad_multiple + packed-seq-ids flags
# ---------------------------------------------------------------------------
def test_shard_batch_contiguous_respects_pad_multiple():
    seq = 6  # divisor = cp_size * max(pad_multiple, 2) = 2 * 4 = 8 -> pad to 8
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
    }
    ctx, out = cs.shard_batch_contiguous(
        _FakeMesh(2, 1),
        None,
        batch,
        padding_token_id=9,
        pad_multiple=4,
    )
    assert ctx is contextlib.nullcontext
    assert out["input_ids"].shape == (1, 4)
    # rank 1 slice covers the pad tail: input pad sentinel + label ignore index
    assert out["input_ids"][0, -1].item() == 9
    assert out["labels"][0, -1].item() == -100
    assert "_packed_seq_ids" not in out


def test_shard_batch_contiguous_extra_seq_keys():
    seq = 8
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "vision_ids": torch.arange(seq).view(1, seq),
    }
    _, out = cs.shard_batch_contiguous(
        _FakeMesh(2, 0),
        None,
        batch,
        extra_seq_keys={"vision_ids": 1},
        extra_pad_values={"vision_ids": -1},
    )
    assert out["vision_ids"].shape == (1, 4)
    torch.testing.assert_close(out["vision_ids"], torch.arange(4).view(1, 4))
    # model-specific keys (e.g. _packed_seq_ids) are the owning model's business
    assert "_packed_seq_ids" not in out
