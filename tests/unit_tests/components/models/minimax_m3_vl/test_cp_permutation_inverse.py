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

"""CPU unit tests for the MiniMax M3 CP load-balance reorder primitives.

The load-balance inverse is a silent-failure trap: a wrong inverse trains
without shape errors but never converges. These tests simulate PyTorch's
causal context-parallel sharding (the sequence is split into ``2 * cp_size``
chunks and rank ``r`` owns the pair ``{r, 2*cp_size-1-r}``), then exercise the
gather -> global-order -> local-order round-trip from
``cp_sparse_attn.order_by_positions`` / ``restore_by_positions`` -- with no
process group required.
"""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import (
    cp_load_balanced_global_slots,
    order_by_positions,
    restore_by_positions,
)


def _load_balanced_shards(seq_len: int, cp_size: int):
    """Return per-rank global-position index tensors for causal CP load balancing.

    Splits ``range(seq_len)`` into ``2 * cp_size`` equal chunks; rank ``r`` owns
    chunk ``r`` and chunk ``2*cp_size - 1 - r`` (concatenated, in that order).
    Mirrors torch.distributed.tensor.experimental context_parallel's causal
    load-balanced layout.
    """
    assert seq_len % (2 * cp_size) == 0, "seq_len must be divisible by 2*cp_size"
    chunks = torch.arange(seq_len).chunk(2 * cp_size)
    shards = []
    for r in range(cp_size):
        shards.append(torch.cat([chunks[r], chunks[2 * cp_size - 1 - r]]))
    return shards  # list[cp_size] of 1-D long tensors, each length seq_len//cp_size


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 48])
def test_positions_cover_global_range(cp_size, seq_len):
    """The simulated shards together cover 0..S-1 exactly once."""
    shards = _load_balanced_shards(seq_len, cp_size)
    assert len(shards) == cp_size
    all_pos = torch.cat(shards)
    assert all_pos.numel() == seq_len
    assert torch.equal(torch.sort(all_pos).values, torch.arange(seq_len))
    # Each shard is the load-balanced length and (for cp_size>1) non-contiguous.
    for s in shards:
        assert s.numel() == seq_len // cp_size


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("seq_len", [16, 48])
def test_order_by_positions_recovers_global(cp_size, seq_len):
    """Gather (concat rank-major) + order_by_positions -> exact global order."""
    shards = _load_balanced_shards(seq_len, cp_size)
    gathered_positions = torch.cat(shards)  # rank-major concat == all_gather+concat order

    # Build a [B, H, S, D]-shaped payload whose seq slot encodes its global position
    # so we can assert ordering by value. Gather it in the SAME rank-major order.
    B, H, D = 2, 3, 4
    full = torch.arange(seq_len).view(1, 1, seq_len, 1).expand(B, H, seq_len, D).float()
    gathered = full.index_select(2, gathered_positions)  # as if all-gathered from ranks

    global_tensor, sort_order = order_by_positions(gathered, gathered_positions, seq_dim=2)

    # global_tensor[..., p, :] must equal global position p
    expected = torch.arange(seq_len).view(1, 1, seq_len, 1).expand(B, H, seq_len, D).float()
    assert torch.equal(global_tensor, expected)
    # sort_order is argsort of the gathered positions
    assert torch.equal(gathered_positions.index_select(0, sort_order), torch.arange(seq_len))


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("seq_len", [16, 48])
def test_round_trip_global_to_local_is_identity(cp_size, seq_len):
    """For every rank, restore_by_positions(global, local_positions) == the rank's shard."""
    shards = _load_balanced_shards(seq_len, cp_size)
    gathered_positions = torch.cat(shards)

    B, H, D = 1, 2, 5
    full = torch.arange(seq_len).view(1, 1, seq_len, 1).expand(B, H, seq_len, D).float()
    gathered = full.index_select(2, gathered_positions)
    global_tensor, _ = order_by_positions(gathered, gathered_positions, seq_dim=2)

    for r, local_positions in enumerate(shards):
        local = restore_by_positions(global_tensor, local_positions, seq_dim=2)
        expected_local = full.index_select(2, local_positions)
        assert torch.equal(local, expected_local), f"rank {r} round-trip mismatch"


def test_order_by_positions_rejects_non_dense_positions():
    """A position set that is not a permutation of 0..S-1 is rejected loudly."""
    gathered = torch.zeros(1, 1, 4, 2)
    bad_positions = torch.tensor([0, 1, 2, 5])  # gap -> not dense
    with pytest.raises(ValueError, match="dense global token positions"):
        order_by_positions(gathered, bad_positions, seq_dim=2)


def test_order_by_positions_requires_1d_positions():
    gathered = torch.zeros(1, 1, 4, 2)
    with pytest.raises(ValueError, match="1-D"):
        order_by_positions(gathered, torch.zeros(1, 4), seq_dim=2)


def test_restore_with_full_global_positions_is_noop():
    """Restoring with the identity position vector returns the input unchanged."""
    x = torch.randn(2, 3, 8, 4)
    ident = torch.arange(8)
    assert torch.equal(restore_by_positions(x, ident, seq_dim=2), x)


# ---------------------------------------------------------------------------
# Structural global-slot reconstruction (the production reorder used by the CP
# sparse forward; independent of position_id values -> robust to cp-padding).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("t_local", [8, 16])
def test_global_slots_are_a_permutation(cp_size, t_local):
    dev = torch.device("cpu")
    gathered = cp_load_balanced_global_slots(cp_size, t_local, dev)
    t_global = cp_size * t_local
    assert gathered.numel() == t_global
    assert torch.equal(torch.sort(gathered).values, torch.arange(t_global))


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("t_local", [8, 16])
def test_per_rank_slots_concat_equals_gathered(cp_size, t_local):
    dev = torch.device("cpu")
    gathered = cp_load_balanced_global_slots(cp_size, t_local, dev)
    per_rank = torch.cat([cp_load_balanced_global_slots(cp_size, t_local, dev, rank=r) for r in range(cp_size)])
    assert torch.equal(gathered, per_rank)


def test_global_slots_match_zigzag_layout():
    """cp_size=2 must match the documented {r, 2*cp-1-r} chunk pairing."""
    # T_global=16, cp=2 -> 4 chunks of 4: chunk0=[0..3] chunk1=[4..7] chunk2=[8..11] chunk3=[12..15]
    # rank0 owns chunks {0,3}; rank1 owns chunks {1,2}.
    dev = torch.device("cpu")
    r0 = cp_load_balanced_global_slots(2, 8, dev, rank=0)
    r1 = cp_load_balanced_global_slots(2, 8, dev, rank=1)
    assert r0.tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert r1.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]


def test_global_slots_match_load_balanced_shards_helper():
    """The structural slots agree with the independent _load_balanced_shards simulation."""
    for cp_size, seq_len in [(2, 16), (4, 48)]:
        t_local = seq_len // cp_size
        shards = _load_balanced_shards(seq_len, cp_size)
        for r in range(cp_size):
            slots = cp_load_balanced_global_slots(cp_size, t_local, torch.device("cpu"), rank=r)
            assert torch.equal(slots, shards[r]), f"cp_size={cp_size} rank={r} mismatch"
