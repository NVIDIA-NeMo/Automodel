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

"""Unit tests for the ContextParallelismSharder contract in components/distributed/cp_sharder.py.

Collectives are not exercised here (CPU CI): the tests cover the pure layout
math — local index generation, index-based token-tensor shard, gathered-shard
reordering — and the ContextParallelismSharder default/override resolution.
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
# round_robin_local_indices (torch context_parallel load-balanced layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [2, 4])
def test_round_robin_local_indices_matches_torch_chunk_pairing(cp_size):
    chunk = 3
    seq = 2 * cp_size * chunk
    chunks = torch.arange(seq).view(2 * cp_size, chunk)
    for rank in range(cp_size):
        expected = torch.cat([chunks[rank], chunks[2 * cp_size - 1 - rank]])
        indices = cs.round_robin_local_indices(_FakeMesh(cp_size, rank), seq)
        assert torch.equal(indices, expected)


def test_round_robin_local_indices_partition_the_sequence():
    cp_size, seq = 4, 24
    all_indices = torch.cat([cs.round_robin_local_indices(_FakeMesh(cp_size, r), seq) for r in range(cp_size)])
    assert torch.equal(torch.sort(all_indices).values, torch.arange(seq))


def test_round_robin_local_indices_requires_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        cs.round_robin_local_indices(_FakeMesh(2, 0), 6)


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
# ContextParallelismSharder default/override resolution
# ---------------------------------------------------------------------------
def test_sharder_default_shard_token_tensor_uses_indices():
    sharder = cs.ContextParallelismSharder(
        shard_batch=lambda *a, **k: (contextlib.nullcontext, {}, None),
        local_token_global_indices=cs.contiguous_local_indices,
    )
    full = torch.randn(1, 8)
    local = sharder.shard_token_tensor(_FakeMesh(2, 1), full, seq_dim=1)
    torch.testing.assert_close(local, full[:, 4:])


def test_shard_batch_contiguous_records_shard_layout():
    """shard_batch reports original/padded lengths as ShardLayout; once installed,
    the token verbs accept caller-coordinate tensors and reject mismatched ones."""
    sharder = cs.ContextParallelismSharder(
        shard_batch=None,
        local_token_global_indices=cs.contiguous_local_indices,
    )
    mesh = _FakeMesh(2, 0)
    batch = {"input_ids": torch.arange(6).unsqueeze(0), "labels": torch.arange(6).unsqueeze(0)}
    _, _, layout = cs.shard_batch_contiguous(mesh, None, batch)  # pads 6 -> 8
    sharder.shard_layout = layout

    assert (sharder.shard_layout.original_seq_len, sharder.shard_layout.padded_seq_len) == (6, 8)
    # down: unpadded tensor auto-pads with the explicit fill, rank 0 owns [0:4]
    local = sharder.shard_token_tensor(mesh, torch.arange(6.0).unsqueeze(0), fill=-1.0)
    assert torch.equal(local, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
    # the pad_multiple silent-misalignment window is closed: a plausible but
    # wrong length raises even though it divides cp_size
    with pytest.raises(ValueError, match="padded_seq_len=8"):
        sharder.shard_token_tensor(mesh, torch.zeros(1, 4))
    # up: trim validates the gathered length against the captured layout (no
    # collective runs in this single-process test, so the gather stays local
    # and the guard must fire rather than mis-trim)
    with pytest.raises(ValueError, match="reported padded_seq_len 8"):
        sharder.gather_token_tensor(mesh, torch.zeros(1, 4), trim=True)


def test_sharder_repositioned_layout_round_trips_input_coordinates():
    """A captured position map (DSV4 packed repad) lets both verbs work in the
    caller's [B, S_in] coordinates: real tokens land on their repositioned
    columns and dropped input pad slots come back as fill."""
    # input row: [a, b, PAD] -> rebuilt row: [a, b, X, X] (doc re-padded to 4)
    positions = torch.tensor([[0, 1, -1]])
    sharder = cs.ContextParallelismSharder(
        shard_batch=None,
        local_token_global_indices=cs.contiguous_local_indices,
        shard_layout=cs.ShardLayout(padded_seq_len=4, input_token_stream_positions=positions),
    )
    mesh = _FakeMesh(2, 0)  # rank 0 owns columns [0:2]
    local = sharder.shard_token_tensor(mesh, torch.tensor([[10.0, 20.0, 99.0]]), fill=0.0)
    assert torch.equal(local, torch.tensor([[10.0, 20.0]]))
    with pytest.raises(ValueError, match="fill"):
        sharder.shard_token_tensor(mesh, torch.tensor([[10.0, 20.0, 99.0]]))

    # up: gather (identity at cp<=1 here) then map back to input coordinates
    full_rows = torch.tensor([[10.0, 20.0, 7.0, 7.0]])
    out = sharder.gather_token_tensor(_FakeMesh(1), full_rows, trim=True, fill=-5.0)
    assert torch.equal(out, torch.tensor([[10.0, 20.0, -5.0]]))
    with pytest.raises(ValueError, match="fill"):
        sharder.gather_token_tensor(_FakeMesh(1), full_rows, trim=True)


def test_gather_trim_raises_without_captured_facts():
    sharder = cs.ContextParallelismSharder(
        shard_batch=None,
        local_token_global_indices=cs.contiguous_local_indices,
    )
    with pytest.raises(NotImplementedError, match="no shard layout to trim to"):
        sharder.gather_token_tensor(_FakeMesh(1), torch.zeros(1, 4), trim=True)


def test_reported_indices_validate_stream_length():
    # Reported index maps flatten + cast to long, and reject a padded_seq_len
    # that does not match the partition the shard reported.
    sharder = cs.ContextParallelismSharder(
        shard_batch=lambda *a, **k: (contextlib.nullcontext, {}, None),
        local_token_global_indices=None,
        shard_layout=cs.ShardLayout(local_token_global_indices=torch.tensor([[1, 0]], dtype=torch.int32)),
    )
    assert torch.equal(sharder._indices(_FakeMesh(2, 0), 4, None), torch.tensor([1, 0]))
    assert torch.equal(sharder._indices(None, 2, None), torch.tensor([1, 0]))  # no mesh -> cp_size 1
    with pytest.raises(ValueError, match="does not match"):
        sharder._indices(_FakeMesh(2, 0), 6, None)


def test_sharder_token_verbs_unavailable_for_data_dependent_layouts():
    # THD/magi layouts depend on batch content (cu_seqlens / dispatch solver),
    # so their framework sharders carry no index map and the token-tensor verbs
    # must fail loudly rather than shard the wrong slice.
    sharder = cs.ContextParallelismSharder(
        shard_batch=lambda *a, **k: (contextlib.nullcontext, {}, None),
        local_token_global_indices=None,
    )
    with pytest.raises(NotImplementedError, match="data-dependent"):
        sharder.shard_token_tensor(_FakeMesh(2, 0), torch.randn(1, 8))
    with pytest.raises(NotImplementedError, match="data-dependent"):
        sharder.gather_token_tensor(_FakeMesh(2, 0), torch.randn(1, 4))


# ---------------------------------------------------------------------------
# shard_batch_contiguous: pad_multiple + packed-seq-ids flags
# ---------------------------------------------------------------------------
def test_shard_batch_contiguous_respects_pad_multiple():
    seq = 6  # divisor = cp_size * max(pad_multiple, 2) = 2 * 4 = 8 -> pad to 8
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
    }
    ctx, out, _ = cs.shard_batch_contiguous(
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


# ---------------------------------------------------------------------------
# shard_sequence_for_cp (in-forward round-robin shard, differentiable)
# ---------------------------------------------------------------------------
def test_shard_sequence_for_cp_roundtrips_across_ranks():
    """Sharding all ranks and reassembling reproduces the full padded sequence;
    the appended CP-pad slots are zero (pad_value=0)."""
    cp_size, seq = 2, 6  # divisor 2*cp = 4 -> pad to 8
    full = torch.randn(1, seq, 4)
    parts, idxs = [], []
    for rank in range(cp_size):
        local, idx, padded_len = cs.shard_sequence_for_cp(_FakeMesh(cp_size, rank), full.clone(), pad_value=0.0)
        assert padded_len == 8
        assert local.shape == (1, 4, 4)  # 8 / cp_size
        parts.append(local)
        idxs.append(idx)
    rebuilt = cs._reorder_gathered_token_tensor(parts, idxs, seq_dim=1)
    torch.testing.assert_close(rebuilt[:, :seq], full)
    torch.testing.assert_close(rebuilt[:, seq:], torch.zeros(1, 2, 4))


def test_shard_sequence_for_cp_matches_round_robin_layout():
    """The kept positions are exactly this rank's round_robin_local_indices."""
    full = torch.arange(8).view(1, 8, 1).float()
    for rank in (0, 1):
        local, idx, _ = cs.shard_sequence_for_cp(_FakeMesh(2, rank), full.clone())
        expected_idx = cs.round_robin_local_indices(_FakeMesh(2, rank), 8)
        assert torch.equal(idx, expected_idx)
        torch.testing.assert_close(local, full.index_select(1, expected_idx))


def test_shard_sequence_for_cp_is_differentiable():
    """Gradient reaches exactly the input positions this rank owns."""
    full = torch.randn(1, 8, 3, requires_grad=True)
    local, idx, _ = cs.shard_sequence_for_cp(_FakeMesh(2, 0), full)
    local.sum().backward()
    touched = (full.grad.abs().sum(dim=(0, 2)) > 0).nonzero().flatten()
    assert torch.equal(touched, idx.sort().values)


def test_shard_sequence_for_cp_identity_without_cp():
    full = torch.randn(1, 5, 2)
    local, idx, padded_len = cs.shard_sequence_for_cp(_FakeMesh(1), full)
    assert local is full and padded_len == 5
    assert torch.equal(idx, torch.arange(5))


# ---------------------------------------------------------------------------
# shard_batch_aux_only vs shard_batch_load_balanced (parity on the aux streams)
# ---------------------------------------------------------------------------
def test_shard_batch_aux_only_matches_load_balanced(monkeypatch):
    """The aux-only shard pads labels/position_ids/loss_mask identically to the
    load-balanced shard but leaves the primary stream full-length and out of the
    CP buffer list."""
    from nemo_automodel.components.distributed import cp_utils

    captured: dict = {}

    def _fake_ctx(cp_mesh, cp_buffers, cp_seq_dims, cp_no_restore_buffers, cp_rotate_method=None):
        captured["buffers"] = list(cp_buffers)
        return contextlib.nullcontext()

    monkeypatch.setattr(cp_utils, "create_context_parallel_ctx", _fake_ctx)
    monkeypatch.setattr(cp_utils, "get_train_context", lambda *a, **k: contextlib.nullcontext)

    cp_size, seq = 2, 6  # -> pad to 8
    mesh = _FakeMesh(cp_size, 0)

    def make_batch():
        return {
            "input_ids": torch.arange(seq).view(1, seq),
            "labels": torch.arange(100, 100 + seq).view(1, seq),
        }

    batch_lb = make_batch()
    cs.shard_batch_load_balanced(mesh, None, batch_lb, loss_mask=torch.ones(1, seq))
    buffers_lb = captured["buffers"]

    batch_aux = make_batch()
    orig_input_ids = batch_aux["input_ids"].clone()
    cs.shard_batch_aux_only(mesh, None, batch_aux, loss_mask=torch.ones(1, seq))
    buffers_aux = captured["buffers"]

    # aux tensors padded identically (labels -> -100 tail, position_ids -> 0 tail)
    torch.testing.assert_close(batch_lb["labels"], batch_aux["labels"])
    torch.testing.assert_close(batch_lb["position_ids"], batch_aux["position_ids"])
    assert batch_aux["labels"].shape == (1, 8)
    assert batch_aux["labels"][0, -1].item() == -100
    # loss_mask is the last buffer in both lists; padded to 8 the same way
    torch.testing.assert_close(buffers_lb[-1], buffers_aux[-1])
    assert buffers_aux[-1].shape == (1, 8)
    # primary stays full-length + untouched in aux-only; load-balanced pads it
    torch.testing.assert_close(batch_aux["input_ids"], orig_input_ids)
    assert batch_lb["input_ids"].shape == (1, 8)
    # aux-only excludes the primary from the CP buffer list (one fewer buffer)
    assert len(buffers_aux) == len(buffers_lb) - 1
    assert all(buf.shape[1] == 8 for buf in buffers_aux)


def test_shard_batch_aux_only_reports_padded_layout(monkeypatch):
    """The returned ShardLayout carries the primary stream's target padded length."""
    from nemo_automodel.components.distributed import cp_utils

    monkeypatch.setattr(cp_utils, "create_context_parallel_ctx", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(cp_utils, "get_train_context", lambda *a, **k: contextlib.nullcontext)

    batch = {"input_ids": torch.arange(6).view(1, 6), "labels": torch.arange(6).view(1, 6)}
    _, _, layout = cs.shard_batch_aux_only(_FakeMesh(2, 0), None, batch)
    assert (layout.original_seq_len, layout.padded_seq_len) == (6, 8)


def test_shard_batch_contiguous_extra_seq_keys():
    seq = 8
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "vision_ids": torch.arange(seq).view(1, seq),
    }
    _, out, _ = cs.shard_batch_contiguous(
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
