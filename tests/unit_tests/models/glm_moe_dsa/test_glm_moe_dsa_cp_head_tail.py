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

"""Head/tail (round-robin) context-parallel query layout for GLM DSA: batch slicing, gather order, cuDNN metadata."""

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel.sharder import (
    contiguous_local_indices,
    round_robin_local_indices,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import cp as glm_cp
from nemo_automodel.components.models.glm_moe_dsa.kernels.cudnn_dsa import prepare_cudnn_dsa_packed_metadata
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import is_cudnn_dsa_available
from nemo_automodel.shared.recompute_replay import RecomputeReplay, RecomputeReplayRecorder

_CPU = torch.device("cpu")


class _FakeMesh:
    def __init__(self, size: int = 2, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_group(self) -> str:
        return "cp-group"

    def get_local_rank(self) -> int:
        return self._rank


def _thd_chunk(num_tokens: int = 8) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(num_tokens),
        "labels": torch.arange(num_tokens) + 100,
        "position_ids": torch.arange(num_tokens) + 200,
        "cu_seqlens": torch.tensor([0, num_tokens], dtype=torch.int64),
        "padding_mask": torch.zeros(num_tokens, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(0, [0, 1, 6, 7]), (1, [2, 3, 4, 5])],
    ids=["rank0_head_and_tail", "rank1_middle"],
)
def test_head_tail_slice_keeps_rank_chunks_and_global_metadata(rank, expected):
    out = glm_cp._slice_thd_chunk_for_cp(
        _thd_chunk(8),
        cp_mesh=_FakeMesh(rank=rank),
        cp_group="cp-group",
        cp_size=2,
        cp_rank=rank,
        padding_token_id=0,
        query_layout="head_tail",
    )
    assert out["glm_dsa_cp_query_indices"].tolist() == expected
    assert out["input_ids"].tolist() == expected
    assert out["labels"].tolist() == [100 + i for i in expected]
    assert out["position_ids"].tolist() == [200 + i for i in expected]
    assert out["cu_seqlens"].tolist() == [0, 8]  # global, unchanged
    contiguous = glm_cp._slice_thd_chunk_for_cp(
        _thd_chunk(8), cp_mesh=_FakeMesh(rank=rank), cp_group="cp-group", cp_size=2, cp_rank=rank, padding_token_id=0
    )
    assert contiguous["glm_dsa_cp_query_indices"].tolist() == list(range(4 * rank, 4 * rank + 4))


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="query layout"):
        glm_cp.local_query_indices("zigzag", _FakeMesh(), 8, _CPU)
    with pytest.raises(ValueError, match="query layout"):
        glm_cp.layout_indices_for_rank("zigzag", 0, 2, 8, _CPU)
    with pytest.raises(ValueError, match="query layout"):
        glm_cp.gather_order("zigzag", 2, 8, _CPU)
    with pytest.raises(ValueError, match="cp_query_layout"):
        BackendConfig(attn="cudnn", cp_query_layout="zigzag")


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4, 8])
def test_layout_indices_for_rank_match_the_shared_sharder_helpers(cp_size):
    total = 2 * cp_size * 5
    for rank in range(cp_size):
        mesh = _FakeMesh(size=cp_size, rank=rank)
        assert torch.equal(
            glm_cp.layout_indices_for_rank("head_tail", rank, cp_size, total, _CPU),
            round_robin_local_indices(mesh, total, _CPU),
        )
        assert torch.equal(
            glm_cp.layout_indices_for_rank("contiguous", rank, cp_size, total, _CPU),
            contiguous_local_indices(mesh, total, _CPU),
        )
    with pytest.raises(ValueError, match="2 \\* cp_size"):
        glm_cp.layout_indices_for_rank("head_tail", 0, cp_size, 2 * cp_size * 5 + 1, _CPU)
    with pytest.raises(ValueError, match="rank"):
        glm_cp.layout_indices_for_rank("head_tail", cp_size, cp_size, total, _CPU)


@pytest.mark.parametrize("cp_size", [2, 3, 4, 8])
def test_gather_order_inverts_the_rank_order_concatenation(cp_size):
    total = 2 * cp_size * 3
    from_global, to_global = glm_cp.gather_order("head_tail", cp_size, total, _CPU)
    rank_order = torch.cat(
        [glm_cp.layout_indices_for_rank("head_tail", rank, cp_size, total, _CPU) for rank in range(cp_size)]
    )
    assert torch.equal(from_global, rank_order)
    everything = torch.arange(total)
    assert torch.equal(from_global[to_global], everything)
    assert torch.equal(to_global[from_global], everything)
    # Rank order is not global order for any head/tail configuration with cp_size >= 2.
    assert not torch.equal(from_global, everything)


def test_gather_order_is_identity_for_contiguous_and_single_rank():
    assert glm_cp.gather_order("contiguous", 4, 24, _CPU) is None
    assert glm_cp.gather_order("head_tail", 1, 24, _CPU) is None
    with pytest.raises(ValueError, match="2 \\* cp_size"):
        glm_cp.gather_order("head_tail", 2, 6, _CPU)


def test_validate_gather_order_rejects_a_batch_sliced_with_another_layout():
    order = glm_cp.gather_order("head_tail", 2, 8, _CPU)
    glm_cp.validate_gather_order(order, 0, 4, torch.tensor([0, 1, 6, 7], dtype=torch.int32))
    glm_cp.validate_gather_order(order, 1, 4, torch.tensor([2, 3, 4, 5], dtype=torch.int32))
    with pytest.raises(ValueError, match="cp_query_layout"):
        # A contiguous slice handed to a head/tail model.
        glm_cp.validate_gather_order(order, 1, 4, torch.tensor([4, 5, 6, 7], dtype=torch.int32))
    with pytest.raises(ValueError, match="cp_query_layout"):
        glm_cp.validate_gather_order(order, 0, 4, torch.tensor([0, 1, 6], dtype=torch.int32))


# Three packed documents whose real lengths differ from their padded storage: pad rows at 7, 18, 19 and 23.
_REAL_CU = torch.tensor([0, 7, 17, 20], dtype=torch.int32)
_PADDED_CU = torch.tensor([0, 8, 20, 24], dtype=torch.int32)
_PAD_ROWS = torch.tensor([7, 18, 19, 23])


def _rank_order_gather(layout: str, cp_size: int, total: int, rows: torch.Tensor) -> torch.Tensor:
    """What ``all_gather_tensor_autograd`` returns: every rank's local rows, rank after rank."""
    return torch.cat(
        [
            rows.index_select(0, glm_cp.layout_indices_for_rank(layout, rank, cp_size, total, _CPU))
            for rank in range(cp_size)
        ]
    )


@pytest.mark.parametrize(("layout", "cp_size"), [("contiguous", 2), ("head_tail", 2), ("head_tail", 4)])
def test_gathered_keys_match_the_global_coordinates_the_kernels_use(layout, cp_size):
    total = int(_PADDED_CU[-1])
    padding_mask = torch.zeros(total, dtype=torch.bool)
    padding_mask[_PAD_ROWS] = True
    identity = torch.arange(total)  # key row i carries its own global coordinate
    gathered = _rank_order_gather(layout, cp_size, total, identity)
    order = glm_cp.gather_order(layout, cp_size, total, _CPU)
    kernel_view = gathered if order is None else gathered.index_select(0, order[1])
    assert torch.equal(kernel_view, identity)

    for rank in range(cp_size):
        local = glm_cp.layout_indices_for_rank(layout, rank, cp_size, total, _CPU)
        meta = prepare_cudnn_dsa_packed_metadata(
            _REAL_CU,
            total,
            query_indices=local.to(torch.int32),
            cu_seqlens_padded=_PADDED_CU,
            padding_mask=padding_mask.index_select(0, local),
        )
        sources = meta.key_source_indices
        assert sources is not None
        # The indexer feeds ``index_k.index_select(0, key_source_indices)`` to the kernel: every row must be
        # the global token the segment metadata means.
        assert torch.equal(kernel_view.index_select(0, sources), identity.index_select(0, sources))
        # Padding rows are invalid queries; real rows keep their causal prefix inside their own document.
        assert torch.equal(meta.query_valid, ~padding_mask.index_select(0, local))
        if layout == "head_tail":
            # Without the reorder, the rank-order gather hands the kernel other ranks' tokens.
            assert not torch.equal(gathered.index_select(0, sources), identity.index_select(0, sources))


def test_packed_metadata_splits_head_and_tail_runs_into_segments():
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    # Rank 0 of CP2 with the head/tail layout: queries 0,1 and 6,7 of one 8-token document.
    meta = prepare_cudnn_dsa_packed_metadata(cu_seqlens, 8, query_indices=torch.tensor([0, 1, 6, 7], dtype=torch.int32))
    assert meta.segment_cu_q.tolist() == [0, 2, 4]
    # Each run pairs with the document's key prefix up to its last query: 2 keys, then 8 keys.
    assert meta.segment_cu_k.tolist() == [0, 2, 10]
    assert meta.q_causal_offsets.tolist() == [0, 6]
    assert meta.causal_lengths.tolist() == [1, 2, 7, 8]
    assert meta.starts.tolist() == [0, 0, 0, 0]
    assert meta.max_seqlen_q == 2 and meta.max_seqlen_k == 8
    assert meta.key_source_indices.tolist() == [0, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    assert meta.all_rows_nonempty and meta.valid_row_indices is None

    middle = prepare_cudnn_dsa_packed_metadata(
        cu_seqlens, 8, query_indices=torch.tensor([2, 3, 4, 5], dtype=torch.int32)
    )
    assert middle.segment_cu_q.tolist() == [0, 4]
    assert middle.segment_cu_k.tolist() == [0, 6]
    assert middle.q_causal_offsets.tolist() == [2]
    assert middle.causal_lengths.tolist() == [3, 4, 5, 6]


def test_packed_metadata_head_tail_across_two_documents():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    meta = prepare_cudnn_dsa_packed_metadata(cu_seqlens, 8, query_indices=torch.tensor([0, 1, 6, 7], dtype=torch.int32))
    assert meta.segment_cu_q.tolist() == [0, 2, 4]
    assert meta.segment_cu_k.tolist() == [0, 2, 6]  # doc 0 keys 0..1, doc 1 keys 4..7
    assert meta.q_causal_offsets.tolist() == [0, 2]
    assert meta.causal_lengths.tolist() == [1, 2, 3, 4]
    assert meta.starts.tolist() == [0, 0, 4, 4]
    assert meta.key_source_indices.tolist() == [0, 1, 4, 5, 6, 7]


def test_packed_metadata_requires_strictly_increasing_queries():
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_cudnn_dsa_packed_metadata(cu_seqlens, 8, query_indices=torch.tensor([6, 7, 0, 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_cudnn_dsa_packed_metadata(cu_seqlens, 8, query_indices=torch.tensor([0, 1, 1, 2], dtype=torch.int32))


def _fake_rank_order_all_gather(global_rows: torch.Tensor, layout: str, cp_size: int):
    """A stand-in for ``all_gather_tensor_autograd``: rank 0 contributes the live tensor, the others constants."""
    total = global_rows.shape[0]

    def fake(received, gather_dim, group):
        shards = [
            received
            if rank == 0
            else global_rows.index_select(0, glm_cp.layout_indices_for_rank(layout, rank, cp_size, total, _CPU))
            for rank in range(cp_size)
        ]
        return torch.cat(shards, dim=gather_dim)

    return fake


def test_glm_dsa_cp_all_gather_returns_head_tail_shards_in_global_order(monkeypatch):
    cp_size, total = 2, 8
    global_rows = torch.arange(total, dtype=torch.float32).view(total, 1).repeat(1, 3) + torch.tensor([0.0, 0.1, 0.2])
    monkeypatch.setattr(glm_cp, "glm_dsa_cp_enabled", lambda cp_group: True)
    monkeypatch.setattr(
        glm_cp.funcol, "all_gather_tensor_autograd", _fake_rank_order_all_gather(global_rows, "head_tail", cp_size)
    )
    rank0 = glm_cp.layout_indices_for_rank("head_tail", 0, cp_size, total, _CPU)
    local = global_rows.index_select(0, rank0).clone().requires_grad_()
    order = glm_cp.gather_order("head_tail", cp_size, total, _CPU)

    out = glm_cp.glm_dsa_cp_all_gather(local, dim=0, cp_group="cp-group", order=order)
    torch.testing.assert_close(out, global_rows)
    weights = torch.arange(1.0, total * 3 + 1).view(total, 3)
    (out * weights).sum().backward()
    # Every global row appears exactly once, so rank 0's rows receive exactly their own weights.
    torch.testing.assert_close(local.grad, weights.index_select(0, rank0))

    # Without the order the collective's rank order comes back unchanged (the contiguous-layout contract).
    plain = glm_cp.glm_dsa_cp_all_gather(local.detach(), dim=0, cp_group="cp-group")
    torch.testing.assert_close(plain, global_rows.index_select(0, order[0]))
    assert not torch.equal(plain, global_rows)


def test_replayed_head_tail_gather_permutes_the_gradient_back_before_the_reduce_scatter(monkeypatch):
    cp_size, total = 2, 8
    global_rows = torch.arange(total, dtype=torch.float32).view(total, 1).repeat(1, 2)
    monkeypatch.setattr(glm_cp, "glm_dsa_cp_enabled", lambda cp_group: True)
    monkeypatch.setattr(
        glm_cp.funcol, "all_gather_tensor_autograd", _fake_rank_order_all_gather(global_rows, "head_tail", cp_size)
    )
    scattered: list[torch.Tensor] = []

    def fake_reduce_scatter_tensor(grad, op, scatter_dim, group):
        assert (op, scatter_dim, group) == ("sum", 0, "cp-group")
        scattered.append(grad.clone())
        head, tail = grad.chunk(cp_size, dim=scatter_dim)
        return head + tail

    monkeypatch.setattr(glm_cp.funcol, "reduce_scatter_tensor", fake_reduce_scatter_tensor)
    order = glm_cp.gather_order("head_tail", cp_size, total, _CPU)
    from_global, _ = order
    rank0 = glm_cp.layout_indices_for_rank("head_tail", 0, cp_size, total, _CPU)
    replay = RecomputeReplay("test gather")
    recorder = RecomputeReplayRecorder()

    local = global_rows.index_select(0, rank0).clone().requires_grad_()
    with replay.scope(recorder, "record"):
        recorded = glm_cp.glm_dsa_cp_all_gather(local, dim=0, cp_group="cp-group", replay=replay, order=order)
    torch.testing.assert_close(recorded, global_rows)
    assert len(recorder) == 1 and torch.equal(recorder.records[0], global_rows)

    recorder.rewind()
    replayed_local = global_rows.index_select(0, rank0).clone().requires_grad_()
    with replay.scope(recorder, "replay"):
        out = glm_cp.glm_dsa_cp_all_gather(replayed_local, dim=0, cp_group="cp-group", replay=replay, order=order)
    torch.testing.assert_close(out, global_rows)

    weights = torch.arange(1.0, total * 2 + 1).view(total, 2)
    (out * weights).sum().backward()
    # The reduce-scatter sees the gradient in the collective's rank order, not in global order.
    assert len(scattered) == 1
    torch.testing.assert_close(scattered[0], weights.index_select(0, from_global))
    rank_order_weights = weights.index_select(0, from_global)
    torch.testing.assert_close(replayed_local.grad, rank_order_weights[: total // 2] + rank_order_weights[total // 2 :])


@pytest.mark.skipif(not torch.cuda.is_available() or not is_cudnn_dsa_available(), reason="cuDNN DSA kernels")
def test_cudnn_indexer_head_tail_queries_select_the_same_keys_as_the_full_sequence():
    from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import cudnn_indexer_topk

    device = torch.device("cuda")
    tokens, heads, dim, topk = 1024, 32, 128, 2048
    torch.manual_seed(0)
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, heads, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 512, tokens], dtype=torch.int32, device=device)
    full = cudnn_indexer_topk(q, k, weights, cu_seqlens, topk)

    cp_size = 2
    # The keys arrive through the rank-order all-gather; the model reorders them into global order first.
    from_global, to_global = glm_cp.gather_order("head_tail", cp_size, tokens, device)
    k_rank_order = k.index_select(0, from_global)
    for rank in range(cp_size):
        local = glm_cp.layout_indices_for_rank("head_tail", rank, cp_size, tokens, device).to(torch.int32)
        meta = prepare_cudnn_dsa_packed_metadata(cu_seqlens, tokens, query_indices=local)
        local_topk = cudnn_indexer_topk(
            q.index_select(0, local.long()),
            k_rank_order.index_select(0, to_global),
            weights.index_select(0, local.long()),
            cu_seqlens,
            topk,
            query_indices=local,
            packed_metadata=meta,
        )
        # Both outputs are canonical (ascending valid prefix, -1 suffix) in global coordinates.
        assert torch.equal(local_topk, full.index_select(0, local.long()))
