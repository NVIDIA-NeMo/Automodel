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

"""Head/tail (round-robin) context-parallel query layout for GLM DSA: batch slicing and cuDNN packed metadata."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import cp as glm_cp
from nemo_automodel.components.models.glm_moe_dsa.kernels.cudnn_dsa import prepare_cudnn_dsa_packed_metadata
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import is_cudnn_dsa_available


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
        glm_cp.local_query_indices("zigzag", _FakeMesh(), 8, torch.device("cpu"))
    with pytest.raises(ValueError, match="cp_query_layout"):
        BackendConfig(attn="cudnn", cp_query_layout="zigzag")


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

    cp_size, chunk = 2, tokens // 4
    for rank in range(cp_size):
        head = torch.arange(rank * chunk, (rank + 1) * chunk, device=device)
        tail_start = (2 * cp_size - 1 - rank) * chunk
        local = torch.cat([head, torch.arange(tail_start, tail_start + chunk, device=device)]).to(torch.int32)
        meta = prepare_cudnn_dsa_packed_metadata(cu_seqlens, tokens, query_indices=local)
        local_topk = cudnn_indexer_topk(
            q.index_select(0, local.long()),
            k,
            weights.index_select(0, local.long()),
            cu_seqlens,
            topk,
            query_indices=local,
            packed_metadata=meta,
        )
        # Both outputs are canonical (ascending valid prefix, -1 suffix) in global coordinates.
        assert torch.equal(local_topk, full.index_select(0, local.long()))
