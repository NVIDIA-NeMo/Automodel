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

"""The vectorized QSA block selection is bitwise identical to the chunk-looped original."""

import math

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    Qwen3_8_FlashNextQSAIndexer,
    _qsa_query_chunk_rows,
    select_qsa_token_ids,
)
from tests.unit_tests.models.qwen3_8_flash_next.test_qwen3_8_flash_next_model import _tiny_config


@torch.no_grad()
def _legacy_select_qsa_token_ids(
    index_queries: torch.Tensor,
    compressed_keys: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    query_chunk_size: int = 128,
    query_position_offset: int = 0,
    global_sequence_length: int | None = None,
) -> torch.Tensor:
    """Verbatim pre-vectorization implementation (per-chunk host syncs) used as the bitwise oracle."""
    if index_queries.ndim != 4:
        raise ValueError(f"index_queries must be [B, S, H, D], got {tuple(index_queries.shape)}")
    if compressed_keys.ndim != 4 or compressed_keys.shape[2] != 1:
        raise ValueError(f"compressed_keys must be [B, P, 1, D], got {tuple(compressed_keys.shape)}")
    batch_size, query_sequence_length, num_heads, head_dim = index_queries.shape
    if compressed_keys.shape[0] != batch_size or compressed_keys.shape[-1] != head_dim:
        raise ValueError("QSA query/key batch and head dimensions must match")
    if sequence_lengths.shape != (batch_size,):
        raise ValueError(f"sequence_lengths must be [{batch_size}], got {tuple(sequence_lengths.shape)}")
    if token_budget <= 0 or compress_ratio <= 1 or token_budget % compress_ratio != 0:
        raise ValueError(
            "QSA requires a positive token_budget divisible by compress_ratio > 1; "
            f"got token_budget={token_budget}, compress_ratio={compress_ratio}"
        )
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError("QSA index queries require positive head count and head dimension")
    if query_position_offset < 0:
        raise ValueError(f"query_position_offset must be non-negative, got {query_position_offset}")
    if global_sequence_length is None:
        global_sequence_length = query_sequence_length
    if global_sequence_length < query_position_offset + query_sequence_length:
        raise ValueError(
            "QSA global sequence does not cover every local query; "
            f"got global={global_sequence_length}, offset={query_position_offset}, local={query_sequence_length}"
        )

    lengths = sequence_lengths.to(device=index_queries.device, dtype=torch.long)
    if bool(((lengths < 0) | (lengths > global_sequence_length)).any()):
        raise ValueError(f"QSA logical lengths must lie in [0, {global_sequence_length}]")
    required_blocks = torch.div(lengths, compress_ratio, rounding_mode="floor")
    num_blocks = compressed_keys.shape[1]
    if bool((required_blocks > num_blocks).any()):
        raise ValueError(
            "compressed_keys do not contain every complete logical block; "
            f"need at least {int(required_blocks.max())}, got {num_blocks}"
        )

    block_budget = token_budget // compress_ratio
    final_width = token_budget + compress_ratio - 1
    selected_tokens = torch.full(
        (batch_size, query_sequence_length, final_width),
        -1,
        dtype=torch.int32,
        device=index_queries.device,
    )
    block_offsets = torch.arange(compress_ratio, device=index_queries.device, dtype=torch.long)
    tail_offsets = torch.arange(compress_ratio - 1, device=index_queries.device, dtype=torch.long)
    score_scale = math.sqrt(head_dim)

    for batch_idx in range(batch_size):
        logical_length = int(lengths[batch_idx])
        available_blocks = logical_length // compress_ratio
        keys = compressed_keys[batch_idx, :available_blocks, 0].float()
        local_valid_length = min(max(logical_length - query_position_offset, 0), query_sequence_length)
        for query_start in range(0, local_valid_length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, local_valid_length)
            query_positions = torch.arange(
                query_position_offset + query_start,
                query_position_offset + query_end,
                device=index_queries.device,
            )
            visible_blocks = torch.div(query_positions + 1, compress_ratio, rounding_mode="floor")
            rows = query_end - query_start
            result = torch.full((rows, final_width), -1, dtype=torch.int32, device=index_queries.device)

            topk_width = min(block_budget, available_blocks)
            if topk_width:
                # Gold fast_topk preserves causal block order while all visible
                # blocks fit the budget.  It starts score-ordered top-k only on
                # the first genuinely sparse row (t=2051 for c4/budget=2048).
                candidate_blocks = torch.arange(topk_width, device=index_queries.device)
                top_blocks = candidate_blocks.unsqueeze(0).expand(rows, -1).clone()
                valid_blocks = candidate_blocks.unsqueeze(0) < visible_blocks.unsqueeze(1)
                sparse_rows = visible_blocks > block_budget
                if bool(sparse_rows.any()):
                    sparse_queries = index_queries[batch_idx, query_start:query_end][sparse_rows].float()
                    scores = torch.einsum("qhd,pd->qhp", sparse_queries, keys)
                    scores = torch.relu(scores).sum(dim=1) / score_scale
                    block_ids = torch.arange(available_blocks, device=index_queries.device)
                    sparse_visible = visible_blocks[sparse_rows]
                    scores = scores.masked_fill(block_ids.unsqueeze(0) >= sparse_visible.unsqueeze(1), -torch.inf)
                    top_blocks[sparse_rows] = torch.topk(scores, k=block_budget, dim=-1).indices
                    valid_blocks[sparse_rows] = True
                expanded = top_blocks.unsqueeze(-1) * compress_ratio + block_offsets
                expanded = torch.where(valid_blocks.unsqueeze(-1), expanded, -torch.ones_like(expanded))
                result[:, : topk_width * compress_ratio] = expanded.reshape(rows, -1).to(torch.int32)

            tail_start = visible_blocks * compress_ratio
            tail_count = query_positions + 1 - tail_start
            valid_block_count = torch.minimum(visible_blocks, torch.full_like(visible_blocks, block_budget))
            tail_values = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail_valid = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
            if bool(tail_valid.any()):
                destination = valid_block_count.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
                row_ids = torch.arange(rows, device=index_queries.device).unsqueeze(1).expand_as(tail_valid)
                result[row_ids[tail_valid], destination[tail_valid]] = tail_values[tail_valid].to(torch.int32)

            selected_tokens[batch_idx, query_start:query_end] = result

    return selected_tokens


def _inputs(batch: int, seq: int, heads: int, dim: int, ratio: int, seed: int):
    """Random fp32 index queries ``[B, S, H, D]`` and compressed keys ``[B, S // ratio, 1, D]``."""
    g = torch.Generator().manual_seed(seed)
    queries = torch.randn(batch, seq, heads, dim, generator=g)
    keys = torch.randn(batch, seq // ratio, 1, dim, generator=g)
    return queries, keys


@pytest.mark.parametrize("query_chunk_size", [None, 1, 3, 7, 64])
def test_vectorized_selection_matches_legacy_bitwise(query_chunk_size: int | None) -> None:
    """Full, short, tiny and empty logical lengths; dense, sparse and tail rows all agree."""
    seq, ratio, budget = 64, 4, 16
    queries, keys = _inputs(4, seq, heads=2, dim=8, ratio=ratio, seed=1)
    lengths = torch.tensor([64, 37, 3, 0])
    expected = _legacy_select_qsa_token_ids(queries, keys, lengths, token_budget=budget, compress_ratio=ratio)
    actual = select_qsa_token_ids(
        queries, keys, lengths, token_budget=budget, compress_ratio=ratio, query_chunk_size=query_chunk_size
    )
    assert actual.dtype == torch.int32 and actual.shape == (4, seq, budget + ratio - 1)
    assert torch.equal(actual, expected)
    # Sparse rows exist (position >= (budget/ratio + 1) * ratio - 1 = 19) and carry a full budget.
    assert (actual[0, 19:, :budget] >= 0).all()


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_vectorized_selection_matches_legacy_under_cp_offset(cp_rank: int) -> None:
    """Contiguous CP: local queries with a global offset against globally compressed keys."""
    global_seq, ratio, budget, local = 64, 4, 16, 32
    queries, keys = _inputs(2, global_seq, heads=2, dim=8, ratio=ratio, seed=2)
    local_queries = queries[:, cp_rank * local : (cp_rank + 1) * local]
    lengths = torch.tensor([64, 45])
    kwargs = dict(
        token_budget=budget,
        compress_ratio=ratio,
        query_position_offset=cp_rank * local,
        global_sequence_length=global_seq,
    )
    expected = _legacy_select_qsa_token_ids(local_queries, keys, lengths, **kwargs)
    actual = select_qsa_token_ids(local_queries, keys, lengths, **kwargs)
    assert torch.equal(actual, expected)


def test_vectorized_selection_rejects_bad_lengths_like_legacy() -> None:
    queries, keys = _inputs(1, 16, heads=1, dim=4, ratio=4, seed=3)
    with pytest.raises(ValueError, match="logical lengths"):
        select_qsa_token_ids(queries, keys, torch.tensor([17]), token_budget=8, compress_ratio=4)
    with pytest.raises(ValueError, match="query_chunk_size"):
        select_qsa_token_ids(queries, keys, torch.tensor([16]), token_budget=8, compress_ratio=4, query_chunk_size=0)


def test_default_chunk_rows_follow_score_memory_budget() -> None:
    # 4k tokens, 4 heads, 1024 blocks: 16 KiB per row, well inside 256 MiB -> one chunk.
    assert _qsa_query_chunk_rows(None, num_heads=4, num_blocks=1024, rows=4096) == 4096
    # 128k tokens: 512 KiB per row -> 512 rows per chunk.
    assert _qsa_query_chunk_rows(None, num_heads=4, num_blocks=32768, rows=131072) == 512
    assert _qsa_query_chunk_rows(None, num_heads=4, num_blocks=1, rows=5) == 5
    assert _qsa_query_chunk_rows(128, num_heads=4, num_blocks=32768, rows=131072) == 128


def test_indexer_reads_optional_chunk_size_from_config() -> None:
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")
    config = _tiny_config().text_config
    assert Qwen3_8_FlashNextQSAIndexer(config, backend).query_chunk_size is None
    config.qsa_indexer_query_chunk_size = 96
    assert Qwen3_8_FlashNextQSAIndexer(config, backend).query_chunk_size == 96
    config.qsa_indexer_query_chunk_size = 0
    with pytest.raises(ValueError, match="qsa_indexer_query_chunk_size"):
        Qwen3_8_FlashNextQSAIndexer(config, backend)
