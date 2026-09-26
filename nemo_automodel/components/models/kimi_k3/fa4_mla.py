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

"""FlashAttention 4 varlen attention for Kimi K3 MLA, with and without context parallelism.

Selected with ``KimiK3TextConfig.mla_attn_backend = "fa4"``. Document-causal attention of a contiguous query
shard against the global keys is expressed as FA4 varlen segments, so no mask is materialized or evaluated per
element and MLA's native 192/128 QK/V head dims run without padding:

* Every document run of a row (padding runs included) is split at the end of the local query shard.
* A piece that overlaps the shard contributes its overlapping queries and all of its keys; FA4's causal mask is
  aligned to the bottom-right corner of each segment, which is exactly "query at global position p sees keys of
  its document up to p".
* Pieces without local queries (other ranks' later tokens, earlier documents) become zero-query segments, so the
  key segments tile the flattened ``[batch * key_sequence]`` tensor and no gather is needed.

Optional FA4/CuTe dependencies are loaded on the first call.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from nemo_automodel.shared.import_utils import safe_import

_PAD_DOC_ID = 0


@functools.cache
def _load_fa4() -> tuple[Callable, Callable]:
    """Import FA4 entry points once; raise a clear error if they are unavailable."""
    available, interface = safe_import("flash_attn.cute.interface")
    if not available:
        raise ImportError("mla_attn_backend='fa4' requires FlashAttention 4 (flash_attn.cute) and nvidia-cutlass-dsl.")
    return interface.flash_attn_func, interface.flash_attn_varlen_func


def _varlen_segments(kv_doc_ids: torch.Tensor, q_length: int, q_global_start: int) -> tuple[list[int], list[int]]:
    """Return per-segment query and key lengths describing the document-causal pattern.

    Args:
        kv_doc_ids: CPU tensor of shape [batch, key_sequence] with 1-based document ids, 0 for padding.
        q_length: Local query length.
        q_global_start: Global sequence offset of the first local query.

    Returns:
        Query lengths and key lengths of every segment, in flattened row-major order.
    """
    q_end = q_global_start + q_length
    q_lengths, k_lengths = [], []
    for row in kv_doc_ids:
        changes = (torch.nonzero(row[1:] != row[:-1]).flatten() + 1).tolist()
        bounds = [0, *changes, row.shape[0]]
        for start, end in zip(bounds[:-1], bounds[1:]):
            # Split at the shard end: keys past it are never visible to local queries.
            for piece_start, piece_end in ((start, min(end, q_end)), (max(start, q_end), end)):
                if piece_end <= piece_start:
                    continue
                q_lengths.append(max(0, min(piece_end, q_end) - max(piece_start, q_global_start)))
                k_lengths.append(piece_end - piece_start)
    return q_lengths, k_lengths


_LAYOUT_CACHE: dict[tuple, Any] = {}
_LAYOUT_GENERATION: list[Any] = [None, None]


def _varlen_layout(q_doc_ids: torch.Tensor, kv_doc_ids: torch.Tensor, *, q_global_start: int):
    """Build (and cache for the step) FA4 varlen ``cu_seqlens`` and max lengths."""
    pointer = kv_doc_ids.data_ptr()
    if pointer != _LAYOUT_GENERATION[0]:
        _LAYOUT_CACHE.clear()
        _LAYOUT_GENERATION[0] = pointer
        # Hold the tensor so the allocator cannot recycle the address mid-step.
        _LAYOUT_GENERATION[1] = kv_doc_ids
    q_length = q_doc_ids.shape[1]
    key = (int(q_global_start), *kv_doc_ids.shape, q_length, kv_doc_ids.device.type, kv_doc_ids.device.index)
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached

    q_lengths, k_lengths = _varlen_segments(kv_doc_ids.cpu(), q_length, int(q_global_start))
    lengths = torch.tensor([q_lengths, k_lengths], dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(dim=1, dtype=torch.int32), (1, 0)).to(kv_doc_ids.device)
    cached = (cu_seqlens[0], cu_seqlens[1], max(q_lengths), max(k_lengths))
    if len(_LAYOUT_CACHE) >= 64:
        _LAYOUT_CACHE.pop(next(iter(_LAYOUT_CACHE)))
    _LAYOUT_CACHE[key] = cached
    return cached


def document_causal_fa4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    q_doc_ids: torch.Tensor,
    kv_doc_ids: torch.Tensor,
    q_global_start: int,
    scale: float,
) -> torch.Tensor:
    """Run causal, per-document attention of local queries against global keys with FA4 varlen.

    Args:
        query: Tensor of shape [batch, heads, query_sequence, qk_head_dim].
        key: Tensor of shape [batch, heads, key_sequence, qk_head_dim].
        value: Tensor of shape [batch, heads, key_sequence, v_head_dim].
        q_doc_ids: Tensor of shape [batch, query_sequence] with 1-based document ids.
        kv_doc_ids: Tensor of shape [batch, key_sequence] with 1-based document ids.
        q_global_start: Global sequence offset of the first query token.
        scale: Softmax scale applied to the query-key product.

    Returns:
        Tensor of shape [batch, heads, query_sequence, v_head_dim]; padding queries are zero.
    """
    flash_attn_varlen_func = _load_fa4()[1]
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = _varlen_layout(
        q_doc_ids, kv_doc_ids, q_global_start=q_global_start
    )
    batch_size, heads, q_length, _ = query.shape
    k_length = key.shape[2]
    output, _ = flash_attn_varlen_func(
        query.transpose(1, 2).reshape(batch_size * q_length, heads, -1),
        key.transpose(1, 2).reshape(batch_size * k_length, heads, -1),
        value.transpose(1, 2).reshape(batch_size * k_length, heads, -1),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
    )
    output = output.view(batch_size, q_length, heads, -1).transpose(1, 2)
    return output.masked_fill((q_doc_ids <= _PAD_DOC_ID)[:, None, :, None], 0)


def causal_fa4_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, scale: float) -> torch.Tensor:
    """Run plain causal attention with FA4 when queries and keys cover the same single-document sequence.

    Args:
        query: Tensor of shape [batch, heads, sequence, qk_head_dim].
        key: Tensor of shape [batch, heads, sequence, qk_head_dim].
        value: Tensor of shape [batch, heads, sequence, v_head_dim].
        scale: Softmax scale applied to the query-key product.

    Returns:
        Tensor of shape [batch, heads, sequence, v_head_dim].
    """
    flash_attn_func = _load_fa4()[0]
    output, _ = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=scale,
        causal=True,
    )
    return output.transpose(1, 2)
