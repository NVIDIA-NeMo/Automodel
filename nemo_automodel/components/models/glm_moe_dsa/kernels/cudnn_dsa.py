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

"""cuDNN and FlashMLA kernels for the split GLM-5.2 DSA path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nemo_automodel.shared.import_utils import safe_import_from

_HAS_CUDNN_DSA, _CUDNN_DSA = safe_import_from(
    "cudnn",
    "DSA",
    msg=(
        "cuDNN Frontend DSA kernels are unavailable. Install "
        "nvidia-cudnn-frontend[cutedsl] to use backend.attn='cudnn'."
    ),
)
_HAS_FLASH_MLA, _FLASH_MLA_SPARSE_FWD = safe_import_from(
    "flash_mla",
    "flash_mla_sparse_fwd",
    msg="FlashMLA sparse prefill is unavailable. Install the FlashMLA nv_dev package.",
)

_INDEX_HEAD_DIM = 128
_ATTENTION_HEAD_DIM = 576
_VALUE_HEAD_DIM = 512
_MAX_TOPK = 2048
_FLASH_MLA_TOPK_ALIGNMENT = 512
_TOPK_SCRATCH_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
_TOPK_SCRATCH_INT32_FACTOR = 2
_TOPK_ROW_ALIGNMENT = 512

@dataclass(frozen=True)
class CudnnDsaPackedMetadata:
    """Reusable THD metadata for local-query/global-key cuDNN DSA."""

    starts: torch.Tensor
    causal_lengths: torch.Tensor
    segment_cu_q: torch.Tensor
    segment_cu_k: torch.Tensor
    key_source_indices: torch.Tensor | None
    max_seqlen_q: int
    max_seqlen_k: int
    total_key_tokens: int


def is_cudnn_dsa_available() -> bool:
    """Return whether both optional libraries required by cuDNN DSA import."""
    return bool(_HAS_CUDNN_DSA and _HAS_FLASH_MLA)


def _require_available() -> None:
    """Raise when either optional runtime required by the split kernel is absent."""
    if not is_cudnn_dsa_available():
        raise RuntimeError(
            "GLM-5.2 cuDNN DSA requires both nvidia-cudnn-frontend[cutedsl] and FlashMLA with flash_mla_sparse_fwd."
        )


def _require_cuda_tensors(operation: str, *tensors: torch.Tensor) -> tuple[int, int]:
    """Validate that arbitrary-layout input tensors share one SM90+ CUDA device."""
    if not tensors or any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError(f"{operation} requires CUDA tensors.")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(f"{operation} requires every tensor on the same CUDA device.")
    major, minor = torch.cuda.get_device_capability(device)
    if major < 9:
        raise RuntimeError(f"{operation} requires SM90 or later, got SM{major}{minor}.")
    return major, minor


def _validate_topk(index_topk: int) -> None:
    """Validate GLM-5.2's fixed sparse-selection width."""
    if isinstance(index_topk, bool) or not isinstance(index_topk, int):
        raise TypeError(f"index_topk must be an int, got {type(index_topk).__name__}.")
    if not 0 < index_topk <= _MAX_TOPK:
        raise ValueError(f"index_topk must be in [1, {_MAX_TOPK}], got {index_topk}.")


def prepare_cudnn_dsa_packed_metadata(
    cu_seqlens: torch.Tensor,
    total_key_tokens: int,
    max_seqlen: int | torch.Tensor | None = None,
    *,
    query_indices: torch.Tensor | None = None,
    cu_seqlens_padded: torch.Tensor | None = None,
) -> CudnnDsaPackedMetadata:
    """Build segmented THD metadata for local queries and gathered global keys.

    Args:
        cu_seqlens: Cumulative real sequence lengths, int32 ``[sequences + 1]``.
        total_key_tokens: Number of tokens in the gathered, padded THD key tensor.
        max_seqlen: Optional precomputed maximum sequence length as a Python integer or
            scalar integer tensor. Its value is checked against ``cu_seqlens``.
        query_indices: Optional contiguous global padded-token coordinates for the local
            query rows. When absent, queries cover all global key tokens (CP=1).
        cu_seqlens_padded: Optional cumulative padded layout boundaries. When absent,
            the real boundaries in ``cu_seqlens`` are also the storage boundaries.

    Returns:
        Cached global starts and real causal lengths per query, positive-length cuDNN
        Q/K segments, optional indices that repack gathered K prefixes, and maxima.

    This validation intentionally performs one device-to-host synchronization. The model
    prepares the object once per pipeline stage and reuses it across every indexer and
    shared-attention layer.
    """
    if isinstance(total_key_tokens, bool) or not isinstance(total_key_tokens, int) or total_key_tokens <= 0:
        raise ValueError(f"total_key_tokens must be a positive integer, got {total_key_tokens!r}.")
    if total_key_tokens >= torch.iinfo(torch.int32).max:
        raise ValueError("The gathered THD key-token count must fit in an int32 global index.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional [num_sequences + 1] tensor.")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}.")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous.")

    layout_cu = cu_seqlens if cu_seqlens_padded is None else cu_seqlens_padded
    if layout_cu.ndim != 1 or layout_cu.shape != cu_seqlens.shape:
        raise ValueError("cu_seqlens_padded must have the same one-dimensional shape as cu_seqlens.")
    if layout_cu.dtype != torch.int32:
        raise TypeError(f"cu_seqlens_padded must be int32, got {layout_cu.dtype}.")
    if layout_cu.device != cu_seqlens.device or not layout_cu.is_contiguous():
        raise ValueError("cu_seqlens and cu_seqlens_padded must be contiguous on the same device.")

    real_lengths = cu_seqlens[1:].to(torch.int64) - cu_seqlens[:-1].to(torch.int64)
    padded_lengths = layout_cu[1:].to(torch.int64) - layout_cu[:-1].to(torch.int64)
    if query_indices is None:
        global_queries = torch.arange(total_key_tokens, dtype=torch.int64, device=cu_seqlens.device)
    else:
        if query_indices.ndim != 1 or query_indices.numel() == 0:
            raise ValueError("query_indices must be a non-empty one-dimensional tensor.")
        if query_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"query_indices must be int32 or int64, got {query_indices.dtype}.")
        if query_indices.device != cu_seqlens.device or not query_indices.is_contiguous():
            raise ValueError("query_indices and cu_seqlens must be contiguous on the same device.")
        global_queries = query_indices.to(torch.int64)

    # AutoModel CP keeps one contiguous interval of global padded-token rows per rank.
    # Filter non-intersecting documents, then pair each local Q segment with that
    # document's K prefix. The unequal Q/K lengths give cuDNN the required bottom-right
    # causal alignment without a per-query causal-offset API.
    document_ids = torch.searchsorted(layout_cu, global_queries, right=True).sub(1)
    safe_document_ids = document_ids.clamp(0, real_lengths.numel() - 1).to(torch.long)
    segment_documents, query_counts = torch.unique_consecutive(safe_document_ids, return_counts=True)
    segment_cu_q64 = torch.nn.functional.pad(query_counts.cumsum(0), (1, 0))
    segment_query_starts = global_queries.index_select(0, segment_cu_q64[:-1])
    segment_document_starts = layout_cu.to(torch.int64).index_select(0, segment_documents)
    key_counts = segment_query_starts + query_counts - segment_document_starts
    segment_cu_k64 = torch.nn.functional.pad(key_counts.cumsum(0), (1, 0))

    query_deltas = global_queries[1:] - global_queries[:-1]
    if query_deltas.numel() == 0:
        query_deltas = global_queries.new_ones(1)
    if max_seqlen is not None:
        if isinstance(max_seqlen, torch.Tensor):
            if max_seqlen.numel() != 1 or max_seqlen.dtype not in (torch.int32, torch.int64):
                raise ValueError("max_seqlen must be a scalar int32 or int64 tensor.")
            provided_max = max_seqlen.to(device=cu_seqlens.device, dtype=torch.int64).reshape(())
        elif isinstance(max_seqlen, bool) or not isinstance(max_seqlen, int):
            raise TypeError(f"max_seqlen must be an int or scalar integer tensor, got {type(max_seqlen).__name__}.")
        else:
            provided_max = torch.tensor(max_seqlen, dtype=torch.int64, device=cu_seqlens.device)
    else:
        provided_max = real_lengths.max()

    summary = torch.stack(
        (
            cu_seqlens[0].to(torch.int64),
            real_lengths.min(),
            real_lengths.max(),
            layout_cu[0].to(torch.int64),
            layout_cu[-1].to(torch.int64),
            padded_lengths.min(),
            (padded_lengths - real_lengths).min(),
            global_queries[0],
            global_queries[-1],
            query_deltas.min(),
            query_deltas.max(),
            document_ids.min(),
            document_ids.max(),
            query_counts.min(),
            query_counts.max(),
            key_counts.min(),
            key_counts.max(),
            key_counts.sum(),
            provided_max,
        )
    ).tolist()
    (
        real_first,
        min_real,
        actual_max_seqlen,
        layout_first,
        layout_last,
        min_padded,
        min_padding,
        query_first,
        query_last,
        min_query_delta,
        max_query_delta,
        min_document_id,
        max_document_id,
        min_query_count,
        max_query_count,
        min_key_count,
        max_key_count,
        segment_key_tokens,
        provided_max_value,
    ) = (int(value) for value in summary)
    if real_first != 0 or min_real <= 0:
        raise ValueError("cu_seqlens must start at zero and be strictly increasing.")
    if layout_first != 0 or layout_last != total_key_tokens or min_padded <= 0:
        raise ValueError(
            "The padded THD layout must start at zero, end at total_key_tokens, and be strictly increasing; "
            f"got first={layout_first}, last={layout_last}, total_key_tokens={total_key_tokens}."
        )
    if min_padding < 0:
        raise ValueError("Every padded sequence length must be at least its real sequence length.")
    if query_first < 0 or query_last >= total_key_tokens or min_query_delta != 1 or max_query_delta != 1:
        raise ValueError(
            "query_indices must be one contiguous increasing interval inside the global padded THD layout."
        )
    if min_document_id < 0 or max_document_id >= real_lengths.numel():
        raise ValueError("query_indices contain positions outside the global padded THD layout.")
    if min_query_count <= 0 or min_key_count <= 0:
        raise ValueError("cuDNN DSA requires every filtered Q/K segment to have positive length.")
    if provided_max_value != actual_max_seqlen:
        raise ValueError(
            f"max_seqlen must equal the largest real packed length ({actual_max_seqlen}), "
            f"got {provided_max_value}."
        )

    starts64 = layout_cu.to(torch.int64).index_select(0, safe_document_ids)
    real_ends = starts64 + real_lengths.index_select(0, safe_document_ids)
    causal_lengths64 = torch.minimum(global_queries + 1, real_ends) - starts64
    causal_lengths64.clamp_(min=1)

    full_identity_query = (
        global_queries.numel() == total_key_tokens and query_first == 0 and query_last == total_key_tokens - 1
    )
    key_source_indices = None
    if not full_identity_query:
        repeated_starts = torch.repeat_interleave(segment_document_starts, key_counts, output_size=segment_key_tokens)
        repeated_offsets = torch.repeat_interleave(segment_cu_k64[:-1], key_counts, output_size=segment_key_tokens)
        key_source_indices = (
            repeated_starts + torch.arange(segment_key_tokens, device=cu_seqlens.device) - repeated_offsets
        ).to(torch.long)

    return CudnnDsaPackedMetadata(
        starts=starts64.to(torch.int32).contiguous(),
        causal_lengths=causal_lengths64.to(torch.int32).contiguous(),
        segment_cu_q=segment_cu_q64.to(torch.int32).contiguous(),
        segment_cu_k=segment_cu_k64.to(torch.int32).contiguous(),
        key_source_indices=key_source_indices.contiguous() if key_source_indices is not None else None,
        max_seqlen_q=max_query_count,
        max_seqlen_k=max_key_count,
        total_key_tokens=total_key_tokens,
    )


def _unpack_packed_metadata(
    packed_metadata: CudnnDsaPackedMetadata,
    *,
    total_query_tokens: int,
    total_key_tokens: int,
    device: torch.device,
) -> CudnnDsaPackedMetadata:
    """Validate reusable packed metadata without synchronizing CUDA values to the host."""
    if not isinstance(packed_metadata, CudnnDsaPackedMetadata):
        raise TypeError("packed_metadata must be a CudnnDsaPackedMetadata object.")
    expected_shape = (total_query_tokens,)
    for name, tensor in (("starts", packed_metadata.starts), ("causal_lengths", packed_metadata.causal_lengths)):
        if tensor.shape != expected_shape or tensor.dtype != torch.int32 or tensor.device != device:
            raise ValueError(
                f"packed {name} must be int32 {expected_shape} on {device}, got "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"packed {name} must be contiguous.")
    for name, tensor in (
        ("segment_cu_q", packed_metadata.segment_cu_q),
        ("segment_cu_k", packed_metadata.segment_cu_k),
    ):
        if tensor.ndim != 1 or tensor.numel() < 2 or tensor.dtype != torch.int32 or tensor.device != device:
            raise ValueError(f"packed {name} must be a CUDA int32 [segments + 1] tensor on {device}.")
        if not tensor.is_contiguous():
            raise ValueError(f"packed {name} must be contiguous.")
    if packed_metadata.segment_cu_q.shape != packed_metadata.segment_cu_k.shape:
        raise ValueError("packed segment_cu_q and segment_cu_k must have the same shape.")
    if packed_metadata.total_key_tokens != total_key_tokens:
        raise ValueError(
            f"packed total_key_tokens must be {total_key_tokens}, got {packed_metadata.total_key_tokens}."
        )
    if packed_metadata.max_seqlen_q <= 0 or packed_metadata.max_seqlen_k <= 0:
        raise ValueError("packed maximum Q/K sequence lengths must be positive Python integers.")
    source = packed_metadata.key_source_indices
    if source is not None and (
        source.ndim != 1 or source.dtype != torch.int64 or source.device != device or not source.is_contiguous()
    ):
        raise ValueError("packed key_source_indices must be a contiguous int64 tensor on the kernel device.")
    return packed_metadata


def _topk_wrapper_chunked(scores: torch.Tensor, seq_lens: torch.Tensor, topk: int) -> torch.Tensor:
    """Select top-k for FP32 scores ``[T, S_max]`` using causal lengths ``[T]``."""
    n_rows, n_cols = scores.shape
    scratch_per_row = max(1, n_cols) * torch.iinfo(torch.int32).bits // 8
    scratch_per_row *= _TOPK_SCRATCH_INT32_FACTOR
    chunk_rows = max(1, _TOPK_SCRATCH_LIMIT_BYTES // max(1, scratch_per_row))
    if chunk_rows >= _TOPK_ROW_ALIGNMENT:
        chunk_rows = (chunk_rows // _TOPK_ROW_ALIGNMENT) * _TOPK_ROW_ALIGNMENT
    chunk_rows = min(n_rows, chunk_rows)

    chunks = []
    for row_start in range(0, n_rows, chunk_rows):
        row_end = min(row_start + chunk_rows, n_rows)
        result = _CUDNN_DSA.indexer_top_k_wrapper(
            scores[row_start:row_end].contiguous(),
            seq_lens[row_start:row_end].contiguous(),
            top_k=topk,
            next_n=1,
            return_val=False,
        )
        chunks.append(result["indices"])
    return torch.cat(chunks, dim=0)


def _compact_and_sort_indices(indices: torch.Tensor, key_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize global indices ``[T, K]`` and return valid lengths ``[T]``."""
    valid = (indices >= 0) & (indices < key_count)
    topk_length = valid.sum(dim=-1, dtype=torch.int32)
    positions = torch.arange(indices.size(-1), device=indices.device).view(1, -1)
    compact_order = torch.where(valid, positions, torch.full_like(positions, indices.size(-1))).argsort(dim=-1)
    indices = torch.gather(indices, -1, compact_order)
    compact_valid = torch.gather(valid, -1, compact_order)
    indices = indices.masked_fill(~compact_valid, -1)

    prefix = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(prefix, indices, torch.full_like(indices, key_count))
    sort_order = sort_key.argsort(dim=-1)
    indices = torch.gather(indices, -1, sort_order)
    sorted_valid = torch.gather(prefix.expand_as(indices), -1, sort_order)
    return indices.masked_fill(~sorted_valid, -1).to(torch.int32).contiguous(), topk_length.contiguous()


def cudnn_indexer_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    index_topk: int,
    query_indices: torch.Tensor | None = None,
    cu_seqlens_padded: torch.Tensor | None = None,
    packed_metadata: CudnnDsaPackedMetadata | None = None,
) -> torch.Tensor:
    """Compute GLM-5.2 packed-THD indexer top-k with cuDNN Frontend.

    Args:
        index_q: Rank-local indexer query in THD layout, BF16 ``[T_q, H_index, 128]``.
        index_k: Gathered global indexer key in TD layout, BF16 ``[T_k, 128]``.
        head_weights: Already-scaled per-head weights in TH layout, FP32 or BF16
            ``[T, H_index]``. The caller owns both ``H_index**-0.5`` and
            ``128**-0.5`` scaling; this function does not rescale them.
        cu_seqlens: Compact packed-sequence offsets, CUDA int32
            ``[num_sequences + 1]``.
        index_topk: Fixed output width ``K`` in ``[1, 2048]``.
        query_indices: Optional contiguous global padded coordinates for local query
            rows, ``[T_q]``. Absent means CP=1 identity coordinates.
        cu_seqlens_padded: Optional global padded packed-layout offsets.
        packed_metadata: Optional metadata object returned by
            :func:`prepare_cudnn_dsa_packed_metadata`. Supplying it avoids rebuilding
            and synchronizing the same metadata in every full-indexer layer.

    Returns:
        CUDA int32 top-k indices in global compact-THD coordinates,
        ``[T, 1, K]``. Each row contains an ascending, compact valid prefix and
        a ``-1`` suffix, and the tensor is contiguous.

    Raises:
        RuntimeError: If optional kernels, CUDA, or SM90+ are unavailable.
        TypeError: If tensor dtypes or ``index_topk`` are invalid.
        ValueError: If tensor shapes or compact packed metadata are invalid.
    """
    _require_available()
    _validate_topk(index_topk)
    major, _ = _require_cuda_tensors("cuDNN DSA indexer", index_q, index_k, head_weights, cu_seqlens)

    if index_q.dtype != torch.bfloat16 or index_k.dtype != torch.bfloat16:
        raise TypeError(f"cuDNN DSA index_q and index_k must be bfloat16, got {index_q.dtype} and {index_k.dtype}.")
    if head_weights.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"head_weights must be float32 or bfloat16, got {head_weights.dtype}.")
    if index_q.ndim != 3 or index_q.shape[-1] != _INDEX_HEAD_DIM:
        raise ValueError(f"index_q must have shape [T, H_index, {_INDEX_HEAD_DIM}], got {tuple(index_q.shape)}.")
    if index_k.ndim != 2 or index_k.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(f"index_k must have shape [T_k, {_INDEX_HEAD_DIM}], got {tuple(index_k.shape)}.")
    if head_weights.shape != index_q.shape[:2]:
        raise ValueError(f"head_weights must have shape {tuple(index_q.shape[:2])}, got {tuple(head_weights.shape)}.")
    supported_heads = (32, 64)
    if index_q.shape[1] not in supported_heads:
        raise ValueError(
            f"cuDNN DSA indexer on SM{major} supports H_index in {supported_heads}, got {index_q.shape[1]}."
        )
    if index_k.shape[0] >= torch.iinfo(torch.int32).max:
        raise ValueError("The gathered THD key-token count must fit in an int32 global index.")

    if packed_metadata is None:
        packed_metadata = prepare_cudnn_dsa_packed_metadata(
            cu_seqlens,
            index_k.shape[0],
            query_indices=query_indices,
            cu_seqlens_padded=cu_seqlens_padded,
        )
    else:
        packed_metadata = _unpack_packed_metadata(
            packed_metadata,
            total_query_tokens=index_q.shape[0],
            total_key_tokens=index_k.shape[0],
            device=index_q.device,
        )
    index_k_segmented = index_k
    if packed_metadata.key_source_indices is not None:
        index_k_segmented = index_k.index_select(0, packed_metadata.key_source_indices)
    scores = _CUDNN_DSA.indexer_forward_wrapper(
        index_q.contiguous(),
        index_k_segmented.unsqueeze(1).contiguous(),
        head_weights.to(torch.bfloat16).contiguous(),
        ratio=1,
        sm_scale=1.0,
        cu_seqlens_q=packed_metadata.segment_cu_q,
        cu_seqlens_k=packed_metadata.segment_cu_k,
        max_seqlen_q=packed_metadata.max_seqlen_q,
        max_seqlen_k=packed_metadata.max_seqlen_k,
    )["scores"]
    actual_topk = min(index_topk, packed_metadata.max_seqlen_k)
    if actual_topk == packed_metadata.max_seqlen_k:
        local_indices = torch.arange(actual_topk, dtype=torch.int32, device=index_q.device).expand(index_q.shape[0], -1)
    else:
        local_indices = _topk_wrapper_chunked(
            scores.contiguous(), packed_metadata.causal_lengths, actual_topk
        ).to(torch.int32)

    valid = (local_indices >= 0) & (local_indices < packed_metadata.causal_lengths.unsqueeze(-1))
    global_indices = torch.where(valid, local_indices + packed_metadata.starts.unsqueeze(-1), -1)
    if actual_topk < index_topk:
        global_indices = torch.nn.functional.pad(global_indices, (0, index_topk - actual_topk), value=-1)
    global_indices, _ = _compact_and_sort_indices(global_indices, index_k.shape[0])
    return global_indices.unsqueeze(1).contiguous()


def _padded_head_count(num_heads: int, major: int) -> int:
    """Return the FlashMLA-supported head count for one SM generation."""
    if major >= 10:
        for padded in (64, 128):
            if num_heads == padded or (num_heads < padded and padded % num_heads == 0):
                return padded
        alignment = 128
    else:
        alignment = 64
    if num_heads % alignment == 0:
        return num_heads
    if num_heads < alignment and alignment % num_heads == 0:
        return alignment
    raise ValueError(f"FlashMLA sparse prefill requires the query-head count to divide {alignment}, got H={num_heads}.")


def _pad_attention_heads(
    q: torch.Tensor, attn_sink: torch.Tensor, padded_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad query ``[T, H, 576]`` and sink ``[H]`` to ``padded_heads``."""
    if q.shape[1] == padded_heads:
        return q, attn_sink
    q_padded = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
    q_padded[:, : q.shape[1]] = q
    sink_padded = attn_sink.new_full((padded_heads,), float("-inf"))
    sink_padded[: q.shape[1]] = attn_sink
    return q_padded, sink_padded


class _CudnnSparseAttention(torch.autograd.Function):
    """Pair FlashMLA forward with cuDNN backward for latent THD attention."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        padded_heads: int,
        topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        """Produce latent values ``[T_q, H, 512]`` from Q/KV and ``[T_q, 1, K]`` indices."""
        kv = kv_latent.squeeze(1).contiguous()
        if topk_length is None:
            indices, topk_length = _compact_and_sort_indices(topk_indices.squeeze(1), kv.shape[0])
        else:
            # Model-prepared lengths imply the indexer already emitted a compact, sorted
            # valid prefix. Reuse it directly instead of sorting [T, K] in every shared layer.
            indices = topk_indices.squeeze(1).contiguous()
        padded_topk = math.ceil(indices.shape[-1] / _FLASH_MLA_TOPK_ALIGNMENT) * _FLASH_MLA_TOPK_ALIGNMENT
        if padded_topk != indices.shape[-1]:
            indices = torch.nn.functional.pad(indices, (0, padded_topk - indices.shape[-1]), value=-1)

        attn_sink = torch.full((q.shape[1],), float("-inf"), dtype=torch.float32, device=q.device)
        q_kernel, sink_kernel = _pad_attention_heads(q.contiguous(), attn_sink, padded_heads)
        out_kernel, _max_logits, lse_kernel = _FLASH_MLA_SPARSE_FWD(
            q_kernel,
            kv.unsqueeze(1),
            indices.unsqueeze(1),
            softmax_scale,
            d_v=_VALUE_HEAD_DIM,
            attn_sink=sink_kernel,
            topk_length=topk_length,
            indexer_topk=0,
        )
        out = out_kernel[:, : q.shape[1]].contiguous()
        lse = lse_kernel[:, : q.shape[1]].contiguous()
        ctx.save_for_backward(q, kv, out, lse, attn_sink, indices.clamp_min(0), topk_length)
        ctx.softmax_scale = softmax_scale
        ctx.padded_heads = padded_heads
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Map output gradients ``[T_q, H, 512]`` to Q and latent KV layouts."""
        q, kv, out, lse, attn_sink, indices, topk_length = ctx.saved_tensors
        q_kernel, sink_kernel = _pad_attention_heads(q, attn_sink, ctx.padded_heads)
        if ctx.padded_heads == q.shape[1]:
            out_kernel = out
            grad_kernel = grad_output.contiguous()
            lse_kernel = lse
        else:
            out_kernel = out.new_zeros((out.shape[0], ctx.padded_heads, out.shape[2]))
            out_kernel[:, : out.shape[1]] = out
            grad_kernel = grad_output.new_zeros((grad_output.shape[0], ctx.padded_heads, grad_output.shape[2]))
            grad_kernel[:, : grad_output.shape[1]] = grad_output
            lse_kernel = lse.new_zeros((lse.shape[0], ctx.padded_heads))
            lse_kernel[:, : lse.shape[1]] = lse

        result = _CUDNN_DSA.sparse_attention_backward_wrapper(
            q_kernel.contiguous(),
            kv,
            out_kernel.contiguous(),
            grad_kernel.contiguous(),
            lse_kernel.contiguous(),
            sink_kernel,
            indices,
            softmax_scale=ctx.softmax_scale,
            topk_length=topk_length,
        )
        grad_q = result["dq"][:, : q.shape[1]].contiguous()
        grad_kv = result["dkv"].unsqueeze(1).contiguous()
        return grad_q, grad_kv, None, None, None, None


def cudnn_sparse_attention(
    q: torch.Tensor,
    kv_latent: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run split GLM-5.2 sparse MLA with FlashMLA forward and cuDNN backward.

    Args:
        q: Absorbed MLA query, CUDA BF16 THD ``[T_q, H, 576]``. The final
            dimension is ``kv_lora_rank + qk_rope_head_dim`` (``512 + 64``).
        kv_latent: Shared latent key/value, CUDA BF16 THD ``[T_kv, 1, 576]``.
        topk_indices: Global flattened-KV indices, CUDA int32
            ``[T_q, 1, K]`` with ``-1`` for invalid slots.
        softmax_scale: Already-computed MLA attention scale. It is forwarded
            unchanged to both FlashMLA and cuDNN backward.
        topk_length: Optional int32 valid-prefix lengths ``[T_q]`` prepared once
            from packed causal metadata. When supplied, ``topk_indices`` must already
            contain the indexer's canonical compact, ascending prefix.

    Returns:
        Latent sparse-attention output, CUDA BF16 ``[T_q, H, 512]``. The caller
        applies the model-owned value up-projection ``w_vc``.

    Raises:
        RuntimeError: If optional kernels, CUDA, or SM90+ are unavailable.
        TypeError: If compute tensors are not BF16 or indices are not int32.
        ValueError: If tensor layouts, dimensions, top-k, or scale are invalid.
    """
    _require_available()
    major, _ = _require_cuda_tensors("cuDNN DSA sparse attention", q, kv_latent, topk_indices)
    if q.dtype != torch.bfloat16 or kv_latent.dtype != torch.bfloat16:
        raise TypeError(f"q and kv_latent must be bfloat16, got {q.dtype} and {kv_latent.dtype}.")
    if topk_indices.dtype != torch.int32:
        raise TypeError(f"topk_indices must be int32, got {topk_indices.dtype}.")
    if q.ndim != 3 or q.shape[-1] != _ATTENTION_HEAD_DIM:
        raise ValueError(f"q must have shape [T_q, H, {_ATTENTION_HEAD_DIM}], got {tuple(q.shape)}.")
    if kv_latent.ndim != 3 or kv_latent.shape[1:] != (1, _ATTENTION_HEAD_DIM):
        raise ValueError(f"kv_latent must have shape [T_kv, 1, {_ATTENTION_HEAD_DIM}], got {tuple(kv_latent.shape)}.")
    if topk_indices.ndim != 3 or topk_indices.shape[:2] != (q.shape[0], 1):
        raise ValueError(f"topk_indices must have shape [T_q, 1, K], got {tuple(topk_indices.shape)}.")
    _validate_topk(topk_indices.shape[-1])
    if kv_latent.shape[0] >= torch.iinfo(torch.int32).max:
        raise ValueError("The flattened KV token count must fit in an int32 global index.")
    if not isinstance(softmax_scale, (float, int)) or not math.isfinite(float(softmax_scale)):
        raise TypeError("softmax_scale must be a finite Python float.")
    if float(softmax_scale) <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}.")
    if topk_length is not None:
        if topk_length.shape != (q.shape[0],) or topk_length.dtype != torch.int32 or topk_length.device != q.device:
            raise ValueError(
                "topk_length must be an int32 tensor on the query device with shape "
                f"{(q.shape[0],)}, got shape={tuple(topk_length.shape)}, "
                f"dtype={topk_length.dtype}, device={topk_length.device}."
            )
        if not topk_length.is_contiguous():
            raise ValueError("topk_length must be contiguous.")

    padded_heads = _padded_head_count(q.shape[1], major)
    return _CudnnSparseAttention.apply(
        q.contiguous(),
        kv_latent.contiguous(),
        topk_indices.contiguous(),
        float(softmax_scale),
        padded_heads,
        topk_length,
    )


__all__ = [
    "CudnnDsaPackedMetadata",
    "cudnn_indexer_topk",
    "cudnn_sparse_attention",
    "is_cudnn_dsa_available",
    "prepare_cudnn_dsa_packed_metadata",
]
