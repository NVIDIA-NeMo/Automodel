# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Direct-THD implementation derived from the Qwen4 sparse-GQA kernel at Automodel commit
# a85bb6c776122d56b70604f0145c7de1b32a9dab. The source kernel is itself an
# Apache-2.0 adaptation of Miles revision
# e561465d0b9bbf06188b7a5e2020dc7fd691f732. Candidate planning follows Kernel
# Design Agents dda6be3cf1baedd3ed9c76511ef02f72243cc14c and KernelWiki
# 76d27b56f804e7e7295d4c570e1e5d7eef4b0a75.

"""Direct packed-THD TileLang forward kernel for Qwen4 sparse GQA.

The attention math deliberately does not consume ``cu_seqlens``. Packed
document boundaries belong to the QSA indexer: every nonnegative route is a
global flattened token ID that is already confined to the query document.
"""

import math

import torch

from nemo_automodel.components.models.qwen4_exp.kernels._tilelang import T, tilelang

_QUERY_HEADS = 24
_KV_HEADS = 2
_GROUP_HEADS = _QUERY_HEADS // _KV_HEADS
_PADDED_GROUP_HEADS = 16
_HEAD_DIM = 256
_TOPK_TILE = 64
_MAX_TOPK = 2051


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_gqa_thd_fwd(
    query_heads: int,
    kv_heads: int,
    dim: int,
    topk: int,
    softmax_scale=None,
    block_size: int = _TOPK_TILE,
    num_stages: int = 2,
    threads: int = 256,
):
    """Build a one-CTA-per-``(token, KV head)`` THD forward kernel."""
    assert query_heads == _QUERY_HEADS
    assert kv_heads == _KV_HEADS
    assert dim == _HEAD_DIM
    assert query_heads % kv_heads == 0
    assert query_heads // kv_heads == _GROUP_HEADS
    assert topk > 0 and topk % block_size == 0
    if softmax_scale is None:
        scale_log2 = dim**-0.5 * 1.4426950408889634
    else:
        scale_log2 = softmax_scale * 1.4426950408889634

    tokens = T.dynamic("tokens")
    query_shape = [tokens, query_heads, dim]
    kv_shape = [tokens, kv_heads, dim]
    ids_shape = [tokens, topk]
    output_shape = [tokens, query_heads, dim]
    # LSE retains the internal padded GQA group. It is private autograd state,
    # so no public padded output tensor is materialized.
    lse_shape = [tokens, kv_heads, _PADDED_GROUP_HEADS]
    dtype = T.bfloat16
    accum_dtype = T.float32
    num_topk_tiles = tilelang.cdiv(topk, block_size)

    @T.prim_func
    def main(
        # TileLang consumes these runtime tensor annotations; static type
        # checkers cannot resolve the DSL's dynamic shapes.
        query: T.Tensor(query_shape, dtype),  # type: ignore
        key: T.Tensor(kv_shape, dtype),  # type: ignore
        value: T.Tensor(kv_shape, dtype),  # type: ignore
        token_ids: T.Tensor(ids_shape, T.int32),  # type: ignore
        valid_mask: T.Tensor(ids_shape, T.int32),  # type: ignore
        output: T.Tensor(output_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(tokens, kv_heads, threads=threads) as (query_idx, kv_head_idx):
            query_shared = T.alloc_shared([_PADDED_GROUP_HEADS, dim], dtype)
            key_shared = T.alloc_shared([block_size, dim], dtype)
            value_shared = T.alloc_shared([block_size, dim], dtype)
            probability_shared = T.alloc_shared([_PADDED_GROUP_HEADS, block_size], dtype)
            tile_valid = T.alloc_fragment([block_size], "bool")

            output_acc = T.alloc_fragment([_PADDED_GROUP_HEADS, dim], accum_dtype)
            score_acc = T.alloc_fragment([_PADDED_GROUP_HEADS, block_size], accum_dtype)
            tile_sum = T.alloc_fragment([_PADDED_GROUP_HEADS], accum_dtype)
            running_sum = T.alloc_fragment([_PADDED_GROUP_HEADS], accum_dtype)
            running_max = T.alloc_fragment([_PADDED_GROUP_HEADS], accum_dtype)
            previous_max = T.alloc_fragment([_PADDED_GROUP_HEADS], accum_dtype)
            correction = T.alloc_fragment([_PADDED_GROUP_HEADS], accum_dtype)
            safe_denominator = T.alloc_fragment([_PADDED_GROUP_HEADS], accum_dtype)

            head_start = kv_head_idx * _GROUP_HEADS
            for pad_offset, dim_idx in T.Parallel(_PADDED_GROUP_HEADS - _GROUP_HEADS, dim):
                query_shared[_GROUP_HEADS + pad_offset, dim_idx] = 0
            for head_offset, dim_idx in T.Parallel(_GROUP_HEADS, dim):
                query_shared[head_offset, dim_idx] = query[query_idx, head_start + head_offset, dim_idx]

            T.fill(output_acc, 0)
            T.fill(running_sum, 0)
            T.fill(running_max, -(2**30))

            for tile_idx in T.Pipelined(num_topk_tiles, num_stages=num_stages):
                for row_idx in T.Parallel(block_size):
                    tile_valid[row_idx] = valid_mask[query_idx, tile_idx * block_size + row_idx] != 0

                for row_idx, dim_idx in T.Parallel(block_size, dim):
                    selected_idx = token_ids[query_idx, tile_idx * block_size + row_idx]
                    key_shared[row_idx, dim_idx] = key[selected_idx, kv_head_idx, dim_idx]
                    value_shared[row_idx, dim_idx] = value[selected_idx, kv_head_idx, dim_idx]

                T.clear(score_acc)
                T.gemm(
                    query_shared,
                    key_shared,
                    score_acc,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for head_offset, row_idx in T.Parallel(_PADDED_GROUP_HEADS, block_size):
                    score_acc[head_offset, row_idx] = T.if_then_else(
                        tile_valid[row_idx],
                        score_acc[head_offset, row_idx],
                        -T.infinity(accum_dtype),
                    )

                T.copy(running_max, previous_max)
                T.reduce_max(score_acc, running_max, dim=1, clear=False)
                for head_offset in T.Parallel(_PADDED_GROUP_HEADS):
                    running_max[head_offset] = T.max(running_max[head_offset], previous_max[head_offset])
                    correction[head_offset] = T.exp2(
                        (previous_max[head_offset] - running_max[head_offset]) * scale_log2
                    )
                for head_offset, row_idx in T.Parallel(_PADDED_GROUP_HEADS, block_size):
                    score_acc[head_offset, row_idx] = T.exp2(
                        (score_acc[head_offset, row_idx] - running_max[head_offset]) * scale_log2
                    )
                T.reduce_sum(score_acc, tile_sum, dim=1)
                for head_offset in T.Parallel(_PADDED_GROUP_HEADS):
                    running_sum[head_offset] = (
                        running_sum[head_offset] * correction[head_offset] + tile_sum[head_offset]
                    )
                for head_offset, dim_idx in T.Parallel(_PADDED_GROUP_HEADS, dim):
                    output_acc[head_offset, dim_idx] *= correction[head_offset]

                T.copy(score_acc, probability_shared)
                T.gemm(
                    probability_shared,
                    value_shared,
                    output_acc,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for head_offset in T.Parallel(_PADDED_GROUP_HEADS):
                safe_denominator[head_offset] = T.if_then_else(
                    running_sum[head_offset] > 0, running_sum[head_offset], 1.0
                )
            for head_offset, dim_idx in T.Parallel(_PADDED_GROUP_HEADS, dim):
                output_acc[head_offset, dim_idx] /= safe_denominator[head_offset]
            for head_offset in T.Parallel(_PADDED_GROUP_HEADS):
                running_sum[head_offset] = T.if_then_else(
                    running_sum[head_offset] > 0,
                    T.log2(running_sum[head_offset]) + running_max[head_offset] * scale_log2,
                    -T.infinity(accum_dtype),
                )

            for head_offset, dim_idx in T.Parallel(_GROUP_HEADS, dim):
                output[query_idx, head_start + head_offset, dim_idx] = output_acc[head_offset, dim_idx]
            T.copy(running_sum, lse[query_idx, kv_head_idx, :])

    return main


def _normalize_token_ids(token_ids: torch.Tensor, tokens: int) -> torch.Tensor:
    """Normalize the shared THD route without duplicating it.

    Args:
        token_ids: Route IDs shaped ``[T, K]`` or ``[T, 1, K]``.
        tokens: Expected physical token count ``T``.

    Returns:
        The nonempty rank-two route view with ``K <= 2051``.
    """
    if token_ids.ndim == 3:
        if token_ids.shape[1] != 1:
            raise ValueError(f"rank-3 THD token_ids must be [T, 1, K], got {tuple(token_ids.shape)}")
        token_ids = token_ids[:, 0, :]
    if token_ids.ndim != 2 or token_ids.shape[0] != tokens or token_ids.shape[1] < 1:
        raise ValueError(f"token_ids must have nonempty shape [T, K] matching T={tokens}, got {tuple(token_ids.shape)}")
    if token_ids.shape[1] > _MAX_TOPK:
        raise ValueError(f"Qwen4 THD sparse GQA supports at most {_MAX_TOPK} selected slots, got {token_ids.shape[1]}")
    return token_ids


def _validate_scale(softmax_scale) -> float:
    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))):
        raise TypeError(f"softmax_scale must be a finite positive number, got {softmax_scale!r}")
    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}")
    return scale


def sparse_gqa_thd_fwd_interface(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale=None,
    *,
    block_size: int = _TOPK_TILE,
    num_stages: int = 2,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run direct THD sparse-GQA forward without pseudo-batch copies.

    Args:
        query: BF16 CUDA queries shaped ``[T, 24, 256]``.
        key: BF16 CUDA keys shaped ``[T, 2, 256]``.
        value: BF16 CUDA values shaped ``[T, 2, 256]``.
        token_ids: INT32/INT64 CUDA route IDs shaped ``[T, K]`` or
            ``[T, 1, K]``, where ``1 <= K <= 2051``.
        softmax_scale: Positive QK scale, defaulting to ``1 / sqrt(256)``.
        block_size: Sparse route tile width. Only 64 is supported.
        num_stages: TileLang software-pipeline stage count.
        threads: CUDA threads per kernel block.

    Returns:
        A pair containing BF16 output ``[T, 24, 256]`` and FP32 base-2 LSE
        ``[T, 2, 16]``. Strided Q/K/V inputs are copied to contiguous storage.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("THD query/key/value must have ranks 3/3/3")
    tokens, query_heads, dim = query.shape
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
    if key.shape != (tokens, _KV_HEADS, dim):
        raise ValueError(f"key/value must be [T, {_KV_HEADS}, D] with the same T/D as query; got {tuple(key.shape)}")
    if (query_heads, dim) != (_QUERY_HEADS, _HEAD_DIM):
        raise ValueError(
            f"direct Qwen4 THD requires Hq={_QUERY_HEADS}, Hkv={_KV_HEADS}, D={_HEAD_DIM}; "
            f"got Hq={query_heads}, Hkv={key.shape[1]}, D={dim}"
        )
    if tokens < 1:
        raise ValueError("direct Qwen4 THD requires at least one token")
    token_ids = _normalize_token_ids(token_ids, tokens)
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must be int32 or int64, got {token_ids.dtype}")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise TypeError("direct Qwen4 THD sparse GQA requires BF16 query, key, and value")
    if not all(tensor.is_cuda for tensor in (query, key, value, token_ids)):
        raise RuntimeError("direct Qwen4 THD sparse GQA requires CUDA tensors")
    if any(tensor.device != query.device for tensor in (key, value, token_ids)):
        raise ValueError("query, key, value, and token_ids must share one CUDA device")
    if block_size != _TOPK_TILE:
        raise NotImplementedError(f"only a {_TOPK_TILE}-token forward tile is supported")
    scale = _validate_scale(softmax_scale)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    topk = token_ids.shape[-1]
    padded_topk = math.ceil(topk / block_size) * block_size
    if padded_topk != topk:
        token_ids = torch.nn.functional.pad(token_ids, (0, padded_topk - topk), value=-1)
    valid_mask = ((token_ids >= 0) & (token_ids < tokens)).to(torch.int32).contiguous()
    safe_ids = token_ids.clamp(min=0, max=tokens - 1).to(torch.int32).contiguous()

    kernel = sparse_gqa_thd_fwd(
        _QUERY_HEADS,
        _KV_HEADS,
        _HEAD_DIM,
        padded_topk,
        scale,
        block_size=block_size,
        num_stages=num_stages,
        threads=threads,
    )
    return kernel(query, key, value, safe_ids, valid_mask)


__all__ = ["sparse_gqa_thd_fwd", "sparse_gqa_thd_fwd_interface"]
