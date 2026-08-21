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
# Derived from the Miles DeepSeek-V4 sparse-MLA forward kernel already vendored
# in ``nemo_automodel.components.models.deepseek_v4.kernels``:
#   Project: https://github.com/yueming-yuan/miles
#   Revision: e561465d0b9bbf06188b7a5e2020dc7fd691f732
#   Source: miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py
#   License: Apache-2.0
#   Upstream copyright: Copyright 2025 Zhipu AI
# This Qwen4-Exp adaptation separates K from V, maps GQA KV heads onto a
# pseudo-batch, and specializes the head dimension to 256.

"""TileLang forward kernel for Qwen4-Exp token-indexed sparse GQA."""

import math

import torch

from nemo_automodel.components.models.qwen4_exp.kernels._tilelang import T, tilelang

_HEAD_DIM = 256
_TOPK_TILE = 64
_MIN_PADDED_HEADS = 16
_MAX_PADDED_HEADS = 64


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_gqa_fwd(
    heads: int,
    dim: int,
    topk: int,
    softmax_scale: float | None = None,
    block_size: int = _TOPK_TILE,
    num_stages: int = 2,
    threads: int = 256,
):
    """Build the pseudo-batch sparse-attention forward kernel."""
    assert dim == tilelang.math.next_power_of_2(dim), f"head dimension must be a power of two, got {dim}"
    assert heads == tilelang.math.next_power_of_2(heads), f"query heads must be a power of two, got {heads}"
    assert _MIN_PADDED_HEADS <= heads <= _MAX_PADDED_HEADS, f"unsupported padded query-head count {heads}"
    assert topk > 0 and topk % block_size == 0, f"top-k width {topk} must be divisible by {block_size}"
    if softmax_scale is None:
        scale_log2 = dim**-0.5 * 1.4426950408889634
    else:
        scale_log2 = softmax_scale * 1.4426950408889634

    pseudo_batch = T.dynamic("pseudo_batch")
    query_length = T.dynamic("query_length")
    kv_length = T.dynamic("kv_length")

    q_shape = [pseudo_batch, query_length, heads, dim]
    kv_shape = [pseudo_batch, kv_length, dim]
    ids_shape = [pseudo_batch, query_length, topk]
    out_shape = [pseudo_batch, query_length, heads, dim]
    lse_shape = [pseudo_batch, query_length, heads]
    dtype = T.bfloat16
    accum_dtype = T.float32

    num_topk_tiles = tilelang.cdiv(topk, block_size)

    @T.prim_func
    def main(
        query: T.Tensor(q_shape, dtype),  # type: ignore
        key: T.Tensor(kv_shape, dtype),  # type: ignore
        value: T.Tensor(kv_shape, dtype),  # type: ignore
        token_ids: T.Tensor(ids_shape, T.int32),  # type: ignore
        valid_mask: T.Tensor(ids_shape, T.int32),  # type: ignore
        output: T.Tensor(out_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        """Stream selected K/V tiles and maintain an FP32 online softmax."""
        with T.Kernel(query_length, pseudo_batch, threads=threads) as (query_idx, batch_idx):
            query_shared = T.alloc_shared([heads, dim], dtype)
            key_shared = T.alloc_shared([block_size, dim], dtype)
            value_shared = T.alloc_shared([block_size, dim], dtype)
            probability_shared = T.alloc_shared([heads, block_size], dtype)
            tile_valid = T.alloc_fragment([block_size], "bool")

            output_acc = T.alloc_fragment([heads, dim], accum_dtype)
            score_acc = T.alloc_fragment([heads, block_size], accum_dtype)
            tile_sum = T.alloc_fragment([heads], accum_dtype)
            running_sum = T.alloc_fragment([heads], accum_dtype)
            running_max = T.alloc_fragment([heads], accum_dtype)
            previous_max = T.alloc_fragment([heads], accum_dtype)
            correction = T.alloc_fragment([heads], accum_dtype)
            safe_denominator = T.alloc_fragment([heads], accum_dtype)

            T.fill(output_acc, 0)
            T.fill(running_sum, 0)
            T.fill(running_max, -(2**30))
            T.copy(query[batch_idx, query_idx, :, :], query_shared)

            for tile_idx in T.Pipelined(num_topk_tiles, num_stages=num_stages):
                for row_idx in T.Parallel(block_size):
                    tile_valid[row_idx] = valid_mask[batch_idx, query_idx, tile_idx * block_size + row_idx] != 0

                for row_idx, dim_idx in T.Parallel(block_size, dim):
                    selected_idx = token_ids[batch_idx, query_idx, tile_idx * block_size + row_idx]
                    key_shared[row_idx, dim_idx] = key[batch_idx, selected_idx, dim_idx]
                    value_shared[row_idx, dim_idx] = value[batch_idx, selected_idx, dim_idx]

                T.clear(score_acc)
                T.gemm(
                    query_shared,
                    key_shared,
                    score_acc,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for head_idx, row_idx in T.Parallel(heads, block_size):
                    score_acc[head_idx, row_idx] = T.if_then_else(
                        tile_valid[row_idx], score_acc[head_idx, row_idx], -T.infinity(accum_dtype)
                    )

                T.copy(running_max, previous_max)
                T.reduce_max(score_acc, running_max, dim=1, clear=False)
                for head_idx in T.Parallel(heads):
                    running_max[head_idx] = T.max(running_max[head_idx], previous_max[head_idx])
                    correction[head_idx] = T.exp2((previous_max[head_idx] - running_max[head_idx]) * scale_log2)
                for head_idx, row_idx in T.Parallel(heads, block_size):
                    score_acc[head_idx, row_idx] = T.exp2(
                        (score_acc[head_idx, row_idx] - running_max[head_idx]) * scale_log2
                    )
                T.reduce_sum(score_acc, tile_sum, dim=1)
                for head_idx in T.Parallel(heads):
                    running_sum[head_idx] = running_sum[head_idx] * correction[head_idx] + tile_sum[head_idx]
                for head_idx, dim_idx in T.Parallel(heads, dim):
                    output_acc[head_idx, dim_idx] *= correction[head_idx]

                T.copy(score_acc, probability_shared)
                T.gemm(
                    probability_shared,
                    value_shared,
                    output_acc,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # A right-padding query can have no valid IDs. Dividing by one keeps
            # its already-zero output finite; its LSE is explicitly -inf.
            for head_idx in T.Parallel(heads):
                safe_denominator[head_idx] = T.if_then_else(running_sum[head_idx] > 0, running_sum[head_idx], 1.0)
            for head_idx, dim_idx in T.Parallel(heads, dim):
                output_acc[head_idx, dim_idx] /= safe_denominator[head_idx]
            for head_idx in T.Parallel(heads):
                running_sum[head_idx] = T.if_then_else(
                    running_sum[head_idx] > 0,
                    T.log2(running_sum[head_idx]) + running_max[head_idx] * scale_log2,
                    -T.infinity(accum_dtype),
                )

            T.copy(output_acc, output[batch_idx, query_idx, :, :])
            T.copy(running_sum, lse[batch_idx, query_idx, :])

    return main


def _padded_head_count(heads: int) -> int:
    padded = max(_MIN_PADDED_HEADS, 1 << (heads - 1).bit_length())
    if padded > _MAX_PADDED_HEADS:
        raise NotImplementedError(
            f"Qwen4-Exp TileLang sparse GQA supports at most {_MAX_PADDED_HEADS} query heads per KV head; got {heads}"
        )
    return padded


def _validate_scale(softmax_scale: float | None, dim: int) -> float:
    if softmax_scale is None:
        return dim**-0.5
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int)):
        raise TypeError(f"softmax_scale must be a finite positive number, got {softmax_scale!r}")
    scale = float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}")
    return scale


def sparse_gqa_fwd_interface(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    block_size: int = _TOPK_TILE,
    num_stages: int = 2,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run sparse attention for one pseudo-batch row per original KV head.

    Args:
        query: BF16 queries shaped ``[B_pseudo, S_q, G, 256]``. ``G`` need
            not be padded; Qwen's 12 heads per KV head are padded to 16 here.
        key: BF16 keys shaped ``[B_pseudo, S_kv, 256]``.
        value: BF16 values shaped ``[B_pseudo, S_kv, 256]``.
        token_ids: Selected token IDs shaped ``[B_pseudo, S_q, K]``. Negative
            and out-of-range entries are masked before their safely clamped
            IDs reach the kernel.
        softmax_scale: Score multiplier, defaulting to ``1 / sqrt(256)``.
        block_size: Number of selected tokens streamed per kernel tile.
        num_stages: TileLang software-pipeline stage count.
        threads: Threads in each query-row CTA.

    Returns:
        A pair ``(output, lse_log2)`` with shapes ``[B_pseudo, S_q, G, 256]``
        and ``[B_pseudo, S_q, G]``. Empty-ID rows are exactly zero with
        ``-inf`` LSE. LSE is in base two, matching the companion backward.
    """
    if query.ndim != 4 or key.ndim != 3 or value.ndim != 3 or token_ids.ndim != 3:
        raise ValueError("pseudo-batch Q/K/V/IDs must have ranks 4/3/3/3")
    pseudo_batch, query_length, heads, dim = query.shape
    kv_batch, kv_length, key_dim = key.shape
    if value.shape != key.shape:
        raise ValueError(f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
    if (kv_batch, key_dim) != (pseudo_batch, dim):
        raise ValueError("pseudo-batch and head dimensions must match across Q/K/V")
    if token_ids.shape[:2] != (pseudo_batch, query_length):
        raise ValueError(f"token_ids must start with {(pseudo_batch, query_length)}, got {tuple(token_ids.shape)}")
    if dim != _HEAD_DIM:
        raise NotImplementedError(f"Qwen4-Exp TileLang sparse GQA requires head_dim={_HEAD_DIM}, got {dim}")
    if heads < 1 or query_length < 1 or kv_length < 1 or token_ids.shape[-1] < 1:
        raise ValueError("query heads, sequence lengths, and top-k width must all be positive")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise TypeError(f"query, key, and value must be bfloat16, got {query.dtype}, {key.dtype}, and {value.dtype}")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must be int32 or int64, got {token_ids.dtype}")
    tensors = (query, key, value, token_ids)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("Qwen4-Exp TileLang sparse GQA requires CUDA tensors")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("query, key, value, and token_ids must share one CUDA device")
    if block_size != _TOPK_TILE:
        raise NotImplementedError(f"only a {_TOPK_TILE}-token top-k tile is currently supported")
    scale = _validate_scale(softmax_scale, dim)

    padded_heads = _padded_head_count(heads)
    if padded_heads != heads:
        padded_query = query.new_zeros((pseudo_batch, query_length, padded_heads, dim))
        padded_query[:, :, :heads] = query
    else:
        padded_query = query
    padded_query = padded_query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    topk = token_ids.shape[-1]
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        padded_ids = torch.full(
            (pseudo_batch, query_length, padded_topk),
            -1,
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        padded_ids[:, :, :topk] = token_ids
    else:
        padded_ids = token_ids
    valid_mask = ((padded_ids >= 0) & (padded_ids < kv_length)).to(torch.int32).contiguous()
    safe_ids = padded_ids.clamp(min=0, max=kv_length - 1).to(torch.int32).contiguous()

    kernel = sparse_gqa_fwd(
        padded_heads,
        dim,
        padded_topk,
        scale,
        block_size=block_size,
        num_stages=num_stages,
        threads=threads,
    )
    output, lse = kernel(padded_query, key, value, safe_ids, valid_mask)
    return output[:, :, :heads].contiguous(), lse[:, :, :heads].contiguous()


def tilelang_sparse_gqa_fwd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Qwen4-Exp sparse GQA from its native BSHD tensors.

    This wrapper turns each KV head into a pseudo-batch row, so one fused MQA
    kernel handles each 12-query-head Qwen GQA group without repeating K/V to
    24 heads or materializing ``[S_q, K, H, D]`` gathered tensors.

    Args:
        query: BF16 queries shaped ``[B, S_q, H_q, 256]``.
        key: BF16 keys shaped ``[B, S_kv, H_kv, 256]``.
        value: BF16 values shaped ``[B, S_kv, H_kv, 256]``.
        token_ids: Shared QSA routing IDs shaped ``[B, S_q, K]``.
        softmax_scale: Score multiplier, defaulting to ``1 / sqrt(256)``.

    Returns:
        A pair ``(output, lse_log2)`` shaped ``[B, S_q, H_q, 256]`` and
        ``[B, S_q, H_q]``.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or token_ids.ndim != 3:
        raise ValueError("native Q/K/V/IDs must have ranks 4/4/4/3")
    batch, query_length, query_heads, dim = query.shape
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
    key_batch, kv_length, kv_heads, key_dim = key.shape
    if (key_batch, key_dim) != (batch, dim):
        raise ValueError("batch and head dimensions must match across Q/K/V")
    if token_ids.shape[:2] != (batch, query_length):
        raise ValueError(f"token_ids must start with {(batch, query_length)}, got {tuple(token_ids.shape)}")
    if kv_heads < 1 or query_heads % kv_heads:
        raise ValueError(f"H_q must be divisible by H_kv, got H_q={query_heads}, H_kv={kv_heads}")

    heads_per_kv = query_heads // kv_heads
    pseudo_query = (
        query.unflatten(2, (kv_heads, heads_per_kv))
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * kv_heads, query_length, heads_per_kv, dim)
        .contiguous()
    )
    pseudo_key = key.permute(0, 2, 1, 3).reshape(batch * kv_heads, kv_length, dim).contiguous()
    pseudo_value = value.permute(0, 2, 1, 3).reshape(batch * kv_heads, kv_length, dim).contiguous()
    pseudo_ids = (
        token_ids.unsqueeze(1)
        .expand(batch, kv_heads, query_length, token_ids.shape[-1])
        .reshape(batch * kv_heads, query_length, token_ids.shape[-1])
        .contiguous()
    )

    pseudo_output, pseudo_lse = sparse_gqa_fwd_interface(
        pseudo_query,
        pseudo_key,
        pseudo_value,
        pseudo_ids,
        softmax_scale,
    )
    output = (
        pseudo_output.unflatten(0, (batch, kv_heads))
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, query_length, query_heads, dim)
        .contiguous()
    )
    lse = (
        pseudo_lse.unflatten(0, (batch, kv_heads))
        .permute(0, 2, 1, 3)
        .reshape(batch, query_length, query_heads)
        .contiguous()
    )
    return output, lse


__all__ = ["sparse_gqa_fwd", "sparse_gqa_fwd_interface", "tilelang_sparse_gqa_fwd"]
