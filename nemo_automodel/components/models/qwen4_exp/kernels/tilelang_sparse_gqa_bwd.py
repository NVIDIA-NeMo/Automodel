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
# Derived from the Miles DeepSeek-V4 sparse-MLA backward kernel already
# vendored in ``nemo_automodel.components.models.deepseek_v4.kernels``:
#   Project: https://github.com/yueming-yuan/miles
#   Revision: e561465d0b9bbf06188b7a5e2020dc7fd691f732
#   Source: miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py
#   License: Apache-2.0
#   Upstream copyright: Copyright 2025 Zhipu AI
# This Qwen4-Exp adaptation separates K from V, computes native D=256 GQA
# gradients, and maps each physical KV head onto a pseudo-batch row.

# ruff: noqa
"""TileLang backward kernel for Qwen4-Exp token-level sparse GQA.

The low-level kernel consumes the pseudo-batch layout prepared by
``sparse_attention.py``: every physical KV head becomes one batch row, so the
kernel itself is sparse MQA with a small group of query heads. Unlike MLA, Qwen
has independent key and value tensors. The backward therefore recomputes the
selected-token probabilities and accumulates separate FP32 ``dK`` and ``dV``
buffers before casting them back to the input dtype.

This implementation follows the streaming backward structure used by the
vendored DeepSeek-V4 TileLang sparse-attention kernel, but keeps Qwen's native
``D=256`` Q/K/V representation instead of concatenating K and V.
"""

import math

import torch

from nemo_automodel.components.models.qwen4_exp.kernels._tilelang import T, tilelang


@tilelang.jit(out_idx=[-1])
def sparse_gqa_bwd_preprocess(
    batch,
    seq_len,
    heads,
    dim,
    block_nd=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Build ``delta = sum(output * doutput, dim=-1)`` in FP32."""
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    tensor_shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def preprocess_kernel(
        Output: T.Tensor(tensor_shape, dtype),
        dOutput: T.Tensor(tensor_shape, dtype),
        Delta: T.Tensor([batch, seq_len, heads], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_nd), batch) as (head_idx, seq_block, batch_idx):
            output_fragment = T.alloc_fragment([block_nd, block_nd], accum_dtype)
            doutput_fragment = T.alloc_fragment([block_nd, block_nd], accum_dtype)
            product = T.alloc_fragment([block_nd, block_nd], accum_dtype)
            delta = T.alloc_fragment([block_nd], accum_dtype)
            T.clear(product)

            for dim_block in T.Pipelined(T.ceildiv(dim, block_nd), num_stages=num_stages):
                T.copy(
                    Output[
                        batch_idx,
                        seq_block * block_nd : (seq_block + 1) * block_nd,
                        head_idx,
                        dim_block * block_nd : (dim_block + 1) * block_nd,
                    ],
                    output_fragment,
                )
                T.copy(
                    dOutput[
                        batch_idx,
                        seq_block * block_nd : (seq_block + 1) * block_nd,
                        head_idx,
                        dim_block * block_nd : (dim_block + 1) * block_nd,
                    ],
                    doutput_fragment,
                )
                for row_idx, dim_idx in T.Parallel(block_nd, block_nd):
                    product[row_idx, dim_idx] += output_fragment[row_idx, dim_idx] * doutput_fragment[row_idx, dim_idx]

            T.reduce_sum(product, delta, dim=1)
            T.copy(
                delta,
                Delta[
                    batch_idx,
                    seq_block * block_nd : (seq_block + 1) * block_nd,
                    head_idx,
                ],
            )

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def sparse_gqa_bwd_cast(
    batch,
    seq_len_kv,
    dim,
    block_n=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Cast one pseudo-batched FP32 KV-gradient buffer to BF16."""
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    tensor_shape = [batch, seq_len_kv, dim]

    @T.prim_func
    def cast_kernel(
        Gradient: T.Tensor(tensor_shape, accum_dtype),
        CastGradient: T.Tensor(tensor_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len_kv, block_n), batch, threads=threads) as (seq_block, batch_idx):
            T.copy(
                Gradient[batch_idx, seq_block * block_n : (seq_block + 1) * block_n, :],
                CastGradient[batch_idx, seq_block * block_n : (seq_block + 1) * block_n, :],
            )

    return cast_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_gqa_bwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    topk,
    softmax_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Create the fused sparse-GQA backward kernel."""
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert dim % 4 == 0, f"dim ({dim}) must be divisible by four for vectorized atomic stores"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if softmax_scale is None:
        softmax_scale = dim ** (-0.5)
    softmax_scale_log2e = softmax_scale * 1.44269504

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    ids_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]

    padded_heads = max(tilelang.math.next_power_of_2(heads), 16)
    block_heads = min(64, padded_heads)
    assert padded_heads % block_heads == 0
    num_head_blocks = padded_heads // block_heads
    num_sparse_blocks = tilelang.cdiv(topk, block_size)
    split_store = 2

    @T.prim_func
    def backward_kernel(
        Query: T.Tensor(q_shape, dtype),
        Key: T.Tensor(kv_shape, dtype),
        Value: T.Tensor(kv_shape, dtype),
        dOutput: T.Tensor(q_shape, dtype),
        TokenIds: T.Tensor(ids_shape, indices_dtype),
        ValidMask: T.Tensor(ids_shape, T.int32),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(lse_shape, accum_dtype),
        dQuery: T.Tensor(q_shape, dtype),
        dKey: T.Tensor(kv_shape, accum_dtype),
        dValue: T.Tensor(kv_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, num_head_blocks, threads=threads) as (seq_idx, batch_idx, head_block):
            query_shared = T.alloc_shared([block_heads, dim], dtype)
            key_shared = T.alloc_shared([block_size, dim], dtype)
            value_shared = T.alloc_shared([block_size, dim], dtype)
            doutput_shared = T.alloc_shared([block_heads, dim], dtype)

            valid = T.alloc_fragment([block_size], T.int32)
            valid_count = T.alloc_fragment([1], T.int32)

            probability_shared = T.alloc_shared([block_heads, block_size], dtype)
            dprobability_shared = T.alloc_shared([block_heads, block_size], dtype)
            dquery_shared = T.alloc_shared([block_heads, dim], dtype)

            probability = T.alloc_fragment([block_heads, block_size], accum_dtype)
            dprobability = T.alloc_fragment([block_heads, block_size], accum_dtype)
            dquery = T.alloc_fragment([block_heads, dim], accum_dtype)
            dquery_tile = T.alloc_fragment([block_heads, dim], accum_dtype)
            dkey_tile = T.alloc_fragment([block_size, dim], accum_dtype)
            dvalue_tile = T.alloc_fragment([block_size, dim], accum_dtype)
            dkey_shared = T.alloc_shared([block_size // split_store, dim], accum_dtype)
            dvalue_shared = T.alloc_shared([block_size // split_store, dim], accum_dtype)

            head_start = head_block * block_heads
            head_end = (head_block + 1) * block_heads
            T.copy(Query[batch_idx, seq_idx, head_start:head_end, :], query_shared)
            T.copy(dOutput[batch_idx, seq_idx, head_start:head_end, :], doutput_shared)
            T.clear(dquery)

            for sparse_block in T.Pipelined(num_sparse_blocks, num_stages=num_stages):
                T.clear(valid_count)
                for token_offset in T.Parallel(block_size):
                    valid[token_offset] = ValidMask[
                        batch_idx,
                        seq_idx,
                        sparse_block * block_size + token_offset,
                    ]
                for token_offset in T.serial(block_size):
                    valid_count[0] += valid[token_offset]

                # Skip a completely padded tile before loading its safe-but-irrelevant K/V rows.
                if valid_count[0] != 0:
                    for token_offset, dim_idx in T.Parallel(block_size, dim):
                        token_idx = TokenIds[
                            batch_idx,
                            seq_idx,
                            sparse_block * block_size + token_offset,
                        ]
                        key_shared[token_offset, dim_idx] = Key[batch_idx, token_idx, dim_idx]
                        value_shared[token_offset, dim_idx] = Value[batch_idx, token_idx, dim_idx]

                    T.gemm(
                        query_shared,
                        key_shared,
                        probability,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )

                    # P = exp(score * scale - LSE). Forward stores LSE in log2 space.
                    for head_offset, token_offset in T.Parallel(block_heads, block_size):
                        probability[head_offset, token_offset] = T.if_then_else(
                            valid[token_offset] != 0,
                            T.exp2(
                                probability[head_offset, token_offset] * softmax_scale_log2e
                                - Lse[batch_idx, seq_idx, head_start + head_offset]
                            ),
                            0,
                        )
                    T.copy(probability, probability_shared)

                    # dScore = P * (dO @ V^T - sum(O * dO)) * scale.
                    T.gemm(
                        doutput_shared,
                        value_shared,
                        dprobability,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    for head_offset, token_offset in T.Parallel(block_heads, block_size):
                        dprobability[head_offset, token_offset] = (
                            probability[head_offset, token_offset]
                            * (
                                dprobability[head_offset, token_offset]
                                - Delta[batch_idx, seq_idx, head_start + head_offset]
                            )
                            * softmax_scale
                        )
                    T.copy(dprobability, dprobability_shared)

                    # dQ += dScore @ K.
                    T.gemm(
                        dprobability_shared,
                        key_shared,
                        dquery_tile,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    for head_offset, dim_idx in T.Parallel(block_heads, dim):
                        dquery[head_offset, dim_idx] += dquery_tile[head_offset, dim_idx]

                    # dK = dScore^T @ Q; dV = P^T @ dO.
                    T.gemm(
                        dprobability_shared,
                        query_shared,
                        dkey_tile,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    T.gemm(
                        probability_shared,
                        doutput_shared,
                        dvalue_tile,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )

                    # Multiple queries and head tiles share each KV row, so accumulate in FP32.
                    for store_split in range(split_store):
                        for token_offset, dim_idx in T.Parallel(block_size, dim):
                            if token_offset < block_size // split_store:
                                source_offset = token_offset + store_split * (block_size // split_store)
                                dkey_shared[token_offset, dim_idx] = dkey_tile[source_offset, dim_idx]
                                dvalue_shared[token_offset, dim_idx] = dvalue_tile[source_offset, dim_idx]

                        for token_offset, dim_quad in T.Parallel(block_size // split_store, dim // 4):
                            ids_offset = token_offset + store_split * (block_size // split_store)
                            if (
                                ValidMask[
                                    batch_idx,
                                    seq_idx,
                                    sparse_block * block_size + ids_offset,
                                ]
                                != 0
                            ):
                                token_idx = TokenIds[
                                    batch_idx,
                                    seq_idx,
                                    sparse_block * block_size + ids_offset,
                                ]
                                T.atomic_addx4(
                                    dKey[batch_idx, token_idx, dim_quad * 4],
                                    dkey_shared[token_offset, dim_quad * 4],
                                )
                                T.atomic_addx4(
                                    dValue[batch_idx, token_idx, dim_quad * 4],
                                    dvalue_shared[token_offset, dim_quad * 4],
                                )

            T.copy(dquery, dquery_shared)
            T.copy(dquery_shared, dQuery[batch_idx, seq_idx, head_start:head_end, :])

    return backward_kernel


def sparse_gqa_bwd_interface(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    doutput: torch.Tensor,
    token_ids: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    return_kv_accum_dtype: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run native sparse-GQA backward on pseudo-batched tensors.

    Args:
        query: Pseudo-batched BF16 queries ``[Bp, S, Hpad, D]``.
        key: Pseudo-batched BF16 keys ``[Bp, S_kv, D]``.
        value: Pseudo-batched BF16 values ``[Bp, S_kv, D]``.
        output: Forward output ``[Bp, S, Hpad, D]``.
        doutput: Output gradient ``[Bp, S, Hpad, D]``.
        token_ids: Selected token IDs ``[Bp, S, K]``. ``-1`` marks padding.
        lse: Forward log-sum-exp in log2 space, ``[Bp, S, Hpad]`` FP32.
        softmax_scale: QK score scale. Defaults to ``1 / sqrt(D)``.
        return_kv_accum_dtype: Keep ``dK`` and ``dV`` in FP32 instead of
            casting them back to BF16.

    Returns:
        ``(dQ, dK, dV)`` with the same layouts as ``query``, ``key``, and
        ``value``. ``dK`` and ``dV`` are BF16 unless
        ``return_kv_accum_dtype=True``.
    """
    if query.ndim != 4:
        raise ValueError(f"query must have shape [Bp, S, H, D], got {tuple(query.shape)}")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError(
            "key and value must have matching [Bp, S_kv, D] shapes; "
            f"got key={tuple(key.shape)}, value={tuple(value.shape)}"
        )

    batch, seq_len, heads, dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != dim:
        raise ValueError("query, key, and value pseudo-batch/head dimensions must match")
    if batch < 1 or seq_len < 1:
        raise ValueError("pseudo-batch size and query length must be positive")
    if output.shape != query.shape or doutput.shape != query.shape:
        raise ValueError(
            "output and doutput must match query shape; "
            f"got query={tuple(query.shape)}, output={tuple(output.shape)}, doutput={tuple(doutput.shape)}"
        )
    if lse.shape != query.shape[:3]:
        raise ValueError(f"lse must have shape {tuple(query.shape[:3])}, got {tuple(lse.shape)}")
    if token_ids.ndim != 3 or token_ids.shape[:2] != (batch, seq_len) or token_ids.shape[-1] == 0:
        raise ValueError(
            f"token_ids must have nonempty shape [Bp, S, K] matching {(batch, seq_len)}, got {tuple(token_ids.shape)}"
        )
    if key.shape[1] == 0:
        raise ValueError("sparse GQA requires at least one key/value token")
    if dim != 256:
        raise ValueError(f"Qwen4-Exp sparse GQA requires head_dim=256, got {dim}")
    if heads < 16 or heads != 1 << (heads - 1).bit_length():
        raise ValueError(f"pseudo-batch query heads must be a power of two >= 16, got {heads}")

    compute_tensors = (query, key, value, output, doutput)
    if any(tensor.dtype != torch.bfloat16 for tensor in compute_tensors):
        raise TypeError("Qwen4-Exp TileLang sparse GQA backward requires BF16 compute tensors")
    if lse.dtype != torch.float32:
        raise TypeError(f"lse must be FP32, got {lse.dtype}")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must be int32 or int64, got {token_ids.dtype}")
    if not all(tensor.is_cuda for tensor in (*compute_tensors, token_ids, lse)):
        raise RuntimeError("Qwen4-Exp TileLang sparse GQA backward requires CUDA tensors")
    if any(tensor.device != query.device for tensor in (*compute_tensors[1:], token_ids, lse)):
        raise ValueError("all sparse-GQA backward tensors must be on the same device")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = output.contiguous()
    doutput = doutput.contiguous()
    token_ids = token_ids.contiguous()
    lse = lse.contiguous()

    block_size = 32
    topk = token_ids.shape[-1]
    padded_topk = math.ceil(topk / block_size) * block_size
    if padded_topk != topk:
        token_ids = torch.nn.functional.pad(token_ids, (0, padded_topk - topk), value=-1)

    seq_len_kv = key.shape[1]
    valid_mask = ((token_ids >= 0) & (token_ids < seq_len_kv)).to(torch.int32).contiguous()
    safe_token_ids = token_ids.clamp(min=0, max=seq_len_kv - 1).to(torch.int32).contiguous()

    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))):
        raise TypeError(f"softmax_scale must be a finite positive number, got {softmax_scale!r}")
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be a positive finite value, got {softmax_scale}")

    preprocess_kernel = sparse_gqa_bwd_preprocess(batch, seq_len, heads, dim)
    delta = preprocess_kernel(output, doutput)

    backward_kernel = sparse_gqa_bwd(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        padded_topk,
        scale,
        block_size=block_size,
    )
    dkey_accum = torch.zeros_like(key, dtype=torch.float32)
    dvalue_accum = torch.zeros_like(value, dtype=torch.float32)
    dquery = backward_kernel(
        query,
        key,
        value,
        doutput,
        safe_token_ids,
        valid_mask,
        lse,
        delta,
        dkey_accum,
        dvalue_accum,
    )

    if return_kv_accum_dtype:
        return dquery, dkey_accum, dvalue_accum

    cast_kernel = sparse_gqa_bwd_cast(batch, seq_len_kv, dim)
    return dquery, cast_kernel(dkey_accum), cast_kernel(dvalue_accum)
