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

"""Direct packed-THD TileLang backward kernel for Qwen4 sparse GQA."""

import math

import torch

from nemo_automodel.components.models.qwen4_exp.kernels._tilelang import T, tilelang

_QUERY_HEADS = 24
_KV_HEADS = 2
_GROUP_HEADS = _QUERY_HEADS // _KV_HEADS
_PADDED_GROUP_HEADS = 16
_HEAD_DIM = 256
_MAX_TOPK = 2051


def _normalize_token_ids(token_ids: torch.Tensor, tokens: int) -> torch.Tensor:
    """Normalize a nonempty shared THD route to ``[T, K]``.

    Args:
        token_ids: Route IDs shaped ``[T, K]`` or ``[T, 1, K]``.
        tokens: Expected physical token count ``T``.

    Returns:
        The nonempty rank-two route view consumed by backward.
    """
    if token_ids.ndim == 3:
        if token_ids.shape[1] != 1:
            raise ValueError(f"rank-3 THD token_ids must be [T, 1, K], got {tuple(token_ids.shape)}")
        token_ids = token_ids[:, 0, :]
    if token_ids.ndim != 2 or token_ids.shape[0] != tokens or token_ids.shape[1] < 1:
        raise ValueError(f"token_ids must have nonempty shape [T, K] matching T={tokens}, got {tuple(token_ids.shape)}")
    return token_ids


def _validate_scale(softmax_scale: float | None) -> float:
    """Return a finite positive QK scale."""
    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))):
        raise TypeError(f"softmax_scale must be a finite positive number, got {softmax_scale!r}")
    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}")
    return scale


@tilelang.jit(out_idx=[-1])
def sparse_gqa_thd_bwd_preprocess(
    tokens,
    query_heads=_QUERY_HEADS,
    kv_heads=_KV_HEADS,
    group_heads=_GROUP_HEADS,
    padded_group_heads=_PADDED_GROUP_HEADS,
    dim=_HEAD_DIM,
    block_nd=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Build padded-group ``delta = sum(output * doutput, dim=-1)``."""
    assert query_heads == kv_heads * group_heads
    assert padded_group_heads >= group_heads
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    tensor_shape = [tokens, query_heads, dim]
    delta_shape = [tokens, kv_heads, padded_group_heads]

    @T.prim_func
    def preprocess_kernel(
        Output: T.Tensor(tensor_shape, dtype),
        dOutput: T.Tensor(tensor_shape, dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
    ):
        with T.Kernel(query_heads, T.ceildiv(tokens, block_nd)) as (
            head_idx,
            token_block,
        ):
            output_fragment = T.alloc_fragment([block_nd, block_nd], accum_dtype)
            doutput_fragment = T.alloc_fragment([block_nd, block_nd], accum_dtype)
            product = T.alloc_fragment([block_nd, block_nd], accum_dtype)
            delta = T.alloc_fragment([block_nd], accum_dtype)
            kv_head_idx = head_idx // group_heads
            group_head_idx = head_idx % group_heads
            T.clear(product)

            for dim_block in T.Pipelined(T.ceildiv(dim, block_nd), num_stages=num_stages):
                T.copy(
                    Output[
                        token_block * block_nd : (token_block + 1) * block_nd,
                        head_idx,
                        dim_block * block_nd : (dim_block + 1) * block_nd,
                    ],
                    output_fragment,
                )
                T.copy(
                    dOutput[
                        token_block * block_nd : (token_block + 1) * block_nd,
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
                    token_block * block_nd : (token_block + 1) * block_nd,
                    kv_head_idx,
                    group_head_idx,
                ],
            )

            # The physical-head-zero CTA for each KV group also initializes
            # the four internal padding lanes. This avoids an extra memset or
            # extra preprocessing CTAs while making padded dK/dV contributions
            # provably zero.
            if group_head_idx == 0:
                for row_idx, pad_idx in T.Parallel(block_nd, padded_group_heads - group_heads):
                    token_idx = token_block * block_nd + row_idx
                    if token_idx < tokens:
                        Delta[token_idx, kv_head_idx, group_heads + pad_idx] = 0

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def sparse_gqa_thd_bwd_cast(
    tokens,
    kv_heads=_KV_HEADS,
    dim=_HEAD_DIM,
    block_n=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    """Cast direct-layout FP32 KV-gradient accumulators to BF16."""
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    tensor_shape = [tokens, kv_heads, dim]

    @T.prim_func
    def cast_kernel(
        Gradient: T.Tensor(tensor_shape, accum_dtype),
        CastGradient: T.Tensor(tensor_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(tokens, block_n), kv_heads, threads=threads) as (
            token_block,
            kv_head_idx,
        ):
            T.copy(
                Gradient[token_block * block_n : (token_block + 1) * block_n, kv_head_idx, :],
                CastGradient[token_block * block_n : (token_block + 1) * block_n, kv_head_idx, :],
            )

    return cast_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_gqa_thd_bwd(
    tokens,
    query_heads,
    kv_heads,
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
    """Build a one-CTA-per-``(token, KV head)`` THD backward kernel."""
    assert query_heads == _QUERY_HEADS
    assert kv_heads == _KV_HEADS
    assert query_heads // kv_heads == _GROUP_HEADS
    assert dim == _HEAD_DIM
    assert topk % block_size == 0
    assert dim % 4 == 0
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32
    if softmax_scale is None:
        softmax_scale = dim ** (-0.5)
    softmax_scale_log2e = softmax_scale * 1.44269504

    query_shape = [tokens, query_heads, dim]
    kv_shape = [tokens, kv_heads, dim]
    ids_shape = [tokens, topk]
    lse_shape = [tokens, kv_heads, _PADDED_GROUP_HEADS]
    num_sparse_blocks = tilelang.cdiv(topk, block_size)
    split_store = 2

    @T.prim_func
    def backward_kernel(
        Query: T.Tensor(query_shape, dtype),
        Key: T.Tensor(kv_shape, dtype),
        Value: T.Tensor(kv_shape, dtype),
        dOutput: T.Tensor(query_shape, dtype),
        TokenIds: T.Tensor(ids_shape, indices_dtype),
        ValidMask: T.Tensor(ids_shape, T.int32),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(lse_shape, accum_dtype),
        dQuery: T.Tensor(query_shape, dtype),
        dKey: T.Tensor(kv_shape, accum_dtype),
        dValue: T.Tensor(kv_shape, accum_dtype),
    ):
        with T.Kernel(tokens, kv_heads, threads=threads) as (token_idx, kv_head_idx):
            query_shared = T.alloc_shared([_PADDED_GROUP_HEADS, dim], dtype)
            key_shared = T.alloc_shared([block_size, dim], dtype)
            value_shared = T.alloc_shared([block_size, dim], dtype)
            doutput_shared = T.alloc_shared([_PADDED_GROUP_HEADS, dim], dtype)

            valid = T.alloc_fragment([block_size], T.int32)
            valid_count = T.alloc_fragment([1], T.int32)
            probability_shared = T.alloc_shared([_PADDED_GROUP_HEADS, block_size], dtype)
            dprobability_shared = T.alloc_shared([_PADDED_GROUP_HEADS, block_size], dtype)
            dquery_shared = T.alloc_shared([_PADDED_GROUP_HEADS, dim], dtype)

            probability = T.alloc_fragment([_PADDED_GROUP_HEADS, block_size], accum_dtype)
            dprobability = T.alloc_fragment([_PADDED_GROUP_HEADS, block_size], accum_dtype)
            dquery = T.alloc_fragment([_PADDED_GROUP_HEADS, dim], accum_dtype)
            dquery_tile = T.alloc_fragment([_PADDED_GROUP_HEADS, dim], accum_dtype)
            dkey_tile = T.alloc_fragment([block_size, dim], accum_dtype)
            dvalue_tile = T.alloc_fragment([block_size, dim], accum_dtype)
            dkey_shared = T.alloc_shared([block_size // split_store, dim], accum_dtype)
            dvalue_shared = T.alloc_shared([block_size // split_store, dim], accum_dtype)

            head_start = kv_head_idx * _GROUP_HEADS
            for pad_offset, dim_idx in T.Parallel(_PADDED_GROUP_HEADS - _GROUP_HEADS, dim):
                query_shared[_GROUP_HEADS + pad_offset, dim_idx] = 0
                doutput_shared[_GROUP_HEADS + pad_offset, dim_idx] = 0
            for head_offset, dim_idx in T.Parallel(_GROUP_HEADS, dim):
                query_shared[head_offset, dim_idx] = Query[token_idx, head_start + head_offset, dim_idx]
                doutput_shared[head_offset, dim_idx] = dOutput[token_idx, head_start + head_offset, dim_idx]
            T.clear(dquery)

            for sparse_block in T.Pipelined(num_sparse_blocks, num_stages=num_stages):
                T.clear(valid_count)
                for token_offset in T.Parallel(block_size):
                    valid[token_offset] = ValidMask[token_idx, sparse_block * block_size + token_offset]
                for token_offset in T.serial(block_size):
                    valid_count[0] += valid[token_offset]

                if valid_count[0] != 0:
                    for token_offset, dim_idx in T.Parallel(block_size, dim):
                        selected_idx = TokenIds[token_idx, sparse_block * block_size + token_offset]
                        key_shared[token_offset, dim_idx] = Key[selected_idx, kv_head_idx, dim_idx]
                        value_shared[token_offset, dim_idx] = Value[selected_idx, kv_head_idx, dim_idx]

                    T.gemm(
                        query_shared,
                        key_shared,
                        probability,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    for head_offset, token_offset in T.Parallel(_PADDED_GROUP_HEADS, block_size):
                        probability[head_offset, token_offset] = T.if_then_else(
                            valid[token_offset] != 0,
                            T.exp2(
                                probability[head_offset, token_offset] * softmax_scale_log2e
                                - Lse[token_idx, kv_head_idx, head_offset]
                            ),
                            0,
                        )
                    T.copy(probability, probability_shared)

                    T.gemm(
                        doutput_shared,
                        value_shared,
                        dprobability,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    for head_offset, token_offset in T.Parallel(_PADDED_GROUP_HEADS, block_size):
                        dprobability[head_offset, token_offset] = (
                            probability[head_offset, token_offset]
                            * (dprobability[head_offset, token_offset] - Delta[token_idx, kv_head_idx, head_offset])
                            * softmax_scale
                        )
                    T.copy(dprobability, dprobability_shared)

                    T.gemm(
                        dprobability_shared,
                        key_shared,
                        dquery_tile,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    for head_offset, dim_idx in T.Parallel(_PADDED_GROUP_HEADS, dim):
                        dquery[head_offset, dim_idx] += dquery_tile[head_offset, dim_idx]

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

                    for store_split in range(split_store):
                        for token_offset, dim_idx in T.Parallel(block_size, dim):
                            if token_offset < block_size // split_store:
                                source_offset = token_offset + store_split * (block_size // split_store)
                                dkey_shared[token_offset, dim_idx] = dkey_tile[source_offset, dim_idx]
                                dvalue_shared[token_offset, dim_idx] = dvalue_tile[source_offset, dim_idx]

                        for token_offset, dim_quad in T.Parallel(block_size // split_store, dim // 4):
                            ids_offset = token_offset + store_split * (block_size // split_store)
                            if ValidMask[token_idx, sparse_block * block_size + ids_offset] != 0:
                                selected_idx = TokenIds[token_idx, sparse_block * block_size + ids_offset]
                                T.atomic_addx4(
                                    dKey[selected_idx, kv_head_idx, dim_quad * 4],
                                    dkey_shared[token_offset, dim_quad * 4],
                                )
                                T.atomic_addx4(
                                    dValue[selected_idx, kv_head_idx, dim_quad * 4],
                                    dvalue_shared[token_offset, dim_quad * 4],
                                )

            T.copy(dquery, dquery_shared)
            for head_offset, dim_idx in T.Parallel(_GROUP_HEADS, dim):
                dQuery[token_idx, head_start + head_offset, dim_idx] = dquery_shared[head_offset, dim_idx]

    return backward_kernel


def sparse_gqa_thd_bwd_interface(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    doutput: torch.Tensor,
    token_ids: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale=None,
    *,
    return_kv_accum_dtype: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run direct THD backward with FP32 atomic dK/dV accumulation.

    Args:
        query: BF16 CUDA queries shaped ``[T, 24, 256]``.
        key: BF16 CUDA keys shaped ``[T, 2, 256]``.
        value: BF16 CUDA values shaped ``[T, 2, 256]``.
        output: BF16 forward output shaped ``[T, 24, 256]``.
        doutput: BF16 upstream gradient shaped ``[T, 24, 256]``.
        token_ids: INT32/INT64 CUDA route IDs shaped ``[T, K]`` or
            ``[T, 1, K]``, where ``1 <= K <= 2051``.
        lse: FP32 base-2 forward LSE shaped ``[T, 2, 16]``.
        softmax_scale: Positive QK scale, defaulting to ``1 / sqrt(256)``.
        return_kv_accum_dtype: Return FP32 dK/dV accumulators when true;
            otherwise cast them to BF16 to match K/V.

    Returns:
        Gradients for Q, K, and V in their corresponding direct THD layouts.
        Strided inputs are copied to contiguous storage before launch.
    """
    if query.ndim != 3 or query.shape[1:] != (_QUERY_HEADS, _HEAD_DIM):
        raise ValueError(f"query must be [T, {_QUERY_HEADS}, {_HEAD_DIM}], got {tuple(query.shape)}")
    tokens = query.shape[0]
    if tokens < 1:
        raise ValueError("direct Qwen4 THD requires at least one token")
    if key.shape != (tokens, _KV_HEADS, _HEAD_DIM) or value.shape != key.shape:
        raise ValueError(
            f"key/value must have matching [T, {_KV_HEADS}, {_HEAD_DIM}] shapes; "
            f"got key={tuple(key.shape)}, value={tuple(value.shape)}"
        )
    if output.shape != query.shape or doutput.shape != query.shape:
        raise ValueError(
            f"output and doutput must match query shape {tuple(query.shape)}; "
            f"got {tuple(output.shape)} and {tuple(doutput.shape)}"
        )
    if lse.shape != (tokens, _KV_HEADS, _PADDED_GROUP_HEADS):
        raise ValueError(f"lse must be [T, {_KV_HEADS}, {_PADDED_GROUP_HEADS}], got {tuple(lse.shape)}")
    token_ids = _normalize_token_ids(token_ids, tokens)
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must be int32 or int64, got {token_ids.dtype}")
    compute_tensors = (query, key, value, output, doutput)
    if any(tensor.dtype != torch.bfloat16 for tensor in compute_tensors):
        raise TypeError("direct Qwen4 THD backward requires BF16 compute tensors")
    if lse.dtype != torch.float32:
        raise TypeError(f"lse must be FP32, got {lse.dtype}")
    if not all(tensor.is_cuda for tensor in (*compute_tensors, token_ids, lse)):
        raise RuntimeError("direct Qwen4 THD backward requires CUDA tensors")
    if any(tensor.device != query.device for tensor in (*compute_tensors[1:], token_ids, lse)):
        raise ValueError("all direct Qwen4 THD backward tensors must share one CUDA device")
    scale = _validate_scale(softmax_scale)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = output.contiguous()
    doutput = doutput.contiguous()
    token_ids = token_ids.contiguous()
    lse = lse.contiguous()

    block_size = 32
    topk = token_ids.shape[-1]
    if topk > _MAX_TOPK:
        raise ValueError(f"Qwen4 THD sparse GQA supports at most {_MAX_TOPK} selected slots, got {topk}")
    padded_topk = math.ceil(topk / block_size) * block_size
    if padded_topk != topk:
        token_ids = torch.nn.functional.pad(token_ids, (0, padded_topk - topk), value=-1)
    valid_mask = ((token_ids >= 0) & (token_ids < tokens)).to(torch.int32).contiguous()
    safe_ids = token_ids.clamp(min=0, max=tokens - 1).to(torch.int32).contiguous()

    preprocess_kernel = sparse_gqa_thd_bwd_preprocess(tokens)
    delta = preprocess_kernel(output, doutput)
    backward_kernel = sparse_gqa_thd_bwd(
        tokens,
        _QUERY_HEADS,
        _KV_HEADS,
        _HEAD_DIM,
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
        safe_ids,
        valid_mask,
        lse,
        delta,
        dkey_accum,
        dvalue_accum,
    )
    if return_kv_accum_dtype:
        return dquery, dkey_accum, dvalue_accum
    cast_kernel = sparse_gqa_thd_bwd_cast(tokens)
    return dquery, cast_kernel(dkey_accum), cast_kernel(dvalue_accum)


__all__ = [
    "sparse_gqa_thd_bwd",
    "sparse_gqa_thd_bwd_cast",
    "sparse_gqa_thd_bwd_interface",
    "sparse_gqa_thd_bwd_preprocess",
]
