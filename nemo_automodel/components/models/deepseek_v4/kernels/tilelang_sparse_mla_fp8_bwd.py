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
#
# ruff: noqa
#
# The sparse-attention structure is adapted from the vendored Miles DSV4
# kernel in tilelang_sparse_mla_bwd.py (Apache-2.0, Zhipu AI).  The mixed
# FP8/BF16 KV loader and straight-through BF16 KV gradient are NVIDIA additions.

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang
from nemo_automodel.components.models.deepseek_v4.kernels.tilelang_sparse_mla_bwd import (
    postprocess,
    preprocess,
)


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def bwd_fp8(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    topk,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    threads=128,
):
    dim = 512
    assert topk % block_size == 0
    if sm_scale is None:
        sm_scale = dim ** (-0.5)
    sm_scale_log2e = sm_scale * 1.44269504

    q_shape = [batch, seq_len, heads, dim]
    nope_shape = [batch, seq_len_kv, 448]
    rope_shape = [batch, seq_len_kv, 64]
    scale_shape = [batch, seq_len_kv, 8]
    kv_grad_shape = [batch, seq_len_kv, dim]
    indices_shape = [batch, seq_len, topk]
    delta_shape = [batch, seq_len, heads]
    sink_shape = [heads]

    padded_heads = max(tilelang.math.next_power_of_2(heads), 16)
    block_heads = min(64, padded_heads)
    assert padded_heads % block_heads == 0
    num_head_blocks = padded_heads // block_heads
    num_topk_blocks = tilelang.cdiv(topk, block_size)
    split_store = 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),
        KVNope: T.Tensor(nope_shape, T.float8_e4m3fn),
        KVRope: T.Tensor(rope_shape, T.bfloat16),
        KVScales: T.Tensor(scale_shape, T.uint8),
        dO: T.Tensor(q_shape, T.bfloat16),
        AttnSink: T.Tensor(sink_shape, T.float32),
        Indices: T.Tensor(indices_shape, T.int32),
        ValidMask: T.Tensor(indices_shape, T.int32),
        Lse: T.Tensor(delta_shape, T.float32),
        Delta: T.Tensor(delta_shape, T.float32),
        dQ: T.Tensor(q_shape, T.bfloat16),
        dKV: T.Tensor(kv_grad_shape, T.float32),
        dAttnSink: T.Tensor(sink_shape, T.float32),
    ):
        with T.Kernel(seq_len, batch, num_head_blocks, threads=threads) as (seq_idx, batch_idx, head_block):
            q_shared = T.alloc_shared([block_heads, dim], T.bfloat16)
            kv_shared = T.alloc_shared([block_size, dim], T.bfloat16)
            do_shared = T.alloc_shared([block_heads, dim], T.bfloat16)
            mask = T.alloc_fragment([block_size], "bool")
            valid_count = T.alloc_fragment([1], T.int32)

            p_shared = T.alloc_shared([block_heads, block_size], T.bfloat16)
            dp_shared = T.alloc_shared([block_heads, block_size], T.bfloat16)
            dq_shared = T.alloc_shared([block_heads, dim], T.bfloat16)

            acc_p = T.alloc_fragment([block_heads, block_size], T.float32)
            acc_dp = T.alloc_fragment([block_heads, block_size], T.float32)
            acc_dq = T.alloc_fragment([block_heads, dim], T.float32)
            acc_dq_i = T.alloc_fragment([block_heads, dim], T.float32)
            acc_dkv = T.alloc_fragment([block_size, dim], T.float32)
            acc_dkv_shared = T.alloc_shared([block_size // split_store, dim], T.float32)

            head_start = head_block * block_heads
            head_end = (head_block + 1) * block_heads
            T.copy(Q[batch_idx, seq_idx, head_start:head_end, :], q_shared)
            T.copy(dO[batch_idx, seq_idx, head_start:head_end, :], do_shared)
            T.clear(acc_dq)

            for topk_block in T.Pipelined(num_topk_blocks, num_stages=num_stages):
                T.clear(valid_count)
                for row in T.Parallel(block_size):
                    mask[row] = ValidMask[batch_idx, seq_idx, topk_block * block_size + row] != 0
                for row in T.serial(block_size):
                    valid_count[0] += T.if_then_else(mask[row], 1, 0)

                if valid_count[0] != 0:
                    for row, d in T.Parallel(block_size, dim):
                        kv_idx = Indices[batch_idx, seq_idx, topk_block * block_size + row]
                        if d < 448:
                            encoded_scale = KVScales[batch_idx, kv_idx, d // 64]
                            descale = T.reinterpret(T.cast(encoded_scale, T.uint32) << 23, T.float32)
                            kv_shared[row, d] = T.cast(KVNope[batch_idx, kv_idx, d], T.float32) * descale
                        else:
                            kv_shared[row, d] = KVRope[batch_idx, kv_idx, d - 448]

                    T.clear(acc_p)
                    T.gemm(q_shared, kv_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    for head, row in T.Parallel(block_heads, block_size):
                        acc_p[head, row] = T.if_then_else(
                            mask[row],
                            T.exp2(acc_p[head, row] * sm_scale_log2e - Lse[batch_idx, seq_idx, head_start + head]),
                            0,
                        )
                    T.copy(acc_p, p_shared)

                    T.gemm(
                        do_shared,
                        kv_shared,
                        acc_dp,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    for head, row in T.Parallel(block_heads, block_size):
                        acc_dp[head, row] = (
                            acc_p[head, row]
                            * (acc_dp[head, row] - Delta[batch_idx, seq_idx, head_start + head])
                            * sm_scale
                        )
                    T.copy(acc_dp, dp_shared)

                    T.gemm(dp_shared, kv_shared, acc_dq_i, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                    for head, d in T.Parallel(block_heads, dim):
                        acc_dq[head, d] += acc_dq_i[head, d]

                    T.gemm(
                        dp_shared,
                        q_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    T.gemm(p_shared, do_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                    for split in range(split_store):
                        for row, d in T.Parallel(block_size, dim):
                            if row < block_size // split_store:
                                acc_dkv_shared[row, d] = acc_dkv[row + split * (block_size // split_store), d]
                        for row, d4 in T.Parallel(block_size // split_store, dim // 4):
                            source_row = row + split * (block_size // split_store)
                            if ValidMask[batch_idx, seq_idx, topk_block * block_size + source_row] != 0:
                                T.atomic_addx4(
                                    dKV[
                                        batch_idx,
                                        Indices[batch_idx, seq_idx, topk_block * block_size + source_row],
                                        d4 * 4,
                                    ],
                                    acc_dkv_shared[row, d4 * 4],
                                )

            T.copy(acc_dq, dq_shared)
            T.copy(dq_shared, dQ[batch_idx, seq_idx, head_start:head_end, :])
            for head in T.Parallel(block_heads):
                T.atomic_add(
                    dAttnSink[head_start + head],
                    -Delta[batch_idx, seq_idx, head_start + head]
                    * T.exp2(AttnSink[head_start + head] * 1.44269504 - Lse[batch_idx, seq_idx, head_start + head]),
                )

    return main


def sparse_mqa_fp8_bwd_interface(
    q,
    kv_nope,
    kv_rope,
    kv_scales,
    attn_sink,
    output,
    grad_output,
    topk_idxs,
    lse,
    sm_scale=None,
    return_dkv_accum_dtype=False,
):
    assert q.is_contiguous() and kv_nope.is_contiguous()
    assert kv_rope.is_contiguous() and kv_scales.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    batch, seq_len, heads, dim = q.shape
    assert dim == 512
    _, seq_len_kv, _ = kv_nope.shape
    topk = topk_idxs.shape[-1]
    block_size = 32
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = torch.full(
            (batch, seq_len, padded_topk - topk),
            -1,
            device=topk_idxs.device,
            dtype=topk_idxs.dtype,
        )
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk
    valid_mask = ((topk_idxs >= 0) & (topk_idxs < seq_len_kv)).to(torch.int32).contiguous()
    topk_idxs = topk_idxs.clamp(min=0, max=max(seq_len_kv - 1, 0)).to(torch.int32).contiguous()

    delta = preprocess(batch, seq_len, heads, dim)(output, grad_output)
    dkv = torch.zeros((batch, seq_len_kv, dim), dtype=torch.float32, device=q.device)
    d_attn_sink = torch.zeros_like(attn_sink)
    dq = bwd_fp8(batch, seq_len, seq_len_kv, heads, topk, sm_scale)(
        q,
        kv_nope,
        kv_rope,
        kv_scales,
        grad_output,
        attn_sink,
        topk_idxs,
        valid_mask,
        lse,
        delta,
        dkv,
        d_attn_sink,
    )
    if not return_dkv_accum_dtype:
        dkv = postprocess(batch, seq_len_kv, dim)(dkv)
    return dq, dkv, d_attn_sink
