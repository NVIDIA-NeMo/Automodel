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

"""True-FP8 DSV4 indexer score kernel matching vLLM 0.21's Q/K boundary."""

# ruff: noqa

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang
from nemo_automodel.components.models.deepseek_v4.kernels.tilelang_indexer_fwd import clean_logits_


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tl_fp8_indexer_fwd_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    @T.prim_func
    def main(
        IndexQ: T.Tensor([seq_len * heads, index_dim], T.float8_e4m3fn),
        IndexK: T.Tensor([seq_len_kv, index_dim], T.float8_e4m3fn),
        QScales: T.Tensor([seq_len * heads, 1], T.float32),
        KScales: T.Tensor([seq_len_kv, 1], T.float32),
        Logits: T.Tensor([seq_len, seq_len_kv], T.float32),
        Weights: T.Tensor([seq_len, heads], T.float32),
        CuSeqLenKS: T.Tensor([seq_len], T.int32),
        CuSeqLenKE: T.Tensor([seq_len], T.int32),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            # E4M3 values multiplied by a power-of-two scale are exactly
            # representable in BF16.  Dequantize while loading shared memory,
            # matching vLLM's FP8 boundary without relying on an FP8xFP8 MMA
            # lowering that is not available on every TileLang/Hopper build.
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], T.bfloat16)
            index_k_shared = T.alloc_shared([block_N, index_dim], T.bfloat16)
            scores = T.alloc_fragment([block_N, block_Q * heads], T.float32)
            scores_3d = T.reshape(scores, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], T.float32)
            weights = T.alloc_fragment([block_Q, heads], T.float32)

            seq_len_i = bx * block_Q
            cu_k_s_min = T.alloc_var(T.int32)
            cu_k_e_max = T.alloc_var(T.int32)
            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            for bq_i, h_i, d_i in T.Parallel(block_Q, heads, index_dim):
                row = (seq_len_i + bq_i) * heads + h_i
                index_q_shared[bq_i * heads + h_i, d_i] = T.cast(IndexQ[row, d_i], T.float32) * QScales[row, 0]
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                k_start = cu_k_s_min + nbn_i * block_N
                for bn_i, d_i in T.Parallel(block_N, index_dim):
                    index_k_shared[bn_i, d_i] = (
                        T.cast(IndexK[k_start + bn_i, d_i], T.float32) * KScales[k_start + bn_i, 0]
                    )

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    scores_3d[bn_i, bq_i, h_i] = T.max(scores_3d[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]

                T.reduce_sum(scores_3d, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, k_start + bn_i] = logits[bn_i, bq_i]

    return main


def indexer_fp8_fwd_interface(
    q_values,
    k_values,
    q_scales,
    k_scales,
    weights,
    cu_seqlen_ks,
    cu_seqlen_ke,
):
    """Single-sample FP8 indexer forward."""
    seq_len, heads, index_dim = q_values.shape
    seq_len_kv = k_values.shape[0]
    block_Q = max(128 // heads, 1)
    padded_seq_len = (seq_len + block_Q - 1) // block_Q * block_Q
    if padded_seq_len != seq_len:
        q_values = torch.cat(
            [
                q_values,
                torch.zeros(
                    padded_seq_len - seq_len,
                    heads,
                    index_dim,
                    device=q_values.device,
                    dtype=q_values.dtype,
                ),
            ],
            dim=0,
        ).contiguous()
        q_scales = torch.cat(
            [
                q_scales,
                torch.ones(
                    padded_seq_len - seq_len,
                    heads,
                    1,
                    device=q_scales.device,
                    dtype=q_scales.dtype,
                ),
            ],
            dim=0,
        ).contiguous()
        weights = torch.cat(
            [
                weights,
                torch.zeros(
                    padded_seq_len - seq_len,
                    heads,
                    device=weights.device,
                    dtype=weights.dtype,
                ),
            ],
            dim=0,
        ).contiguous()
        cu_pad = torch.full(
            (padded_seq_len - seq_len,),
            seq_len_kv,
            device=cu_seqlen_ks.device,
            dtype=cu_seqlen_ks.dtype,
        )
        cu_seqlen_ks = torch.cat([cu_seqlen_ks, cu_pad], dim=0).contiguous()
        cu_seqlen_ke = torch.cat([cu_seqlen_ke, cu_pad], dim=0).contiguous()

    block_N = 256
    padded_seq_len_kv = (seq_len_kv + block_N - 1) // block_N * block_N
    if padded_seq_len_kv != seq_len_kv:
        k_values = torch.cat(
            [
                k_values,
                torch.zeros(
                    padded_seq_len_kv - seq_len_kv,
                    index_dim,
                    device=k_values.device,
                    dtype=k_values.dtype,
                ),
            ],
            dim=0,
        ).contiguous()
        k_scales = torch.cat(
            [
                k_scales,
                torch.ones(
                    padded_seq_len_kv - seq_len_kv,
                    1,
                    device=k_scales.device,
                    dtype=k_scales.dtype,
                ),
            ],
            dim=0,
        ).contiguous()

    logits = torch.empty((padded_seq_len, padded_seq_len_kv), dtype=torch.float32, device=q_values.device)
    tl_fp8_indexer_fwd_impl(heads=heads, index_dim=index_dim, block_Q=block_Q)(
        q_values.view(padded_seq_len * heads, index_dim),
        k_values,
        q_scales.view(padded_seq_len * heads, 1),
        k_scales,
        logits,
        weights.float(),
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    clean_logits_()(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits[:seq_len, :seq_len_kv]


def batched_indexer_fp8_fwd(q, k, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Run the true-FP8 indexer for every batch entry."""
    q_values, q_scales = q
    k_values, k_scales = k
    seqlen, batch = q_values.shape[:2]
    seq_len_kv = k_values.shape[0]
    output = torch.empty((batch, seqlen, seq_len_kv), dtype=torch.float32, device=q_values.device)
    for batch_idx in range(batch):
        output[batch_idx] = indexer_fp8_fwd_interface(
            q_values[:, batch_idx].contiguous(),
            k_values[:, batch_idx].contiguous(),
            q_scales[:, batch_idx].contiguous(),
            k_scales[:, batch_idx].contiguous(),
            weights[:, batch_idx].contiguous(),
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
    return output
