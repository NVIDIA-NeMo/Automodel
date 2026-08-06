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
# kernel in tilelang_sparse_mla_fwd.py (Apache-2.0, Zhipu AI).  The mixed
# FP8/BF16 KV loader and vLLM-compatible UE8M0 scaling are NVIDIA additions.

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


_FLASHMLA_SOFTMAX_SOURCE = r"""
__device__ __forceinline__ void dsv4_flashmla_softmax_64(
    float* scores,
    float scale_log2,
    const float* max_rows,
    float* sums) {
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    float cur_sum = 0.0f;
#pragma unroll
    for (int i = row * 2; i < 32; i += 4) {
      const float p0 = exp2f(scores[i] * scale_log2 - max_rows[row]);
      const float p1 = exp2f(scores[i + 1] * scale_log2 - max_rows[row]);
      scores[i] = p0;
      scores[i + 1] = p1;
      // FlashMLA deliberately rounds each pair before accumulating it.
      cur_sum += p0 + p1;
    }
    // This order is also intentional: FlashMLA reduces lanes 1 then 2,
    // whereas TileLang's generic butterfly reduction uses 2 then 1.
    cur_sum += __shfl_xor_sync(0xffffffff, cur_sum, 1);
    cur_sum += __shfl_xor_sync(0xffffffff, cur_sum, 2);
    sums[row] = cur_sum;
  }
}

__device__ __forceinline__ void dsv4_flashmla_softmax_local_update_64(
    float* scores,
    float scale_log2,
    const float* max_rows,
    const float* running_scales,
    float* local_sums) {
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    float cur_sum = 0.0f;
#pragma unroll
    for (int i = row * 2; i < 32; i += 4) {
      const float p0 = exp2f(scores[i] * scale_log2 - max_rows[row]);
      const float p1 = exp2f(scores[i + 1] * scale_log2 - max_rows[row]);
      scores[i] = p0;
      scores[i + 1] = p1;
      cur_sum += p0 + p1;
    }
    // Keep the expression identical to FlashMLA's per-warpgroup running L.
    local_sums[row] = local_sums[row] * running_scales[row] + cur_sum;
  }
}

__device__ __forceinline__ void dsv4_flashmla_scale_local_64(
    float* local_sums,
    const float* scales) {
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    local_sums[row] *= scales[row];
  }
}

__device__ __forceinline__ void dsv4_flashmla_pair_final_reduce_64(
    const float* local_sums0,
    const float* local_sums1,
    float* sums) {
#pragma unroll
  for (int row = 0; row < 2; ++row) {
    float sum0 = local_sums0[row];
    float sum1 = local_sums1[row];
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    sums[row] = sum0 + sum1;
  }
}

"""


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mqa_fp8_fwd(
    heads,
    topk,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=128,
):
    dim = 512
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504
    assert topk % block_I == 0

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    q_shape = [batch, seq_len, heads, dim]
    nope_shape = [batch, seq_len_kv, 448]
    rope_shape = [batch, seq_len_kv, 64]
    scale_shape = [batch, seq_len_kv, 8]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    attn_sink_shape = [heads]

    padded_H = max(tilelang.math.next_power_of_2(heads), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    # FlashMLA advances the sparse prefill pipeline in pairs of 64-token
    # blocks. The one-block case is a separate epilogue; all larger inputs are
    # padded by the Python interface to a whole number of pairs.
    assert NI == 1 or NI % 2 == 0
    if heads > 64:
        assert heads % 64 == 0
        replicate_H = heads // 64
    else:
        replicate_H = 1
    heads_per_block = padded_H if replicate_H == 1 else 64
    if NI > 1:
        assert heads_per_block == 64 and threads == 128

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, T.bfloat16),
        KVNope: T.Tensor(nope_shape, T.float8_e4m3fn),
        KVRope: T.Tensor(rope_shape, T.bfloat16),
        KVScales: T.Tensor(scale_shape, T.uint8),
        AttnSink: T.Tensor(attn_sink_shape, T.float32),
        Indices: T.Tensor(indices_shape, T.int32),
        ValidMask: T.Tensor(indices_shape, T.int32),
        Output: T.Tensor(o_shape, T.bfloat16),
        Lse: T.Tensor(lse_shape, T.float32),
    ):
        with T.Kernel(seq_len * replicate_H, batch, threads=threads) as (bx, by):
            T.import_source(_FLASHMLA_SOFTMAX_SOURCE)
            # FlashMLA's two consumer warpgroups each own half of the value
            # dimension.  Keeping the halves separate lets the first block use
            # RS for the left half and SS for the right half (and vice versa
            # for the second block), exactly as its paired-block pipeline does.
            q_left_shared = T.alloc_shared([heads_per_block, dim // 2], T.bfloat16)
            q_right_shared = T.alloc_shared([heads_per_block, dim // 2], T.bfloat16)
            kv_left_shared = T.alloc_shared([BI, dim // 2], T.bfloat16)
            kv_right_shared = T.alloc_shared([BI, dim // 2], T.bfloat16)
            # Keep these allocations unconditional. TileLang's fragment
            # storage analysis cannot lower buffers declared inside a
            # compile-time Python branch, even though NI is specialized.
            kv0_right_shared = T.alloc_shared([BI, dim // 2], T.bfloat16)
            p0_fp32_shared = T.alloc_shared([heads_per_block, BI], T.float32)
            mask = T.alloc_fragment([BI], "bool")

            acc_o_left = T.alloc_fragment([heads_per_block, dim // 2], T.float32)
            acc_o_right = T.alloc_fragment([heads_per_block, dim // 2], T.float32)
            acc_s = T.alloc_fragment([heads_per_block, BI], T.float32)
            # FlashMLA keeps the BF16 softmax probabilities in registers and
            # feeds them to the S@V WGMMA through its register-source (RS)
            # operand.  Staging them through shared memory selects the SS
            # instruction instead, whose accumulator rounding differs at a
            # handful of BF16 output boundaries.
            p_fragment = T.alloc_fragment([heads_per_block, BI], T.bfloat16)
            p_shared = T.alloc_shared([heads_per_block, BI], T.bfloat16)
            # FlashMLA keeps separate, unreduced per-lane denominators for its
            # two consumer warpgroups and reduces them together only once at
            # the end. Plain local buffers intentionally give every CUDA
            # thread two row slots without TileLang fragment-layout inference.
            local_sum0 = T.alloc_local([2], T.float32)
            local_sum1 = T.alloc_local([2], T.float32)
            sumexp = T.alloc_fragment([heads_per_block], T.float32)
            sumexp_i = T.alloc_fragment([heads_per_block], T.float32)
            scale_factor = T.alloc_fragment([heads_per_block], T.float32)
            alpha = T.alloc_fragment([heads_per_block], T.float32)
            m_i = T.alloc_fragment([heads_per_block], T.float32)
            m_i_prev = T.alloc_fragment([heads_per_block], T.float32)
            m_i_block = T.alloc_fragment([heads_per_block], T.float32)

            T.fill(acc_o_left, 0)
            T.fill(acc_o_right, 0)
            T.fill(sumexp, 0)
            # The exact two-block path otherwise touches this fragment only
            # through imported CUDA helpers. Give TileLang an explicit access
            # from which to infer its per-row fragment layout.
            for head in T.Parallel(heads_per_block):
                sumexp_i[head] = 0
            local_sum0[0] = 0
            local_sum0[1] = 0
            local_sum1[0] = 0
            local_sum1[1] = 0
            # FlashMLA tracks the running maximum in the scaled log2 domain.
            # Keeping an unscaled maximum and multiplying it later is
            # mathematically equivalent but changes FP32 rounding at BF16
            # output boundaries.
            T.fill(m_i, -1.0e30)

            batch_idx = by
            seq_idx = bx if replicate_H == 1 else bx // replicate_H
            head_start = 0 if replicate_H == 1 else (bx % replicate_H) * 64
            head_end = head_start + heads_per_block
            T.copy(Q[batch_idx, seq_idx, head_start:head_end, : dim // 2], q_left_shared)
            T.copy(Q[batch_idx, seq_idx, head_start:head_end, dim // 2 :], q_right_shared)

            for topk_block in T.serial(NI):
                for row in T.Parallel(BI):
                    mask[row] = ValidMask[batch_idx, seq_idx, topk_block * BI + row] != 0

                for row, d in T.Parallel(BI, dim // 2):
                    kv_idx = Indices[batch_idx, seq_idx, topk_block * BI + row]
                    encoded_scale = KVScales[batch_idx, kv_idx, d // 64]
                    descale = T.reinterpret(T.cast(encoded_scale, T.uint32) << 23, T.float32)
                    kv_left_shared[row, d] = T.cast(KVNope[batch_idx, kv_idx, d], T.float32) * descale
                for row, d in T.Parallel(BI, dim // 2):
                    kv_idx = Indices[batch_idx, seq_idx, topk_block * BI + row]
                    if d < 192:
                        encoded_scale = KVScales[batch_idx, kv_idx, (d + dim // 2) // 64]
                        descale = T.reinterpret(T.cast(encoded_scale, T.uint32) << 23, T.float32)
                        kv_right_shared[row, d] = T.cast(KVNope[batch_idx, kv_idx, d + dim // 2], T.float32) * descale
                    else:
                        kv_right_shared[row, d] = KVRope[batch_idx, kv_idx, d - 192]
                if NI > 1:
                    if topk_block % 2 == 0:
                        T.copy(kv_right_shared, kv0_right_shared)

                T.clear(acc_s)
                if NI == 1 or topk_block % 2 == 0:
                    # FlashMLA WG0 accumulates QK tiles from low to high D.
                    T.gemm(
                        q_left_shared,
                        kv_left_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        q_right_shared,
                        kv_right_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                else:
                    # WG1 deliberately issues tiles 4..7 before 0..3.  The
                    # reverse FP32 WGMMA accumulation order is observable at
                    # softmax/output rounding boundaries.
                    T.gemm(
                        q_right_shared,
                        kv_right_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        q_left_shared,
                        kv_left_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                for head, row in T.Parallel(heads_per_block, BI):
                    acc_s[head, row] = T.if_then_else(mask[row], acc_s[head, row], -T.infinity(acc_s.dtype))
                T.copy(m_i, m_i_prev)
                if NI > 1:
                    if topk_block % 2 == 0:
                        # Preserve the global max at the start of the pair.
                        # WG1 scales its running O/L once from this value to
                        # the final max after the odd block.
                        for head in T.Parallel(heads_per_block):
                            scale_factor[head] = m_i_prev[head]
                T.fill(m_i_block, -T.infinity(T.float32))
                T.reduce_max(acc_s, m_i_block, dim=1, clear=False)
                for head in T.Parallel(heads_per_block):
                    m_i[head] = T.max(m_i_block[head] * sm_scale, m_i_prev[head])
                    alpha[head] = T.exp2(m_i_prev[head] - m_i[head])
                if NI > 1:
                    if topk_block % 2 == 0:
                        T.call_extern(
                            "handle",
                            "dsv4_flashmla_softmax_local_update_64",
                            acc_s.data,
                            sm_scale,
                            m_i.data,
                            alpha.data,
                            local_sum0.data,
                        )
                    else:
                        for head in T.Parallel(heads_per_block):
                            scale_factor[head] = T.exp2(scale_factor[head] - m_i[head])
                        T.call_extern(
                            "handle",
                            "dsv4_flashmla_softmax_local_update_64",
                            acc_s.data,
                            sm_scale,
                            m_i.data,
                            scale_factor.data,
                            local_sum1.data,
                        )
                        T.call_extern(
                            "handle",
                            "dsv4_flashmla_scale_local_64",
                            local_sum0.data,
                            alpha.data,
                        )
                elif heads_per_block == 64 and threads == 128:
                    # For a 64x64 WGMMA tile each thread owns two query rows
                    # and 16 scores per row.  Use FlashMLA's exact pairwise
                    # accumulation and lane-shuffle order; the generic
                    # TileLang reduction is numerically equivalent but not
                    # bitwise equal.
                    T.call_extern(
                        "handle",
                        "dsv4_flashmla_softmax_64",
                        acc_s.data,
                        sm_scale,
                        m_i.data,
                        sumexp_i.data,
                    )
                else:
                    for head, row in T.Parallel(heads_per_block, BI):
                        acc_s[head, row] = T.exp2(acc_s[head, row] * sm_scale - m_i[head])
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                if NI == 1:
                    for head in T.Parallel(heads_per_block):
                        sumexp[head] = sumexp[head] * alpha[head] + sumexp_i[head]
                    for head, d in T.Parallel(heads_per_block, dim // 2):
                        acc_o_left[head, d] = acc_o_left[head, d] * alpha[head]
                        acc_o_right[head, d] = acc_o_right[head, d] * alpha[head]
                elif topk_block % 2 == 0:
                    # WG0 advances through the even-block max first.
                    for head, d in T.Parallel(heads_per_block, dim // 2):
                        acc_o_left[head, d] = acc_o_left[head, d] * alpha[head]
                else:
                    # WG0 applies the second max transition separately, while
                    # WG1 scales once from the previous pair's global max.
                    for head, d in T.Parallel(heads_per_block, dim // 2):
                        acc_o_left[head, d] = acc_o_left[head, d] * alpha[head]
                        acc_o_right[head, d] = acc_o_right[head, d] * scale_factor[head]
                # The score accumulator and PV register operand have different
                # WGMMA fragment layouts.  A shared-memory round trip performs
                # that layout conversion while the actual PV GEMM remains RS.
                if NI == 1:
                    T.copy(acc_s, p_shared)
                    T.copy(p_shared, p_fragment)
                    T.gemm(
                        p_fragment,
                        kv_left_shared,
                        acc_o_left,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        p_shared,
                        kv_right_shared,
                        acc_o_right,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                else:
                    if topk_block % 2 == 0:
                        # WG0 owns the even-block probabilities and left value
                        # half.  Preserve the FP32 probabilities so WG1's
                        # global max can rescale them before the remote SS PV.
                        T.copy(acc_s, p0_fp32_shared)
                        T.copy(acc_s, p_shared)
                        T.copy(p_shared, p_fragment)
                        T.gemm(
                            p_fragment,
                            kv_left_shared,
                            acc_o_left,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    else:
                        # WG1's odd-block probabilities are local (RS) for the
                        # right half and remote (SS) for the left half.
                        T.copy(acc_s, p_shared)
                        T.copy(p_shared, p_fragment)
                        T.gemm(
                            p_shared,
                            kv_left_shared,
                            acc_o_left,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            p_fragment,
                            kv_right_shared,
                            acc_o_right,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        # FlashMLA rescales the even block in FP32 and rounds it to
                        # BF16 again before WG1 consumes it through SS.
                        T.copy(p0_fp32_shared, acc_s)
                        for head, row in T.Parallel(heads_per_block, BI):
                            acc_s[head, row] *= alpha[head]
                        T.copy(acc_s, p_shared)
                        T.gemm(
                            p_shared,
                            kv0_right_shared,
                            acc_o_right,
                            policy=T.GemmWarpPolicy.FullRow,
                        )

            if NI > 1:
                T.call_extern(
                    "handle",
                    "dsv4_flashmla_pair_final_reduce_64",
                    local_sum0.data,
                    local_sum1.data,
                    sumexp.data,
                )

            for head in T.Parallel(heads_per_block):
                sumexp[head] += T.exp2(AttnSink[head_start + head] * 1.44269504 - m_i[head])
                # Match FlashMLA's epilogue exactly: calculate one reciprocal
                # per row, then multiply every numerator element by it.
                # Per-element division is mathematically equivalent but flips
                # a handful of values at BF16 round-to-nearest boundaries.
                scale_factor[head] = 1.0 / sumexp[head]
            for head, d in T.Parallel(heads_per_block, dim // 2):
                acc_o_left[head, d] *= scale_factor[head]
                acc_o_right[head, d] *= scale_factor[head]
            for head in T.Parallel(heads_per_block):
                sumexp[head] = T.log2(sumexp[head]) + m_i[head]

            T.copy(acc_o_left, Output[batch_idx, seq_idx, head_start:head_end, : dim // 2])
            T.copy(acc_o_right, Output[batch_idx, seq_idx, head_start:head_end, dim // 2 :])
            T.copy(sumexp, Lse[batch_idx, seq_idx, head_start:head_end])

    return main


def sparse_mqa_fp8_fwd_interface(
    q,
    kv_nope,
    kv_rope,
    kv_scales,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=128,
):
    assert q.is_contiguous() and kv_nope.is_contiguous()
    assert kv_rope.is_contiguous() and kv_scales.is_contiguous()
    assert q.dtype == torch.bfloat16
    assert kv_nope.dtype == torch.float8_e4m3fn
    assert kv_rope.dtype == torch.bfloat16 and kv_scales.dtype == torch.uint8
    batch, seq_len, heads, dim = q.shape
    actual_heads = heads
    assert dim == 512
    _, seq_len_kv, nope_dim = kv_nope.shape
    assert nope_dim == 448 and kv_rope.shape == (batch, seq_len_kv, 64)
    assert kv_scales.shape == (batch, seq_len_kv, 8)
    topk = topk_idxs.shape[-1]

    if heads < 64:
        q_padded = q.new_zeros((batch, seq_len, 64, dim))
        q_padded[:, :, :heads].copy_(q)
        sink_padded = torch.full(
            (64,),
            -torch.inf,
            dtype=attn_sink.dtype,
            device=attn_sink.device,
        )
        sink_padded[:heads].copy_(attn_sink)
        q = q_padded
        attn_sink = sink_padded
        heads = 64

    if topk <= block_I:
        padded_topk = block_I
    else:
        # FlashMLA's sparse-prefill consumer warpgroups advance in pairs.
        # Keep the final odd block as an all-invalid peer instead of selecting
        # a numerically different one-block epilogue for the tail.
        pair_size = 2 * block_I
        padded_topk = (topk + pair_size - 1) // pair_size * pair_size
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
    kernel = sparse_mqa_fp8_fwd(
        heads,
        topk,
        sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    output, lse = kernel(q, kv_nope, kv_rope, kv_scales, attn_sink, topk_idxs, valid_mask)
    if actual_heads != heads:
        output = output[:, :, :actual_heads].contiguous()
        lse = lse[:, :, :actual_heads].contiguous()
    return output, lse
