"""Custom ops for MHA/XQA attention."""

import math
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, field, fields
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Type, Union

import torch
import torch.nn.functional as F
import triton
from torch.export import Dim
from triton import language as tl


@triton.jit
def update_kv_cache(
    k_ptr,  # [B*S, N, D]
    v_ptr,  # [B*S, N, D]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    input_pos_ptr,  # Specifies the sequence index in the caches at which to write the provided kv
    cache_loc_ptr,  # Specifies the batch index for each of the input sequences
    MAX_SEQ_LENGTH: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    Q_D_HEAD: tl.constexpr,
    V_D_HEAD: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    GENERATE_ONLY: tl.constexpr,
):
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    if GENERATE_ONLY:
        seq_start_index = batch_id
        seq_len: tl.constexpr = 1
    else:
        seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
        seq_len = tl.load(seq_len_ptr + batch_id)

    # cache is [bsnd]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)

    kv_position = tl.load(input_pos_ptr + batch_id)

    K_D_HEAD: tl.constexpr = Q_D_HEAD
    k_cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH * K_D_HEAD
    v_cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH * V_D_HEAD

    k_dhead_offsets = tl.arange(0, triton.next_power_of_2(K_D_HEAD))
    k_dhead_mask = k_dhead_offsets < K_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    k_load_mask = seq_mask[:, None] * k_dhead_mask[None, :]
    v_load_mask = seq_mask[:, None] * v_dhead_mask[None, :]

    k_batch_offset = seq_start_index * N_KV_HEADS * K_D_HEAD
    v_batch_offset = seq_start_index * N_KV_HEADS * V_D_HEAD
    # Write back to kv-caches
    ks = tl.load(
        k_ptr
        + k_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * K_D_HEAD
        + head_id * K_D_HEAD
        + k_dhead_offsets[None, :],
        mask=k_load_mask,
    )
    vs = tl.load(
        v_ptr
        + v_batch_offset
        + seq_offsets[:, None] * N_KV_HEADS * V_D_HEAD
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        mask=v_load_mask,
    )

    kv_writeback_seq_offsets = seq_offsets + kv_position

    k_cache_offset = (
        k_cache_batch_offset
        + kv_writeback_seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
        + head_id * K_D_HEAD
        + k_dhead_offsets[None, :]
    )

    v_cache_offset = (
        v_cache_batch_offset
        + kv_writeback_seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :]
    )
    tl.store(k_cache_ptr + k_cache_offset, ks, k_load_mask)
    tl.store(v_cache_ptr + v_cache_offset, vs, v_load_mask)


@triton.jit
def gqa_attention_kv_stage1(
    q_ptr,  # [Batch, 1, N_HEADS, D_HEAD]
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    cache_loc_ptr,  # [Batch] # Specifies the batch index for each of the generate tokens.
    input_pos_ptr,  # [Batch]
    output_values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    output_logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    num_blocks,
    MAX_SEQ_LEN: tl.constexpr,  # Maximum supported sequence length
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    Q_D_HEAD: tl.constexpr,  # Dimension of each query head.
    V_D_HEAD: tl.constexpr,  # Dimension of each key/value head
    SEQ_BLOCK_SIZE: tl.constexpr,  # Block size used for tiling the sequence dim.
    HEAD_BLOCK_SIZE: tl.constexpr,  # pad to 16 if HEAD_RATIO is < 16 to invoke tensor cores.
):
    """Attention kernel to be used for generate-only batches.

    Specialized for GQA.

    Assumes that kv caches have been updated.

    Supports non-power-of-2 D_HEAD

    Uses flash decoding.
    KV-cache layout is assumed to be [Batch,Seq, Head, Dim]
    1. Fetch the K-cache from 0 to input_pos
    2. Fetch the V-cache from 0 to input_pos
    3. A = Q*K^T [1,D_HEAD] * [1,seq_len,D_HEAD] -> [1, seq_len]
    4. S = softmax(A)
    5. O = S*V [1, seq_len] * [1, seq_len, D_HEAD] -> [1, D_HEAD]
    """
    # Assume KV-cache layout: [Batch, Seq, Head, Dim]
    # A program is responsible for 1 batch, 1 head and a block of sequences.
    batch_id = tl.program_id(axis=0)
    kv_head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    kv_position = tl.load(input_pos_ptr + batch_id)
    kv_batch_id = tl.load(cache_loc_ptr + batch_id)
    K_D_HEAD: tl.constexpr = Q_D_HEAD
    batch_offset = kv_batch_id * N_KV_HEADS * MAX_SEQ_LEN

    # Offsets for the block of sequences this program processes.
    seq_start_pos = seq_block_id * SEQ_BLOCK_SIZE

    # The number of Q heads that map to each KV head.
    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS  # This needs to be a power-of-2
    if seq_start_pos > kv_position:
        return
    seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
    seq_mask = seq_offsets <= kv_position

    # Need to pad the head dim to 16 if HEAD_RATIO is < 16 so that tensor cores can be invoked
    #
    head_offsets = kv_head_id * HEAD_RATIO + tl.arange(0, HEAD_BLOCK_SIZE)
    head_mask = head_offsets < (kv_head_id * HEAD_RATIO + HEAD_RATIO)
    # Assuming D_HEAD is a power of 2
    q_dhead_offsets = tl.arange(0, triton.next_power_of_2(Q_D_HEAD))
    q_dhead_mask = q_dhead_offsets < Q_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    sm_scale: tl.constexpr = 1.0 / (Q_D_HEAD**0.5)

    # Program loads the entire Q for the head assigned to it.
    # [NUM_HEADS, Q_D_HEAD]
    q_batch_offset = batch_id * N_HEADS * Q_D_HEAD
    q_head_offsets = head_offsets * Q_D_HEAD

    # Q layout : BSND
    q = tl.load(
        q_ptr + q_batch_offset + q_head_offsets[:, None] + q_dhead_offsets[None, :],
        mask=head_mask[:, None] * q_dhead_mask[None, :],
        other=0.0,
    )

    # [BSND]
    k_block_offsets = (
        batch_offset * K_D_HEAD
        + seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
        + kv_head_id * K_D_HEAD
        + q_dhead_offsets[None, :]
    )
    k_mask = seq_mask[:, None] * q_dhead_mask[None, :]  # K and Q share the same head dim
    k = tl.load(k_cache_ptr + k_block_offsets, mask=k_mask, other=0.0)

    v_block_offsets = (
        batch_offset * V_D_HEAD
        + seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
        + kv_head_id * V_D_HEAD
        + v_dhead_offsets[None, :]
    )
    v_mask = seq_mask[:, None] * v_dhead_mask[None, :]

    # [seq_block, V_D_HEAD]
    v = tl.load(v_cache_ptr + v_block_offsets, mask=v_mask, other=0.0)

    # Note: check the output precision of the sum.
    # compute q*K^T
    # [NUM_HEADS, Q_D_HEAD] * [seq_block, Q_D_HEAD], sum along axis 1
    attn = tl.dot(q, k.trans())  # [N, seq_block]
    attn = attn.to(tl.float32)
    attn *= sm_scale
    max_attn = tl.max(attn, axis=1)  # [N, 1]
    # Set to -inf attn values where mask is not set. This forces exp(attn) to 0.
    attn = tl.where(head_mask[:, None] * seq_mask[None, :], attn, float("-inf"))
    exp_attn = tl.exp(attn - max_attn[:, None])

    sumexp = tl.sum(exp_attn, axis=1)  # [N, 1]

    # [NUM_HEADS, seq_len] * [seq_len, V_D_HEAD], sum along axis 0
    output = tl.dot(exp_attn.to(v.dtype), v)

    output = output / sumexp[:, None]  # [N, D_HEAD]

    # We store the log-sum-exp after removing the max.
    logsumexp = tl.log(sumexp) + max_attn
    # when seq_mask is all false, max_attn will be -inf and sumexp is zero

    tl.store(
        output_values_ptr
        + batch_id * N_HEADS * V_D_HEAD * num_blocks
        + head_offsets[:, None] * V_D_HEAD * num_blocks
        + seq_block_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        output,
        mask=head_mask[:, None] * v_dhead_mask[None, :],
    )
    tl.store(
        output_logsumexp_ptr + batch_id * N_HEADS * num_blocks + head_offsets * num_blocks + seq_block_id,
        logsumexp,
        mask=head_mask,
    )


@triton.jit
def attention_kv_stage1(
    q_ptr,  # [Batch, 1, N_HEADS, D_HEAD]
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    cache_loc_ptr,  # [Batch] # Specifies the batch index for each of the generate tokens.
    input_pos_ptr,  # [Batch]
    output_values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    output_logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    num_blocks,
    MAX_SEQ_LEN: tl.constexpr,  # Maximum supported sequence length
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    SEQ_BLOCK_SIZE: tl.constexpr,  # Block size used for tiling the sequence dim.
):
    """Attention kernel to be used for generate-only batches.

    Assumes that kv caches have been updated.

    Uses flash decoding.
    KV-cache layout is assumed to be [Batch,Seq, Head, Dim]
    1. Fetch the K-cache from 0 to input_pos
    2. Fetch the V-cache from 0 to input_pos
    3. A = Q*K^T [1,D_HEAD] * [1,seq_len,D_HEAD] -> [1, seq_len]
    4. S = softmax(A)
    5. O = S*V [1, seq_len] * [1, seq_len, D_HEAD] -> [1, D_HEAD]
    """
    # Assume KV-cache layout: [Batch, Seq, Head, Dim]
    # A program is responsible for 1 batch, 1 head and a block of sequences.
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)
    epsilon: tl.constexpr = 1e-38  # float32 smallest positive number

    kv_position = tl.load(input_pos_ptr + batch_id)
    kv_batch_id = tl.load(cache_loc_ptr + batch_id)
    kv_batch_offset = kv_batch_id * N_KV_HEADS * MAX_SEQ_LEN * D_HEAD
    # Offsets for the block of sequences this program processes.
    seq_start_pos = seq_block_id * SEQ_BLOCK_SIZE

    if seq_start_pos > kv_position:
        return
    seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
    seq_mask = seq_offsets <= kv_position
    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, triton.next_power_of_2(D_HEAD))
    dhead_mask = dhead_offsets < D_HEAD

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    kv_head_offset = (head_id // HEAD_RATIO) * D_HEAD

    sm_scale: tl.constexpr = 1.0 / (D_HEAD**0.5)

    # Program loads the entire Q for the head assigned to it.
    # [D_HEAD]
    q_batch_offset = batch_id * N_HEADS * D_HEAD
    q_head_offset = head_id * D_HEAD
    q = tl.load(q_ptr + q_batch_offset + q_head_offset + dhead_offsets, mask=dhead_mask)

    kv_block_offsets = (
        kv_batch_offset + seq_offsets[:, None] * D_HEAD * N_KV_HEADS + kv_head_offset + dhead_offsets[None, :]
    )  # [BSND]
    kv_mask = seq_mask[:, None] * dhead_mask[None, :]

    # [seq_block, D_HEAD]
    k = tl.load(k_cache_ptr + kv_block_offsets, mask=kv_mask, other=0.0)
    v = tl.load(v_cache_ptr + kv_block_offsets, mask=kv_mask, other=0.0)

    # Note: check the output precision of the sum.
    # compute q*K^T
    # [D_HEAD] * [seq_block, D_HEAD], sum along axis 1
    attn = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)  # [seq_block]

    attn *= sm_scale
    max_attn = tl.max(attn)
    # Set to -inf attn values where mask is not set. This forces exp(attn) to 0.
    attn = tl.where(seq_mask, attn, float("-inf"))
    exp_attn = tl.exp(attn - max_attn)
    exp_attn = tl.where(exp_attn == 0, epsilon, exp_attn)
    sumexp = tl.sum(exp_attn, axis=0)  # scalar.

    # [seq_len] * [seq_len, D_HEAD], sum along axis 0
    output = tl.sum(exp_attn[:, None] * v, axis=0)  # [D_HEAD]

    output = output / sumexp

    # We store the log-sum-exp after removing the max.
    logsumexp = tl.log(sumexp) + max_attn
    # when seq_mask is all false, max_attn will be -inf and sumexp is zero

    tl.store(
        output_values_ptr
        + batch_id * N_HEADS * D_HEAD * num_blocks
        + head_id * D_HEAD * num_blocks
        + seq_block_id * D_HEAD
        + dhead_offsets,
        output,
        mask=dhead_mask,
    )
    tl.store(
        output_logsumexp_ptr + batch_id * N_HEADS * num_blocks + head_id * num_blocks + seq_block_id,
        logsumexp,
    )


@triton.jit
def attention_kv_stage2(
    values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    output_ptr,  # [Batch, N_HEADS, D_HEAD]
    input_pos_ptr,
    NUM_BLOCKS: tl.constexpr,
    N_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    SEQ_BLOCK_SIZE: tl.constexpr,  # Nearest power of 2 for num_blocks
):
    # There are batch * N_HEADS programs
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    dhead_offsets = tl.arange(0, triton.next_power_of_2(D_HEAD))
    dhead_mask = dhead_offsets < D_HEAD

    kv_position = tl.load(input_pos_ptr + batch_id)
    block_id = kv_position // SEQ_BLOCK_SIZE + 1

    NUM_BLOCKS_POW2: tl.constexpr = triton.next_power_of_2(NUM_BLOCKS)
    block_offsets = tl.arange(0, NUM_BLOCKS_POW2)

    block_mask = block_offsets < block_id
    logsumexp = tl.load(
        logsumexp_ptr + batch_id * N_HEADS * NUM_BLOCKS + head_id * NUM_BLOCKS + block_offsets,
        mask=block_mask,
        other=float("-inf"),
    )
    max_logsumexp = tl.max(logsumexp)
    sumexp = tl.exp(logsumexp - max_logsumexp)  # [NUM_BLOCKS_POW2]

    aggregate_sumexp = tl.sum(sumexp, axis=0)

    values_offsets = block_offsets[:, None] * D_HEAD + dhead_offsets[None, :]
    values_mask = block_mask[:, None] * dhead_mask[None, :]

    values = tl.load(
        values_ptr + batch_id * N_HEADS * D_HEAD * NUM_BLOCKS + head_id * D_HEAD * NUM_BLOCKS + values_offsets,
        mask=values_mask,
        other=0.0,
    )  # [BLOCK_SIZE, D_HEAD]
    values *= sumexp[:, None]
    values /= aggregate_sumexp

    output = tl.sum(values, axis=0)  # [DHEAD]

    tl.store(
        output_ptr + batch_id * N_HEADS * D_HEAD + head_id * D_HEAD + dhead_offsets,
        output,
        mask=dhead_mask,
    )


@triton.jit
def context_attention_kv(
    q_ptr,  # [bsnd]
    k_ptr,  # [bsnd]
    v_ptr,  # [bsnd]
    k_cache_ptr,  # [bsnd]
    v_cache_ptr,  # [bsnd]
    seq_len,
    o_ptr,
    softmax_scale,
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    Q_D_HEAD: tl.constexpr,  # Dimension of each query head.
    V_D_HEAD: tl.constexpr,  # Dimension of each value head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
):
    """Kernel for context phase.

    Assuming:
    1. Self-attention [seqlen(Q) == seqlen(K)]
    2. Causal attention
    3. QKV layout: [bsnd]
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    K_D_HEAD: tl.constexpr = Q_D_HEAD

    q_dhead_offsets = tl.arange(0, triton.next_power_of_2(Q_D_HEAD))
    q_dhead_mask = q_dhead_offsets < Q_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    q_load_mask = seq_mask[:, None] * q_dhead_mask[None, :]

    q_batch_offset = batch_id * seq_len * N_HEADS
    kv_batch_offset = batch_id * seq_len * N_KV_HEADS

    k_head_offset = (head_id // HEAD_RATIO) * K_D_HEAD
    v_head_offset = (head_id // HEAD_RATIO) * V_D_HEAD

    # Q will stay in SRAM
    q = tl.load(
        q_ptr
        + q_batch_offset * Q_D_HEAD
        + seq_offsets[:, None] * N_HEADS * Q_D_HEAD
        + head_id * Q_D_HEAD
        + q_dhead_offsets[None, :],
        mask=q_load_mask,
    )
    acc = tl.zeros([SEQ_BLOCK, triton.next_power_of_2(V_D_HEAD)], dtype=tl.float32)
    lse_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")

    for s in range(0, seq_block_id + 1, 1):
        kv_seq_offsets = s * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
        kv_seq_mask = kv_seq_offsets < seq_len
        k_load_mask = kv_seq_mask[:, None] * q_dhead_mask[None, :]

        k = tl.load(
            k_ptr
            + kv_batch_offset * K_D_HEAD
            + kv_seq_offsets[:, None] * N_KV_HEADS * K_D_HEAD
            + k_head_offset
            + q_dhead_offsets[None, :],
            mask=k_load_mask,
        )
        qk = tl.zeros([SEQ_BLOCK, SEQ_BLOCK], dtype=tl.float32)
        qk += tl.dot(q, k.trans())
        # causal mask
        qk = tl.where(seq_offsets[:, None] >= kv_seq_offsets[None, :], qk, float("-inf"))
        qk *= softmax_scale
        # rowmax
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])  # [S,S]
        v = tl.load(
            v_ptr
            + kv_batch_offset * V_D_HEAD
            + kv_seq_offsets[:, None] * N_KV_HEADS * V_D_HEAD
            + v_head_offset
            + v_dhead_offsets[None, :],
            mask=kv_seq_mask[:, None] * v_dhead_mask[None, :],
        )

        l_ij = tl.sum(p, 1)
        acc_scale = tl.exp(m_i - m_ij)
        acc = acc * acc_scale[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)

    acc = acc * o_scale[:, None]

    tl.store(
        o_ptr
        + batch_id * seq_len * N_HEADS * V_D_HEAD
        + seq_offsets[:, None] * N_HEADS * V_D_HEAD
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        acc,
        mask=seq_mask[:, None] * v_dhead_mask[None, :],
    )

    # Write back to kv-caches

    ks = tl.load(
        k_ptr
        + kv_batch_offset * K_D_HEAD
        + seq_offsets[:, None] * N_KV_HEADS * K_D_HEAD
        + k_head_offset
        + q_dhead_offsets[None, :],
        mask=seq_mask[:, None] * q_dhead_mask[None, :],
    )
    vs = tl.load(
        v_ptr
        + kv_batch_offset * V_D_HEAD
        + seq_offsets[:, None] * N_KV_HEADS * V_D_HEAD
        + v_head_offset
        + v_dhead_offsets[None, :],
        mask=seq_mask[:, None] * v_dhead_mask[None, :],
    )
    # cache is [bsnd]
    k_cache_offset = (
        batch_id * N_KV_HEADS * MAX_SEQ_LENGTH * K_D_HEAD
        + seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
        + k_head_offset
        + q_dhead_offsets[None, :]
    )

    v_cache_offset = (
        batch_id * N_KV_HEADS * MAX_SEQ_LENGTH * V_D_HEAD
        + seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
        + v_head_offset
        + v_dhead_offsets[None, :]
    )
    tl.store(k_cache_ptr + k_cache_offset, ks, seq_mask[:, None] * q_dhead_mask[None, :])
    tl.store(v_cache_ptr + v_cache_offset, vs, seq_mask[:, None] * v_dhead_mask[None, :])


@triton.jit
def context_attention_kv_flattened(
    q_ptr,  # [b*s,nd]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [bsnd]
    v_cache_ptr,  # [bsnd]
    input_pos_ptr,  # [b] # specifies the location in the sequence where kv must be written back.
    cache_loc_ptr,  # [b] # location of the sequence in the cache.
    o_ptr,
    softmax_scale: tl.constexpr,
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    Q_D_HEAD: tl.constexpr,  # Dimension of each query head.
    V_D_HEAD: tl.constexpr,  # Dimension of each value head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
):
    """Kernel for context phase.

    Assumes that kv caches have been updated.
    Assuming QKV layout: [b*s,n,d]
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
    seq_len = tl.load(seq_len_ptr + batch_id)
    K_D_HEAD: tl.constexpr = Q_D_HEAD
    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS

    # cache is [bsnd]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)

    cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH
    cache_head_offset = head_id // HEAD_RATIO

    q_dhead_offsets = tl.arange(0, triton.next_power_of_2(Q_D_HEAD))
    q_dhead_mask = q_dhead_offsets < Q_D_HEAD

    v_dhead_offsets = tl.arange(0, triton.next_power_of_2(V_D_HEAD))
    v_dhead_mask = v_dhead_offsets < V_D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    # Q will stay in SRAM
    q = tl.load(
        q_ptr
        + seq_start_index * N_HEADS * Q_D_HEAD
        + seq_offsets[:, None] * N_HEADS * Q_D_HEAD
        + head_id * Q_D_HEAD
        + q_dhead_offsets[None, :],
        mask=seq_mask[:, None] * q_dhead_mask[None, :],
    )

    acc = tl.zeros([SEQ_BLOCK, triton.next_power_of_2(V_D_HEAD)], dtype=tl.float32)
    lse_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")

    # Loop over the entire KV-history
    # input_pos_ptr stores the location at which kv must be written back for the given batch.
    kv_position = tl.load(input_pos_ptr + batch_id)
    num_blocks = (kv_position + seq_len + SEQ_BLOCK - 1) // SEQ_BLOCK
    for s in range(0, num_blocks + 1, 1):
        kv_seq_offsets = s * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
        kv_seq_mask = kv_seq_offsets < (kv_position + seq_len)

        k = tl.load(
            k_cache_ptr
            + cache_batch_offset * K_D_HEAD
            + kv_seq_offsets[:, None] * K_D_HEAD * N_KV_HEADS
            + cache_head_offset * K_D_HEAD
            + q_dhead_offsets[None, :],
            mask=kv_seq_mask[:, None] * q_dhead_mask[None, :],
        )
        qk = tl.zeros([SEQ_BLOCK, SEQ_BLOCK], dtype=tl.float32)
        qk += tl.dot(q, k.trans())
        qk = tl.where((seq_offsets[:, None] + kv_position) >= kv_seq_offsets[None, :], qk, float("-inf"))
        qk *= softmax_scale
        # rowmax
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        v = tl.load(
            v_cache_ptr
            + cache_batch_offset * V_D_HEAD
            + kv_seq_offsets[:, None] * V_D_HEAD * N_KV_HEADS
            + cache_head_offset * V_D_HEAD
            + v_dhead_offsets[None, :],
            mask=kv_seq_mask[:, None] * v_dhead_mask[None, :],
        )

        l_ij = tl.sum(p, 1)
        acc_scale = tl.exp(m_i - m_ij)
        acc = acc * acc_scale[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)

    acc = acc * o_scale[:, None]

    tl.store(
        o_ptr
        + seq_start_index * N_HEADS * V_D_HEAD
        + seq_offsets[:, None] * N_HEADS * V_D_HEAD
        + head_id * V_D_HEAD
        + v_dhead_offsets[None, :],
        acc,
        mask=seq_mask[:, None] * v_dhead_mask[None, :],
    )


@triton.jit
def update_kv_cache_rope_fusion(
    q_ptr,  # [B*S, N, D]
    k_ptr,  # [B*S, N, D]
    v_ptr,  # [B*S, N, D]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    q_rope_ptr,  # [B*S, N, D], roped q result
    k_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    v_cache_ptr,  # [MAX_BATCH_SIZE, MAX_SEQ_LEN, N_HEADS, D_HEAD]
    input_pos_ptr,  # Specifies the sequence index in the caches at which to write the provided kv
    cache_loc_ptr,  # Specifies the batch index for each of the input sequences
    f_ptr,  # [MAX_SEQ_LEN, D_HEAD//2, 2] # frequencies for rope embadding.
    MAX_SEQ_LENGTH: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    HEAD_BLOCK_SIZE: tl.constexpr,  # pad to 16 if HEAD_RATIO is < 16 to invoke tensor cores.
    GENERATE_ONLY: tl.constexpr,
):
    """Fuse q and k rope with update_kv_cache kernel.

    The input is interleaved as [2, D//2] in D_HEAD dim.
    Update q_rope with the post-rope-embadding q values.
    Update k_cache with the post-rope-embadding k values.
    For rope computation, q and k need to load and store in tensors pair of 2 * [D//2].
    Update v_cache with v.
    """
    batch_id = tl.program_id(axis=0)
    kv_head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    if GENERATE_ONLY:
        seq_start_index = batch_id
        seq_len: tl.constexpr = 1
    else:
        seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
        seq_len = tl.load(seq_len_ptr + batch_id)

    # cache is [bsnd]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)

    kv_position = tl.load(input_pos_ptr + batch_id)

    cache_batch_offset = cache_loc * N_KV_HEADS * MAX_SEQ_LENGTH * D_HEAD
    cache_head_offset = kv_head_id * D_HEAD

    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    load_mask = seq_mask[:, None] * dhead_mask[None, :]

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS  # This needs to be a power-of-2
    q_head_offsets = kv_head_id * HEAD_RATIO + tl.arange(0, HEAD_BLOCK_SIZE)
    q_head_mask = q_head_offsets < (kv_head_id * HEAD_RATIO + HEAD_RATIO)

    q_batch_offset = seq_start_index * N_HEADS * D_HEAD

    kv_batch_offset = seq_start_index * N_KV_HEADS * D_HEAD
    kv_head_offset = cache_head_offset

    D2: tl.constexpr = D_HEAD // 2
    # input is interleaved as [2, D//2] in dim [D_HEAD].
    d2_offsets = tl.arange(0, D2)
    dhead_offsets1 = d2_offsets
    dhead_offsets2 = d2_offsets + D2
    d2_mask = dhead_offsets2 < D_HEAD
    d2_load_mask = seq_mask[:, None] * d2_mask[None, :]

    # offsets of [bsn]
    q_offsets_base = (
        q_batch_offset + seq_offsets[:, None, None] * N_HEADS * D_HEAD + q_head_offsets[None, :, None] * D_HEAD
    )
    q_offsets1 = q_offsets_base + dhead_offsets1[None, None, :]
    q_offsets2 = q_offsets_base + dhead_offsets2[None, None, :]
    q_mask = d2_load_mask[:, None, :] * q_head_mask[None, :, None]

    q1 = tl.load(q_ptr + q_offsets1, mask=q_mask).to(tl.float32)
    q2 = tl.load(q_ptr + q_offsets2, mask=q_mask).to(tl.float32)

    k_offsets_base = kv_batch_offset + seq_offsets[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset
    k_offsets1 = k_offsets_base + dhead_offsets1[None, :]
    k_offsets2 = k_offsets_base + dhead_offsets2[None, :]

    k1 = tl.load(k_ptr + k_offsets1, mask=d2_load_mask).to(tl.float32)
    k2 = tl.load(k_ptr + k_offsets2, mask=d2_load_mask).to(tl.float32)

    # -----------------------------------
    # torch version sin/cos
    # cos and sin values are interleaved in frequencies tensor.
    f_offsets = seq_offsets[:, None] * D2 + d2_offsets[None, :]
    cos_ref = tl.load(f_ptr + kv_position * D_HEAD + f_offsets * 2, mask=d2_load_mask).to(dtype=tl.float32)
    sin_ref = tl.load(f_ptr + kv_position * D_HEAD + f_offsets * 2 + 1, mask=d2_load_mask).to(dtype=tl.float32)

    qs1 = cos_ref[:, None, :] * q1 - sin_ref[:, None, :] * q2
    qs2 = sin_ref[:, None, :] * q1 + cos_ref[:, None, :] * q2

    tl.store(q_rope_ptr + q_offsets1, qs1, mask=q_mask)
    tl.store(q_rope_ptr + q_offsets2, qs2, mask=q_mask)

    ks1 = cos_ref * k1 - sin_ref * k2
    ks2 = sin_ref * k1 + cos_ref * k2

    # Write back to kv-caches
    vs = tl.load(
        v_ptr + kv_batch_offset + seq_offsets[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset + dhead_offsets[None, :],
        mask=load_mask,
    )

    kv_writeback_seq_offsets = seq_offsets + kv_position

    cache_offset_base = cache_batch_offset + kv_writeback_seq_offsets[:, None] * D_HEAD * N_KV_HEADS + cache_head_offset

    k_cache_offset1 = cache_offset_base + dhead_offsets1[None, :]
    k_cache_offset2 = cache_offset_base + dhead_offsets2[None, :]
    tl.store(k_cache_ptr + k_cache_offset1, ks1, mask=d2_load_mask)
    tl.store(k_cache_ptr + k_cache_offset2, ks2, mask=d2_load_mask)

    v_cache_offset = cache_offset_base + dhead_offsets[None, :]
    tl.store(v_cache_ptr + v_cache_offset, vs, load_mask)


"""
Kernels based on paged KV Cache.
Parameter infos:
    tensors:
    - q: [b*s, n, d], flattened queries.
    - k/v: [b*s, n, d], flattened key/value.
    - seq_len: [b], length of each sequence in the batch.
        `seq_len` can be 1 (generate) or larger (context).
    - seq_start: [b], start index of each sequence in b*s dim of q/k/v.
    - k_cache/v_cache: [num_pages, PAGE_SIZE, n, d], paged KV Cache.
        New-coming k/v is split into small group of PAGE_SIZE, and then
        mapped to incontinuous memory in KV Cache.
    - page_table: [b, max_num_pages_per_seq], mapping logic of each sequence.
    - cache_loc: [b], mapping logic of `batch_id` in q/k/v to index in `page_table`.
    - cache_len: [b], existing cached k/v length of each sequence.

    constexpr:
    - N_HEADS/N_KV_HEADS: shape of dim [n] in q or k/v.
    - D_HEAD: shape of dim [d] in q/k/v.
        Assuming power of 2.
    - SEQ_BLOCK: block size to split dim [s].
        Assuming power of 2.
        Split k/v in update kernel and split q in context/generate kernel.
    - MAX_SEQ_LENGTH: seq_len <= MAX_SEQ_LENGTH.
    - PAGE_SIZE: shape of each kv cache page,
        Assuming power of 2 and SEQ_BLOCK % PAGE_SIZE = 0.
    - PAGE_TABLE_STIDE: stride of dim [b] in `page_table`.

KV Cache access logic in update kernel:
    1. batch_id i access k[seq_start[i] : seq_start[i] + seq_len[i]]
        and can be split into pages [a:b] in the sequence.
    2. Look up cache_len[i] to find if the sequence has cached k/v.
    3. Look up page_table[cache_loc[i], cache_len[i] + a : cache_len[i] + b]
       to get the corresponding pages in the k_cache, with result [c:d].
    4. Then update k_cache[c:d] with the k value.

"""


@triton.jit
def update_paged_kv_cache(
    k_ptr,  # [B*S, N, D]
    v_ptr,  # [B*S, N, D]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_indices_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [num_pages, page_size, n, d]
    v_cache_ptr,  # [num_pages, page_size, n, d]
    cache_loc_ptr,  # [b] # index of the sequence in the page table.
    cache_len_ptr,  # [b] # length of the sequence already in kv cache.
    page_table_ptr,  # [b, max_num_pages_per_seq] # loc of the block page in the cache.
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_STRIDE: tl.constexpr,
    GENERATE_ONLY: tl.constexpr,
):
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    if GENERATE_ONLY:
        seq_start_index = batch_id
        seq_len: tl.constexpr = 1
    else:
        seq_start_index = tl.load(seq_start_indices_ptr + batch_id)
        seq_len = tl.load(seq_len_ptr + batch_id)

    cache_len = tl.load(cache_len_ptr + batch_id)

    # cache is [num_pages, page_size, n, d]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)
    cache_head_offset = head_id * D_HEAD

    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    seq_offsets = seq_block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    seq_mask = seq_offsets < seq_len

    load_mask = seq_mask[:, None] * dhead_mask[None, :]

    kv_batch_offset = seq_start_index * N_KV_HEADS * D_HEAD
    kv_head_offset = cache_head_offset

    # Write back to kv-caches
    ks = tl.load(
        k_ptr + kv_batch_offset + seq_offsets[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset + dhead_offsets[None, :],
        mask=load_mask,
    )
    vs = tl.load(
        v_ptr + kv_batch_offset + seq_offsets[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset + dhead_offsets[None, :],
        mask=load_mask,
    )

    # assuming SEQ_BLOCK can be divided by PAGE_SIZE and PAGE_SIZE is a power of 2.
    SEQ_BLOCK_PAGE: tl.constexpr = SEQ_BLOCK // PAGE_SIZE
    MAX_NUM_PAGES: tl.constexpr = (MAX_SEQ_LENGTH + PAGE_SIZE - 1) // PAGE_SIZE
    # cache_len // PAGE_SIZE means history pages
    # if decode sequence, then seq_len = 1 and only seq_block_id = 0 works,
    kv_pages = seq_block_id * SEQ_BLOCK_PAGE + tl.arange(0, SEQ_BLOCK_PAGE) + cache_len // PAGE_SIZE
    cache_pages = tl.load(page_table_ptr + cache_loc * PAGE_TABLE_STRIDE + kv_pages, mask=kv_pages < MAX_NUM_PAGES)

    page_offsets = tl.arange(0, PAGE_SIZE)
    # shape [SEQ_BLOCK], means [cache_pages, page_offsets]
    cache_seq_offset = tl.reshape(cache_pages[:, None] * PAGE_SIZE + page_offsets[None, :], [SEQ_BLOCK])
    # write offset inside the page
    cache_seq_offset += cache_len % PAGE_SIZE

    cache_offsets = cache_seq_offset[:, None] * N_KV_HEADS * D_HEAD + kv_head_offset + dhead_offsets[None, :]
    tl.store(k_cache_ptr + cache_offsets, ks, load_mask)
    tl.store(v_cache_ptr + cache_offsets, vs, load_mask)


# TODO: Write a doc describing the 2 stage algorithm
@triton.jit
def attention_kv_paged_stage1(
    q_ptr,  # [Batch, 1, N_HEADS, D_HEAD]
    k_cache_ptr,  # [NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD]
    v_cache_ptr,  # [NUM_PAGES, PAGE_SIZE, N_HEADS, D_HEAD]
    cache_loc_ptr,  # [Batch] # Specifies the batch index for each of the generate tokens.
    page_table_ptr,  # [Batch, num_pages_per_seq]
    cache_len_ptr,  # [Batch] # Number of tokens in kv cache.
    output_values_ptr,  # [Batch, N_HEADS, num_blocks, D_HEAD]
    output_logsumexp_ptr,  # [Batch, N_HEADS, num_blocks]
    num_blocks,
    MAX_SEQ_LEN: tl.constexpr,  # Maximum supported sequence length
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    # Block size used for tiling the sequence dim.
    SEQ_BLOCK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_STRIDE: tl.constexpr,
):
    """Attention kernel to be used during the generate phase.

    Uses flash decoding.
    KV-cache layout is assumed to be [Batch, Head, Seq, Dim]
    1. Fetch the K-cache from 0 to input_pos
    2. Fetch the V-cache from 0 to input_pos
    3. A = Q*K^T [1,D_HEAD] * [1,seq_len,D_HEAD] -> [1, seq_len]
    4. S = softmax(A)
    5. O = S*V [1, seq_len] * [1, seq_len, D_HEAD] -> [1, D_HEAD]
    """
    # Assume KV-cache layout: [Batch, Head, Seq, Dim]
    # A program is responsible for 1 batch, 1 head and a block of sequences.
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    SEQ_BLOCK_PAGE: tl.constexpr = SEQ_BLOCK_SIZE // PAGE_SIZE
    MAX_NUM_PAGES: tl.constexpr = MAX_SEQ_LEN // PAGE_SIZE

    cache_loc = tl.load(cache_loc_ptr + batch_id)
    seq_len = tl.load(cache_len_ptr + batch_id)
    # Offsets for the block of sequences this program processes.
    seq_start_pos = seq_block_id * SEQ_BLOCK_SIZE

    if seq_start_pos > seq_len:
        return
    seq_offsets = seq_start_pos + tl.arange(0, SEQ_BLOCK_SIZE)
    seq_mask = seq_offsets <= seq_len
    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS
    cache_head_offset = (head_id // HEAD_RATIO) * D_HEAD

    sm_scale: tl.constexpr = 1 / (D_HEAD**0.5)

    # Program loads the entire Q for the head assigned to it.
    # [D_HEAD]
    q_batch_offset = batch_id * N_HEADS * D_HEAD
    q_head_offset = head_id * D_HEAD
    q = tl.load(q_ptr + q_batch_offset + q_head_offset + dhead_offsets)

    kv_mask = seq_mask[:, None] * dhead_mask[None, :]

    kv_pages = seq_block_id * SEQ_BLOCK_PAGE + tl.arange(0, SEQ_BLOCK_PAGE)
    cache_pages = tl.load(page_table_ptr + cache_loc * PAGE_TABLE_STRIDE + kv_pages, mask=kv_pages < MAX_NUM_PAGES)

    page_offsets = tl.arange(0, PAGE_SIZE)
    # shape [SEQ_BLOCK], means [cache_pages, page_offsets]
    # token offsets in the paged kv cache
    cache_seq_offset = tl.reshape(cache_pages[:, None] * PAGE_SIZE + page_offsets[None, :], [SEQ_BLOCK_SIZE])

    cache_offsets = cache_seq_offset[:, None] * N_KV_HEADS * D_HEAD + cache_head_offset + dhead_offsets[None, :]

    k = tl.load(k_cache_ptr + cache_offsets, mask=kv_mask)
    v = tl.load(v_cache_ptr + cache_offsets, mask=kv_mask)

    # Note: check the output precision of the sum.
    # compute q*K^T
    # [D_HEAD] * [seq_block, D_HEAD], sum along axis 1
    attn = tl.sum(q[None, :] * k, axis=1)  # [seq_block]
    attn = attn.to(tl.float32)
    attn *= sm_scale
    max_attn = tl.max(attn)
    # Set to -inf attn values where mask is not set. This forces exp(attn) to 0.
    attn = tl.where(seq_mask, attn, float("-inf"))
    exp_attn = tl.exp(attn - max_attn)

    sumexp = tl.sum(exp_attn, axis=0)  # scalar.

    # [seq_len] * [seq_len, D_HEAD], sum along axis 0
    output = tl.sum(exp_attn[:, None] * v, axis=0)  # [D_HEAD]

    output = output / sumexp

    # We store the log-sum-exp after removing the max.
    logsumexp = tl.log(sumexp) + max_attn
    # when seq_mask is all false, max_attn will be -inf and sumexp is zero

    tl.store(
        output_values_ptr
        + batch_id * N_HEADS * D_HEAD * num_blocks
        + head_id * D_HEAD * num_blocks
        + seq_block_id * D_HEAD
        + dhead_offsets,
        output,
    )
    tl.store(
        output_logsumexp_ptr + batch_id * N_HEADS * num_blocks + head_id * num_blocks + seq_block_id,
        logsumexp,
    )


@triton.jit
def context_attention_kv_paged(
    q_ptr,  # [b*s,nd]
    seq_len_ptr,  # [b] # length of each sequence in a batch
    seq_start_ptr,  # [b] # start indices of a sequence in flattened q/k/v.
    k_cache_ptr,  # [num_pages, page_size, n, d]
    v_cache_ptr,  # [num_pages, page_size, n, d]
    cache_loc_ptr,  # [b] # index of the sequence in the page table.
    cache_len_ptr,  # [Batch] # Number of tokens in kv cache.
    page_table_ptr,  # [b, max_num_pages_per_seq] # loc of the block page in the cache.
    softmax_scale,
    o_ptr,
    N_HEADS: tl.constexpr,  # Number of heads
    N_KV_HEADS: tl.constexpr,  # Number of KV heads.
    D_HEAD: tl.constexpr,  # Dimension of each head.
    SEQ_BLOCK: tl.constexpr,
    MAX_SEQ_LENGTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_STRIDE: tl.constexpr,
):
    """Kernel for context phase.

    Fuses rope
    Assuming:
    1. Self-attention [seqlen(Q) == seqlen(K)]
    2. Causal attention
    3. QKV layout: [b*s,n,d]
    """
    batch_id = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)
    seq_block_id = tl.program_id(axis=2)

    # Each program is responsible for a block of tokens in a single batch.
    seq_start_index = tl.load(seq_start_ptr + batch_id)
    seq_len = tl.load(seq_len_ptr + batch_id)

    HEAD_RATIO: tl.constexpr = N_HEADS // N_KV_HEADS

    # assuming SEQ_BLOCK can be divided by PAGE_SIZE and PAGE_SIZE is a power of 2.
    SEQ_BLOCK_PAGE: tl.constexpr = SEQ_BLOCK // PAGE_SIZE
    MAX_NUM_PAGES: tl.constexpr = (MAX_SEQ_LENGTH + PAGE_SIZE - 1) // PAGE_SIZE

    # cache is [num_pages, page_size, n, d]
    # cache_loc_ptr stores the batch index for the sequences provided to the kernel.
    cache_loc = tl.load(cache_loc_ptr + batch_id)
    table_batch_offset = cache_loc * PAGE_TABLE_STRIDE

    # Assuming D_HEAD is a power of 2
    dhead_offsets = tl.arange(0, D_HEAD)
    dhead_mask = dhead_offsets < D_HEAD

    seq_offsets = tl.arange(0, SEQ_BLOCK)
    q_seq_offsets = seq_block_id * SEQ_BLOCK + seq_offsets
    seq_mask = q_seq_offsets < seq_len

    load_mask = seq_mask[:, None] * dhead_mask[None, :]

    q_batch_offset = seq_start_index * N_HEADS * D_HEAD
    q_head_offset = head_id * D_HEAD
    cache_head_offset = (head_id // HEAD_RATIO) * D_HEAD

    # Q will stay in SRAM
    q = tl.load(
        q_ptr + q_batch_offset + q_seq_offsets[:, None] * N_HEADS * D_HEAD + q_head_offset + dhead_offsets[None, :],
        mask=load_mask,
    )
    acc = tl.zeros([SEQ_BLOCK, D_HEAD], dtype=tl.float32)
    lse_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([SEQ_BLOCK], dtype=tl.float32) - float("inf")

    cache_len = tl.load(cache_len_ptr + batch_id)
    total_len = cache_len + seq_len
    num_blocks = (total_len + SEQ_BLOCK - 1) // SEQ_BLOCK
    for s in range(0, num_blocks + 1, 1):
        kv_pages = s * SEQ_BLOCK_PAGE + tl.arange(0, SEQ_BLOCK_PAGE)
        cache_pages = tl.load(page_table_ptr + table_batch_offset + kv_pages, mask=kv_pages < MAX_NUM_PAGES)

        page_offsets = tl.arange(0, PAGE_SIZE)
        # shape [SEQ_BLOCK], means [cache_pages, page_offsets]
        # physical token offsets in the paged kv cache
        cache_seq_offset = tl.reshape(cache_pages[:, None] * PAGE_SIZE + page_offsets[None, :], [SEQ_BLOCK])
        cache_offsets = cache_seq_offset[:, None] * N_KV_HEADS * D_HEAD + cache_head_offset + dhead_offsets[None, :]

        # logical kv tokens offsets
        kv_seq_offsets = s * SEQ_BLOCK + seq_offsets
        kv_seq_mask = kv_seq_offsets < total_len
        kv_load_mask = kv_seq_mask[:, None] * dhead_mask[None, :]

        k = tl.load(k_cache_ptr + cache_offsets, mask=kv_load_mask)
        qk = tl.zeros([SEQ_BLOCK, SEQ_BLOCK], dtype=tl.float32)
        qk += tl.dot(q, k.trans())
        # causal mask, need to use kv_seq_offsets
        qk = tl.where((q_seq_offsets[:, None] + cache_len) >= kv_seq_offsets[None, :], qk, float("-inf"))

        qk *= softmax_scale
        # rowmax
        m_ij = tl.maximum(tl.max(qk, 1), lse_i)
        p = tl.exp(qk - m_ij[:, None])
        v = tl.load(v_cache_ptr + cache_offsets, mask=kv_load_mask)

        l_ij = tl.sum(p, 1)
        acc_scale = tl.exp(m_i - m_ij)
        acc = acc * acc_scale[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)

    acc = acc * o_scale[:, None]

    tl.store(
        o_ptr + q_batch_offset + q_seq_offsets[:, None] * N_HEADS * D_HEAD + q_head_offset + dhead_offsets[None, :],
        acc,
        mask=load_mask,
    )


@dataclass
class PositionalEmbeddingConfig:
    """A dataclass to hold positional embedding information."""

    mode: Optional[Literal["rope"]] = None
    rope_theta: float = 10000.0
    rope_scale: float = 1.0

    def __post_init__(self):
        assert self.mode in [None, "rope"], f"Invalid mode: {self.mode}."
        if self.mode == "rope":
            assert self.rope_theta > 0, f"Invalid rope theta: {self.rope_theta}."


@dataclass
class CacheConfig:
    """A dataclass to hold information how to configure the cache."""

    dtype: Optional[torch.dtype] = None


@dataclass
class AttentionInfo:
    """Information about the attention op.

    This is the dataclass collected by the kvcache transformation and passed in to the
    AttentionDescriptor methods to inform the attention op about the attention configuration.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int  # embedding size of each head
    dtype: torch.dtype

    cache_config: CacheConfig
    pos_embd_config: PositionalEmbeddingConfig
    # rope_dim represents embedding size of decoupled q/k that carry rope information
    # when rope_dim != 0 the decoupled q/k tensor carrying rope information is the last part of the tensor [-rope_dim: ]
    rope_dim: Optional[int] = 0


@dataclass
class SequenceInfo:
    """A dataclass to hold information about how the sequence is laid out and stored in cache.

    We assume the sequence + cache is laid out in the following way:

    - input_ids: [id_0, ..., id_{s_total-1}]
      flattened sequence of [b, 1] or [1, s_total]. We use [b, 1] to denote generate-only batches.
    - seq_len: [s_0, s_1, ..., s_{b-1}] such that s_total = sum(s_i)
      Describes how long each sequence is. For example,
      input_ids[:s_0] will correspond to sequence 0 in the batch and input_ids[s_0:s_1] will
      correspond to sequence 1 in the batch.
    - input_pos: [pos_0, ..., pos_{b-1}]
      Corresponds to the total number of tokens that has been already been cached for each sequence
      in the batch.
    - cache_loc: [c0, ...., c_{np-1}] where np is total number of pages allocated to describe all
      sequences in the batch.
    - pages_per_seq: [ps_0, ps_1, ..., ps_{b-1}] where ps_i is the number of pages allocated for
      sequence i. Note that, for example, cache_loc[p_0:p_1] will correspond to the pages associated
      with sequence 1 in the batch.

    Here are a couple of notes to emphasize this notation:

    - The total number of allocated token space for sequence i is given by ps_i * page_size. This is
      the total number of tokens that can be cached for each sequence.

    - NOTE: It must hold that pos_i + s_i <= ps_i * page_size for all i in [0, b-1]. Moreover, it is
      the responsibility of the cache manager and/or runtime to ensure sufficient page allocation
      for each sequence.

    """

    ## USE TO INITIALIZE DATA CLASS  ###############################################################
    # max_seq_len corresponds the maximum number of tokens in any sequence. It includes the tokens in the
    # input sequence and the tokens generated by the model.
    max_seq_len: int = 1
    # max_batch_size corresponds to the maximum number of sequences (or requests) that the model can process.
    max_batch_size: int = 1
    # page_size is the granularity with which the cache pages are allocated for a paged kv cache.
    # For an unpaged cache, the page size should be set to max_seq_len.
    # Also note that two sequences in a batch can not share a page.
    page_size: int = 0
    # max_num_tokens is the maximum number of tokens that the model can process across all sequences in the batch.
    # If a batch is composed of context-only requests of input sequence length ISL,
    # then the maximum number of sequences possible in the batch is min (max_batch_size, max_num_tokens // ISL).
    # Similarly, if a batch is composed of generate-only requests,
    # then the maximum number of sequences possible in the batch is min (max_batch_size, max_num_tokens).
    max_num_tokens: int = 0

    ## [UPDATE WITH CARE] TENSOR FIELDS THAT WILL BE PASSED TO PREPARE_METADATA OP #################
    # input_ids MUST ALWAYS BE THE FIRST FIELD
    input_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, dtype=torch.int))
    seq_len: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
    input_pos: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.int))
    cache_loc: torch.Tensor = field(default_factory=lambda: torch.arange(1, dtype=torch.int))
    pages_per_seq: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
    ################################################################################################

    ## PRIVATE FIELDS ##############################################################################
    _sequence_lengths: List[int] = field(default_factory=list)
    _num_pages: int = 1

    def __post_init__(self):
        if self.page_size < 1:
            self.page_size = self.max_seq_len
        if self.max_num_tokens < 1:
            self.max_num_tokens = self.max_batch_size * self.max_seq_len
        # if the provided max_num_tokens is less than the max_batch_size * max_seq_len,
        # we use the provided max_num_tokens to calculate the number of pages
        total_tokens = min(self.max_num_tokens, self.max_batch_size * self.max_seq_len)
        self._num_pages = (total_tokens) // self.page_size + (total_tokens % self.page_size > 0)
        self.input_ids = torch.ones(self.max_batch_size, 1, dtype=torch.int)
        self.seq_len = torch.empty(self.max_batch_size, dtype=torch.int)
        self.input_pos = torch.empty_like(self.seq_len)
        self.cache_loc = torch.empty(self.num_pages, dtype=torch.int)
        self.pages_per_seq = torch.empty_like(self.seq_len)

        # dynamic shape descriptors for tensor args
        self._dynamic_shapes: Optional[Tuple[Dict[str, Dim]]] = None

        # keep a list-like object of sequence lengths for simplicity as well
        self._sequence_lengths = [0] * self.max_batch_size

        # call reset once to initialize the tensors
        self.reset()

    @property
    def device(self) -> torch.device:
        return self.input_pos.device

    @property
    def args(self) -> List[torch.Tensor]:
        args = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                args.append(val)
        return args

    @property
    def extra_arg_names(self) -> List[str]:
        """Return extra arg names for the prepare_metadata op beyond input_ids."""
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)][1:]

    @property
    def dynamic_shapes(self) -> Tuple[Dict[str, Dim]]:
        """Return dynamic shapes of sequence info tensors.

        NOTE: will be lazily initialized since the Dim object is not picklable for multi-processing.
        """
        if self._dynamic_shapes is None:
            dynamic_shapes = ({},)
            if self.max_batch_size > 1:
                dynamic_shapes[0][0] = Dim("batch_size", max=self.max_batch_size)
            dynamic_shapes[0][1] = Dim("seq_len", max=self.max_seq_len)
            dynamic_shapes += ({},) * len(self.extra_arg_names)
            self._dynamic_shapes = dynamic_shapes
        return self._dynamic_shapes

    @property
    def num_sequences(self) -> int:
        return len(self._sequence_lengths)

    @property
    def sequence_lengths(self) -> List[int]:
        return self._sequence_lengths

    @property
    def input_positions(self) -> List[int]:
        return self.input_pos[: self.num_sequences].tolist()

    @property
    def is_generate(self) -> bool:
        return all(sl == 1 for sl in self.sequence_lengths)

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @num_pages.setter
    def num_pages(self, value):
        self._num_pages = value
        # update the cache_loc tensor
        self.cache_loc.resize_(value)

    @property
    def is_paged(self) -> bool:
        return self.page_size < self.max_seq_len

    @property
    def page_assignments(self) -> List[List[int]]:
        """Return the page assignments for each sequence."""
        pages_per_seq = self.pages_per_seq[: self.num_sequences].tolist()
        return [
            c_loc_one_seq.tolist() for c_loc_one_seq in torch.split(self.cache_loc[: sum(pages_per_seq)], pages_per_seq)
        ]

    @classmethod
    def _get_sanitized_seq_len(cls, input_ids: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Sanitize sequence lengths.

        We want to cover the following scenarios with this function:

        1. Pre-fill:
            input_ids: [1, s_total, ...]
            seq_len: [s_0, s_1, ..., s_{b-1}, 0, 0, ..., 0]
            ---> returns [s_0, s_1, ..., s_{b-1}]
        2. Decode:
            input_ids: [b, 1, ...]
            seq_len: [1, 1, ..., 1, 0, 0, ..., ..., ..., ..., 0]
                     |---- b ----|--- (max_batch_size - b) ---|
            --> returns [1,] * b
        3. Decode in Cudagraph:
            input_ids: [b_cudagraph, 1, ...]
            seq_len: [1, 1, ..., 1, 0, 0, ..., ..., ..., ..., 0]
                     |---- b ----|--- (max_batch_size - b) ---|

            --> returns [1,] * b_cudagraph
            Here b <= b_cudagraph. We want to make sure that the seq_len is one-padded to
            b_cudagraph.

            # TODO: I could see one possible issue with this approach in the future.
            # If we have b < b_cudagraph we now one-pad. However, we don't pad the cache location
            # information. What could happen is that the for the padded sequences the cache location
            # tensors point to allocated pages. This could lead to a situation where we write into
            # allocated cache pages polluting the cache of other sequences. Now this is not an issue
            # if we write the dummy sequences into unallocated cache pages... One fix could be to
            # pad not only the seq len but also pad the cache locations by just repeating the last
            # valid cache location in the batch. This would ensure that the dummy sequences just
            # repeats valid computation...
        """
        _, s = input_ids.shape[:2]
        num_seq = cls._get_sanitized_num_sequences(input_ids, seq_len)
        if s > 1:
            return seq_len[:num_seq].detach().clone()
        else:
            return torch.ones(num_seq, dtype=seq_len.dtype, device=seq_len.device)

    @staticmethod
    def _get_sanitized_num_sequences(input_ids: torch.Tensor, seq_len: torch.Tensor) -> int:
        """Get number of sequences.

        We makes sure that this function is compatible with both torch graph capture and cudagraph.
        Both can be a bit temparamental when trying to extract the number of sequences from a tensor
        with max_batch_size or max_batch_size*max_seq_len.
        """
        b, s = input_ids.shape[:2]
        if s > 1:
            num_seq = torch.sum(seq_len > 0)
            assert seq_len[num_seq:].sum() == 0, "seq_len should be zero-padded"
        else:
            num_seq = b
        return num_seq

    def to(self, *args, **kwargs) -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(*args, **kwargs))

    def sync(self, other: "SequenceInfo") -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            val_other = getattr(other, f.name)
            if f.name == "input_ids":
                setattr(self, f.name, val_other.to(self.device))
            elif f.name == "_sequence_lengths":
                self._sequence_lengths = val_other
            elif isinstance(val, torch.Tensor):
                val[: len(val_other)] = val_other.to(self.device)
            else:
                assert val == val_other, f"Field {f.name} mismatch: {val} != {val_other}."

    def reset(self) -> None:
        """Reset the sequence information.

        After reset the sequence information should correspond to a "generate-only" batch of
        sequences (b, s==1) without cache history.
        """
        # set a dummy sequence corresponding to a generate-only batch
        self.nest_sequences(torch.zeros(self.max_batch_size, 1, dtype=torch.int))

        # reset cache information
        self.input_pos.zero_()
        self.cache_loc[:] = torch.arange(self.num_pages, dtype=torch.int, device=self.device)
        self.pages_per_seq.fill_(1)

    def _set_example_sequence(self) -> None:
        """Set an example sequence for export purposes."""
        self.reset()
        input_ids = torch.ones(
            min(2, self.max_batch_size),
            min(4, self.max_seq_len),
            dtype=torch.int,
            device=self.device,
        )
        self.nest_sequences(input_ids)
        self.input_ids = input_ids

    def _set_max_num_tokens_sample(self) -> None:
        """Set an example sequence with max_num_tokens."""
        self.reset()
        seq_len = self.max_num_tokens // self.max_batch_size
        input_ids = torch.ones(
            self.max_batch_size,
            seq_len,
            dtype=torch.int,
            device=self.device,
        )
        self.pages_per_seq.fill_(seq_len // self.page_size)
        self.nest_sequences(input_ids)

    def _set_generate_only_batch(self) -> None:
        """Set an example sequence for generate-only batch."""
        self.reset()
        self.nest_sequences([[1]] * self.max_batch_size)

    def nest_sequences(self, input_ids: Sequence[Sequence[int]]) -> None:
        """Create and store a flattened list of input_ids from the provided list of sequences.

        This i/f will also update any relevant sequence information.
        """
        # set new sequence lengths
        seq_lens = [len(ids) for ids in input_ids]
        self.seq_len.zero_()
        self.seq_len[: len(seq_lens)].copy_(torch.tensor(seq_lens), non_blocking=True)

        # set new input_ids as new tensor from flattened input_ids
        ids_tnsr_list = [
            lst.detach() if isinstance(lst, torch.Tensor) else torch.tensor(lst, dtype=torch.int) for lst in input_ids
        ]
        self.input_ids = torch.cat(ids_tnsr_list, dim=0).to(self.device)

        # set derivative properties
        self._sequence_lengths = seq_lens

        # use [b,1] shape to indicate generate-only batch, otherwise use [1,total_len]
        if self.is_generate:
            self.input_ids = self.input_ids.view(-1, 1, *self.input_ids.shape[1:])
        else:
            self.input_ids = self.input_ids.view(1, -1, *self.input_ids.shape[1:])

    def unnest_sequences(self, t_nested: torch.Tensor) -> List[torch.Tensor]:
        t_squeezed = t_nested.squeeze(1) if self.is_generate else t_nested.squeeze(0)
        return list(torch.split(t_squeezed, self.sequence_lengths))

    def update_pos(self, seq_len: Union[torch.Tensor, List[int], int], reset: bool = False) -> None:
        """Update the starting position for each sequence in the cache.

        If ``reset=True`, ``input_pos`` will be reset to zero before updating.
        """
        if not isinstance(seq_len, torch.Tensor):
            seq_len = torch.tensor(seq_len, dtype=torch.int)
        bs = len(seq_len) if seq_len.dim() > 0 else self.max_batch_size

        if reset:
            self.input_pos[:bs] = seq_len.to(self.device)
        else:
            self.input_pos[:bs] += seq_len.to(self.device)

    def assign_cache_loc(self, page_assignments: Sequence[Sequence[int]]) -> None:
        """Set the cache location and pages_per_seq tensors from page assignments."""
        cache_loc_flat = torch.tensor([p_idx for pages in page_assignments for p_idx in pages], dtype=torch.int)
        self.cache_loc[: len(cache_loc_flat)].copy_(cache_loc_flat, non_blocking=True)

        pages_per_seq = torch.tensor([len(p) for p in page_assignments], dtype=torch.int)
        self.pages_per_seq[: len(pages_per_seq)].copy_(pages_per_seq, non_blocking=True)


Constant = Union[int, float, str, None]


class MHACallable(Protocol):
    def __call__(
        self,
        *qkv_metadata_and_caches: Union[torch.Tensor, Constant],
    ) -> torch.Tensor: ...


class PrepareMetadataCallable(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        seq_len: torch.Tensor,
        input_pos: torch.Tensor,
        cache_loc: torch.Tensor,
        pages_per_seq: torch.Tensor,
        page_size: int,
    ) -> List[torch.Tensor]: ...


class GetCacheCallable(Protocol):
    def __call__(self, sequence_info: SequenceInfo) -> torch.Tensor: ...


class GetBufferCallable(GetCacheCallable):
    pass


class GetAttentionInfo(Protocol):
    def __call__() -> AttentionInfo: ...


CacheInitializerDict = Dict[str, GetCacheCallable]
BufferInitializerDict = Dict[str, GetBufferCallable]


class AttentionDescriptor(ABC):
    """An interface to define a functional attention operator.

    The main logic is contained with the actual attention op as well as the prepare_metadata op. The
    prepare_metadata op is responsible for converting the standardized sequence info into metadata
    specific to the attention op.
    """

    @classmethod
    @abstractmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""

    @classmethod
    def get_attention_op(cls) -> Tuple[MHACallable, int]:
        """Get the attention op and the number of arguments corresponding to qkv.

        The attention_op should follow the below signature:

        ```
        def attention_op(
            *qkv,       # list of tensors corresponding to Q, K, V as in original op
            *metadata,  # global info about the sequences as returned by the prepare_metadata op
            *caches,    # contains layer-specific caches per provided cache initializers
            *buffers,   # global buffers used by the attention op as provided by buffer initializers
            *constants, # basic arguments (int, float, str, None) added as CONSTANTS in the graph
        ) -> torch.Tensor: ...
        ```

        **Note that the attention op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**

        **Note that the `qkv` tuple should be consistent across both the cached attention
        op and the op that it is replacing.**

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        """Get the prepare_metadata op.

        The prepare_metadata op should follow the below signature:

        ```
        def prepare_metadata(
            input_ids: torch.Tensor,
            seq_len: torch.Tensor,
            input_pos: torch.Tensor,
            cache_loc: torch.Tensor,
        ) -> List[torch.Tensor]: ...
        ```
        The metadata should contain all necessary global information required for the underlying
        attention op to process the input sequence and the returned list of tensors will be passed
        on to each invocation of the attention op in the graph.

        prepare_metadata is called once at the beginning of the forward pass.

        **Note that the prepare_metadata op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**
        """
        return NotImplementedError

    @classmethod
    @abstractmethod
    def get_cache_initializers(cls, get_info: GetAttentionInfo) -> CacheInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize the caches.

        The key corresponds to the argument name used in the attention op signature. The function
        key doesn't need to be unique across multiple attention nodes in the graph. The key used to
        describe the cache in the graph will be patched with the attention node index to ensure
        uniqueness.

        ``get_cache_initializers`` will be called *once* after the model initialization and before
        the initial forward pass for each attention op detected in the graph. The caches will be
        managed by the global CacheManager and passed back to the attention op during the forward
        pass.

        If the cache initializer requires information about the attention op, the ``get_info``
        function can be called **inside** the cache initializer to retrieve the necessary
        information.
        """
        raise NotImplementedError

    @classmethod
    def get_global_buffer_initializers(cls, get_info: GetAttentionInfo) -> BufferInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize buffers.

        The key corresponds to the buffer name used in the graph module and will **not**
        be patched unlike a cache key. Hence, it is a **global** key that is shared across all
        attention ops in the model much like a regular buffer in an nn.Module. That means if this
        i/f is called for multiple attention ops, the same buffer will be shared across all of them
        if this function provides the same key multiple times.

        Buffers are initialize *once* after the model initialization and before the initial forward
        pass for each attention op detected in the graph. The buffer will be managed by the global
        CacheManager and passed back to the attention op during the forward pass.

        If the buffer initializer requires information about the attention op, the ``get_info``
        function can be called **inside** the buffer initializer to retrieve the necessary
        information.
        """
        return {}

    @classmethod
    def get_constants(cls, attention_info: AttentionInfo) -> List[Constant]:
        """Provide a list of constant arguments to be passed to the attention op.

        The constant arguments are passed to the attention op as additional arguments after the
        caches and buffers. The constants are expected to be of type int, float, str, or None.
        """
        return []


class AttentionRegistry:
    """A simple registry to look up different attention implementations."""

    _attention_registry: Dict[str, Type["AttentionDescriptor"]] = {}

    @classmethod
    def register(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        def decorator(attention_cls: Type["AttentionDescriptor"]):
            assert kernel_source not in cls._attention_registry, f"Attention source {kernel_source} already registered."
            cls._attention_registry[kernel_source] = attention_cls
            return attention_cls

        return decorator

    @classmethod
    def get(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        assert cls.has(kernel_source), f"Attention source {kernel_source} not registered."
        return cls._attention_registry[kernel_source]

    @classmethod
    def has(cls, kernel_source: str) -> bool:
        return kernel_source in cls._attention_registry


@torch.library.custom_op("attention::scaled_dot_product_attention", mutates_args=())
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """A carbon copy of torch.nn.functional.scaled_dot_product_attention as custom op.

    Using this custom op instead of using the functional directly ensures consistent representation
    of the vanilla sdpa in a graph.
    """
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@scaled_dot_product_attention.register_fake
def scaled_dot_product_attention_fake(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Fake implementation of scaled_dot_product_attention."""
    return torch.empty_like(query)


def _generate_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_locs: torch.Tensor,
    input_pos: torch.Tensor,
    out: torch.Tensor,
):
    b, (n_heads, q_d_head) = q.shape[0], q.shape[-2:]
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    device = q.device

    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    SEQ_BLOCK_SIZE = 256
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE

    stage1_output_values = torch.empty(b, n_heads, num_blocks, v_d_head, device=device, dtype=torch.float32)
    stage1_output_logsumexp = torch.empty(b, n_heads, num_blocks, device=device, dtype=torch.float32) - float("inf")

    (
        update_kv_cache[(b, n_kv_heads, 1)](
            k,
            v,
            None,
            None,
            k_cache,
            v_cache,
            input_pos,
            cache_locs,
            max_seq_len,
            n_kv_heads,
            q_d_head,
            v_d_head,
            1,
            GENERATE_ONLY=True,
        ),
    )

    gqa_attention_kv_stage1[
        (
            b,
            n_kv_heads,
            num_blocks,
        )
    ](
        q,
        k_cache,
        v_cache,
        cache_locs,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        max_seq_len,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK_SIZE,
        HEAD_BLOCK_SIZE,
    )
    attention_kv_stage2[(b, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        v_d_head,
        SEQ_BLOCK_SIZE,
    )


def _context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
):
    b, s, n_heads, q_d_head = q.shape
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]

    SEQ_BLOCK = 128
    softmax_scale = 1.0 / math.sqrt(q_d_head)
    grid = (b, n_heads, (s + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        s,
        out,
        softmax_scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_mha_with_cache", mutates_args=())
def fused_mha_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused MHA with cache that takes raw input from q, k, v GEMMs."""
    # b, s info
    b, s = q.shape[:2]
    head_dim = k_cache.shape[-1]

    # reshapes with num_heads and head_dim
    q = q.view(b, s, -1, head_dim)
    k = k.view(b, s, -1, head_dim)
    v = v.view(b, s, -1, head_dim)

    # rope embedding
    if freqs_cis is not None:
        q = torch.ops.rope.apply_rope_with_input_pos(q, freqs_cis, input_pos, "bsnd")
        k = torch.ops.rope.apply_rope_with_input_pos(k, freqs_cis, input_pos, "bsnd")

    # attention (assumed layout is bsnd)
    y = torch.empty_like(q)
    if s > 1:
        # context phase
        _context_mha(q, k, v, k_cache, v_cache, y)
    else:
        # generate phase
        cache_locs = torch.arange(0, b, device=q.device, dtype=torch.int32)
        _generate_mha(q, k, v, k_cache, v_cache, cache_locs, input_pos, y)

    return y.view(b, s, -1)  # [b,s,n*h_d]


@fused_mha_with_cache.register_fake
def fused_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    return torch.empty_like(q.contiguous())


def _flattened_context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, q_d_head = q.shape
    max_cache_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    BATCH_SIZE: int = len(input_pos)
    SEQ_BLOCK = 32
    (
        update_kv_cache[(BATCH_SIZE, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)](
            k,
            v,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            input_pos,
            cache_loc,
            max_cache_seq_len,
            n_kv_heads,
            q_d_head,
            v_d_head,
            32,
            GENERATE_ONLY=False,
        ),
    )
    # TODO: use input_pos to get the correct cache locations
    softmax_scale = 1.0 / math.sqrt(q_d_head)
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_flattened[grid](
        q,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        out,
        softmax_scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_flattened_mha_with_cache", mutates_args=())
def fused_flattened_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    freqs_cis: torch.Tensor,
    # CONSTANTS
    # <none>
) -> torch.Tensor:
    """Flattened & fused MHA with cache that takes raw input from q, k, v GEMMs.

    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    head_dim = k_cache.shape[-1]
    b, s, d = q.shape

    # reshapes with num_heads and head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)
    q = q.view(*bs_view, q.shape[2] // head_dim, head_dim)
    k = k.view(*bs_view, k.shape[2] // head_dim, head_dim)
    v = v.view(*bs_view, v.shape[2] // head_dim, head_dim)

    # rope embedding for generate-only or mixed
    if freqs_cis is not None and freqs_cis.numel() > 0:
        if s == 1:
            rope_args = (freqs_cis, input_pos, "bsnd")
            fn_rope = torch.ops.rope.apply_rope_with_input_pos
        else:
            rope_args = (freqs_cis, input_pos, seq_len, seq_start)
            fn_rope = torch.ops.rope.apply_rope_on_flattened_inputs
        q = fn_rope(q, *rope_args)
        k = fn_rope(k, *rope_args)

    # run attention
    y = torch.empty_like(q)
    if s == 1:
        # generate-only phase
        _generate_mha(q, k, v, k_cache, v_cache, cache_loc, input_pos, y)
    else:
        # mixed context + generate phase
        _flattened_context_mha(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            y,
        )

    return y.view(b, s, d)  # [b,s,n*h_d]


@fused_flattened_mha_with_cache.register_fake
def fused_flattened_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    return torch.empty_like(q.contiguous())


def _generate_mha_rope_fusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_locs: torch.Tensor,
    input_pos: torch.Tensor,
    out: torch.Tensor,
):
    b, (n_heads, d_head) = q.shape[0], q.shape[-2:]
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    device = q.device

    SEQ_BLOCK_SIZE = 64
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE
    stage1_output_values = torch.empty(b, n_heads, num_blocks, d_head, device=device, dtype=torch.float32)
    stage1_output_logsumexp = torch.empty(b, n_heads, num_blocks, device=device, dtype=torch.float32) - float("inf")
    q_rope = torch.empty_like(q)
    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))

    (
        update_kv_cache_rope_fusion[(b, n_kv_heads, 1)](
            q,
            k,
            v,
            None,
            None,
            q_rope,
            k_cache,
            v_cache,
            input_pos,
            cache_locs,
            freqs_cis,
            max_seq_len,
            n_heads,
            n_kv_heads,
            d_head,
            1,
            HEAD_BLOCK_SIZE,
            GENERATE_ONLY=True,
        ),
    )

    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    gqa_attention_kv_stage1[
        (
            b,
            n_kv_heads,
            num_blocks,
        )
    ](
        q_rope,
        k_cache,
        v_cache,
        cache_locs,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        max_seq_len,
        n_heads,
        n_kv_heads,
        d_head,
        d_head,
        SEQ_BLOCK_SIZE,
        HEAD_BLOCK_SIZE,
    )
    attention_kv_stage2[(b, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        d_head,
        SEQ_BLOCK_SIZE,
    )


def _flattened_context_mha_rope_fusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, d_head = q.shape
    max_cache_seq_len, n_kv_heads = k_cache.shape[1:3]
    BATCH_SIZE: int = len(input_pos)
    SEQ_BLOCK = 32
    q_rope = torch.empty_like(q)
    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    (
        update_kv_cache_rope_fusion[(BATCH_SIZE, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)](
            q,
            k,
            v,
            seq_len,
            seq_start,
            q_rope,
            k_cache,
            v_cache,
            input_pos,
            cache_loc,
            freqs_cis,
            max_cache_seq_len,
            n_heads,
            n_kv_heads,
            d_head,
            32,
            HEAD_BLOCK_SIZE,
            GENERATE_ONLY=False,
        ),
    )
    # TODO: use input_pos to get the correct cache locations
    softmax_scale = 1.0 / math.sqrt(d_head)
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_flattened[grid](
        q_rope,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        out,
        softmax_scale,
        n_heads,
        n_kv_heads,
        d_head,
        d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_flattened_mha_with_cache_rope_fusion", mutates_args=())
def fused_flattened_mha_with_cache_rope_fusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Flattened & fused MHA with cache that takes raw input from q, k, v GEMMs.

    Fuse k rope in update_kv_cache and q rope in attention.
    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # this function only handle requests with rope embadding.
    if freqs_cis is None:
        return fused_flattened_mha_with_cache(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            freqs_cis,
        )

    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    b, s, d = q.shape
    head_dim = k_cache.shape[-1]

    # reshapes with num_heads and head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)
    q = q.view(*bs_view, q.shape[2] // head_dim, head_dim)
    k = k.view(*bs_view, k.shape[2] // head_dim, head_dim)
    v = v.view(*bs_view, v.shape[2] // head_dim, head_dim)

    # run attention
    y = torch.empty_like(q)
    if s == 1:
        # generate-only phase
        _generate_mha_rope_fusion(q, k, v, freqs_cis, k_cache, v_cache, cache_loc, input_pos, y)
    else:
        # mixed context + generate phase
        _flattened_context_mha_rope_fusion(
            q,
            k,
            v,
            freqs_cis,
            input_pos,
            cache_loc,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            y,
        )

    return y.view(b, s, d)  # [b,s,n*h_d]


@fused_flattened_mha_with_cache_rope_fusion.register_fake
def fused_flattened_mha_with_cache_rope_fusion_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    return torch.empty_like(q.contiguous())


def _paged_generate_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    input_pos: torch.Tensor,
    out: torch.Tensor,
    max_seq_len: int,
):
    b, (n_heads, d_head) = q.shape[0], q.shape[-2:]
    PAGE_SIZE, n_kv_heads = k_cache.shape[1:3]
    device = q.device

    SEQ_BLOCK_SIZE = PAGE_SIZE  # 256
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE
    stage1_output_values = torch.empty(b, n_heads, num_blocks, d_head, device=device, dtype=torch.float32)
    stage1_output_logsumexp = torch.empty(b, n_heads, num_blocks, device=device, dtype=torch.float32) - float("inf")

    (
        update_paged_kv_cache[(b, n_kv_heads, 1)](
            k,
            v,
            None,
            None,
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            page_table,
            n_kv_heads,
            d_head,
            SEQ_BLOCK_SIZE,
            max_seq_len,
            PAGE_SIZE,
            page_table.stride(0),
            GENERATE_ONLY=True,
        ),
    )

    attention_kv_paged_stage1[
        (
            b,
            n_heads,
            num_blocks,
        )
    ](
        q,
        k_cache,
        v_cache,
        cache_loc,
        page_table,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        max_seq_len,
        n_heads,
        n_kv_heads,
        d_head,
        SEQ_BLOCK_SIZE,
        PAGE_SIZE,
        page_table.stride(0),
    )
    attention_kv_stage2[(b, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        d_head,
        SEQ_BLOCK_SIZE,
    )


def _paged_context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    out: torch.Tensor,
    max_seq_len: int,  # max cache length of sequence, kv_cache shape don't provide this info.
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, d_head = q.shape
    PAGE_SIZE, n_kv_heads = k_cache.shape[1:3]
    BATCH_SIZE = len(input_pos)
    SEQ_BLOCK = PAGE_SIZE  # 32
    (
        update_paged_kv_cache[(BATCH_SIZE, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)](
            k,
            v,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            page_table,
            n_kv_heads,
            d_head,
            SEQ_BLOCK,
            max_seq_len,
            PAGE_SIZE,
            page_table.stride(0),
            GENERATE_ONLY=False,
        ),
    )
    softmax_scale = 1.0 / math.sqrt(d_head)
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_paged[grid](
        q,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        cache_loc,
        input_pos,
        page_table,
        softmax_scale,
        out,
        n_heads,
        n_kv_heads,
        d_head,
        SEQ_BLOCK,
        max_seq_len,
        PAGE_SIZE,
        page_table.stride(0),
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_mha_with_paged_cache", mutates_args=())
def fused_mha_with_paged_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused MHA with paged cache that takes raw input from q, k, v GEMMs.

    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    #    Assuming that context seq_len always > 0.
    b, s, d = q.shape
    head_dim = k_cache.shape[-1]

    # reshapes with num_heads and head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)
    q = q.view(*bs_view, q.shape[2] // head_dim, head_dim)
    k = k.view(*bs_view, k.shape[2] // head_dim, head_dim)
    v = v.view(*bs_view, v.shape[2] // head_dim, head_dim)

    # rope embedding for generate-only or mixed
    if freqs_cis is not None:
        if s == 1:
            rope_args = (freqs_cis, input_pos, "bsnd")
            fn_rope = torch.ops.rope.apply_rope_with_input_pos
        else:
            rope_args = (freqs_cis, input_pos, seq_len, seq_start)
            fn_rope = torch.ops.rope.apply_rope_on_flattened_inputs
        q = fn_rope(q, *rope_args)
        k = fn_rope(k, *rope_args)

    # run attention
    y = torch.empty_like(q)
    if s == 1:
        # generate-only phase
        _paged_generate_mha(q, k, v, page_table, k_cache, v_cache, cache_loc, input_pos, y, max_seq_len)
    else:
        # mixed context + generate phase
        _paged_context_mha(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            page_table,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            y,
            max_seq_len,
        )

    return y.view(b, s, d)  # [b,s,n*h_d]


@fused_mha_with_paged_cache.register_fake
def fused_mha_with_paged_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty_like(q.contiguous())


@torch.library.custom_op("attention::prepare_fused_mha_metadata", mutates_args=())
def prepare_fused_mha_metadata(
    input_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    page_size: int,
) -> List[torch.Tensor]:
    num_seq = SequenceInfo._get_sanitized_num_sequences(input_ids, seq_len)
    seq_start = torch.zeros_like(seq_len[:num_seq])
    seq_start[1:] = torch.cumsum(seq_len[: num_seq - 1], 0)
    return (
        seq_len[:num_seq].clone(),
        input_pos[:num_seq].clone(),
        cache_loc[:num_seq].clone(),
        seq_start,
    )


@prepare_fused_mha_metadata.register_fake
def prepare_fused_mha_metadata_fake(input_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size):
    return (
        torch.empty_like(seq_len),
        torch.empty_like(input_pos),
        torch.empty_like(cache_loc),
        torch.empty_like(seq_len),
    )


@AttentionRegistry.register("TritonWithFlattenedInputs")
class TritonWithFlattenedInputs(AttentionDescriptor):
    @classmethod
    def is_paged(cls):
        """Return if the attention op is paged or not."""
        return False

    @classmethod
    def get_attention_op(cls):
        return torch.ops.attention.fused_flattened_mha_with_cache, 3

    @classmethod
    def get_prepare_metadata_op(cls):
        return torch.ops.attention.prepare_fused_mha_metadata, 4

    @classmethod
    def get_cache_initializers(cls, get_info):
        def _get_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for TritonWithFlattenedInputs"
            attention_info = get_info()
            return torch.empty(
                si.num_pages,
                si.page_size,
                attention_info.num_kv_heads,
                attention_info.head_dim,
                device=si.device,
                dtype=attention_info.cache_config.dtype or attention_info.dtype,
            )

        return {"k_cache": _get_cache, "v_cache": _get_cache}

    @classmethod
    def get_global_buffer_initializers(cls, get_info):
        attention_info = get_info()
        head_dim = attention_info.head_dim
        pos_embd_config = attention_info.pos_embd_config

        def _get_freqs_cis(si: SequenceInfo):
            if pos_embd_config.mode is None:
                return torch.empty(0, device=si.device)
            assert pos_embd_config.mode == "rope", f"Mode {pos_embd_config.mode=} not supported"
            assert pos_embd_config.rope_scale == 1.0, f"{pos_embd_config.rope_scale=} not supported"
            rope_theta = pos_embd_config.rope_theta
            return cls._precompute_freqs_cis(2 * si.max_seq_len, head_dim, rope_theta).to(si.device)

        k_full = "_".join(map(str, ["freqs_cis", *astuple(pos_embd_config)])).replace(".", "_")
        return {k_full: _get_freqs_cis}

    @staticmethod
    def _precompute_freqs_cis(seq_len: int, head_dim: int, rope_theta: Optional[float] = None) -> torch.Tensor:
        if rope_theta is None:
            rope_theta = 1e4
        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # cos and sin (real and img) are packed
        cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
        return cache.to(dtype=torch.float16)
