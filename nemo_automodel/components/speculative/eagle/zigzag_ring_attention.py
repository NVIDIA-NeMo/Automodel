# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# Upstream attribution:
#   Zig-zag ring attention vendored from modelscope/ms-swift
#   (swift/sequence_parallel/zigzag_ring_attn.py, Apache-2.0), which in turn
#   borrows from zhuzilin/ring-flash-attention. Adapted here: the NPU path is
#   dropped, and the per-block flash calls target the installed ``flash_attn``
#   varlen kernels directly (ms-swift's signature-introspection helper does not
#   work against flash_attn 2.8.3's torch-library-wrapped ops).

"""Load-balanced (zig-zag) ring FlashAttention over a context-parallel group.

The sequence is chunked into ``2 * cp`` pieces; rank ``r`` owns chunk ``r`` (early)
and chunk ``2*cp-1-r`` (late). Pairing an early + late chunk balances the causal
triangle so every rank does equal work -- unlike a contiguous shard where the last
rank does ~2x. Varlen (``cu_seqlens``) throughout, matching the FlashAttention
varlen kernels; ``causal=True`` only (zig-zag is meaningless otherwise).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import safe_import

HAVE_FLASH_ATTN, _fa = safe_import("flash_attn.flash_attn_interface")


class RingComm:
    """P2P ring: send to ``rank+1``, receive from ``rank-1`` (cp group order)."""

    def __init__(self, process_group):
        self._pg = process_group
        self._ops: list = []
        self._reqs = None
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        send = (self.rank + 1) % self.world_size
        recv = (self.rank - 1) % self.world_size
        if process_group is not None:
            send = dist.get_global_rank(process_group, send)
            recv = dist.get_global_rank(process_group, recv)
        self.send_rank, self.recv_rank = send, recv

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        self._ops.append(dist.P2POp(dist.isend, to_send.contiguous(), self.send_rank, group=self._pg))
        self._ops.append(dist.P2POp(dist.irecv, res, self.recv_rank, group=self._pg))
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(self, k, v, k_buffer=None, v_buffer=None):
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


def get_half_index(cu_seqlens, *, front: bool):
    """Index selecting the front (early) or back (late) half of each varlen document."""
    if len(cu_seqlens) == 2:
        return slice(None, cu_seqlens[-1] // 2) if front else slice(cu_seqlens[-1] // 2, None)
    index = torch.zeros((cu_seqlens[-1].item(),), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


@torch.jit.script
def get_half_lse(lse, cu_seqlens, *, front: bool):
    """Front/back half of a ``[num_heads, seqlen]`` lse, per varlen document."""
    new_lse = torch.empty((lse.shape[0], lse.shape[1] // 2), dtype=lse.dtype, device=lse.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        if front:
            end -= (end - start) // 2
        else:
            start += (end - start) // 2
        new_lse[:, new_start:new_end] = lse[:, start:end]
    return new_lse


def update_out_and_lse(out, lse, block_out, block_lse):
    """Online-softmax merge; also returns ``sigmoid(block_lse - lse)`` for the backward."""
    if out is None:
        return block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(dim=-1), None
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    sig_diff = torch.sigmoid(block_lse - lse)
    out = out - sig_diff * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse, sig_diff


def squeeze_batch(*t):
    """Drop a leading batch dim of size 1 from each tensor (varlen tensors are unbatched)."""
    return tuple(sub.squeeze(0) if sub.shape[0] == 1 else sub for sub in t)


def _block_forward(q, k, v, causal, cu_seqlens, max_seqlen, block_seq_len, softmax_scale, window_size):
    """One flash varlen forward block; front/back half uses the halved cu_seqlens."""
    half_cu, half_max = cu_seqlens // 2, max_seqlen // 2
    cu_q = half_cu if q.shape[0] == block_seq_len else cu_seqlens
    max_q = half_max if q.shape[0] == block_seq_len else max_seqlen
    cu_kv = half_cu if k.shape[0] == block_seq_len else cu_seqlens
    max_kv = half_max if k.shape[0] == block_seq_len else max_seqlen
    out = _fa._flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        max_q,
        max_kv,
        0.0,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        0.0,
        None,
        False,
    )
    return out[0], out[1]  # block_out, block_lse


def _block_backward(
    dout, q, k, v, out, lse, causal, cu_seqlens, max_seqlen, block_seq_len, dq_buf, dk_buf, dv_buf, scale, det, window
):
    half_cu, half_max = cu_seqlens // 2, max_seqlen // 2
    cu_q = half_cu if q.shape[0] == block_seq_len else cu_seqlens
    max_q = half_max if q.shape[0] == block_seq_len else max_seqlen
    cu_kv = half_cu if k.shape[0] == block_seq_len else cu_seqlens
    max_kv = half_max if k.shape[0] == block_seq_len else max_seqlen
    _fa._flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq_buf[: q.shape[0]],
        dk_buf[: k.shape[0]],
        dv_buf[: k.shape[0]],
        cu_q,
        cu_kv,
        max_q,
        max_kv,
        0.0,
        scale,
        causal,
        window[0],
        window[1],
        0.0,
        None,
        det,
        None,
        False,
    )


def zigzag_ring_flash_attn_varlen_forward(
    process_group,
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    causal=True,
    window_size=(-1, -1),
):
    """Zig-zag ring forward: rotate K/V, per-block flash, online-softmax merge -> (out, lse)."""
    assert causal, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)
    q, k, v = squeeze_batch(q, k, v)
    q1 = q[half_index1]
    cu_seqlens = cu_seqlens // comm.world_size
    max_seqlen = max_seqlen // comm.world_size
    block_seq_len = q.shape[0] // 2
    out = lse = None
    next_k = next_v = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if step == 0:
            block_out, block_lse = _block_forward(
                q, k, v, True, cu_seqlens, max_seqlen, block_seq_len, softmax_scale, window_size
            )
            out, lse, _ = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            block_out, block_lse = _block_forward(
                q,
                k[half_index0],
                v[half_index0],
                False,
                cu_seqlens,
                max_seqlen,
                block_seq_len,
                softmax_scale,
                window_size,
            )
            out, lse, _ = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = _block_forward(
                q1, k, v, False, cu_seqlens, max_seqlen, block_seq_len, softmax_scale, window_size
            )
            out[half_index1], lse[half_index1], _ = update_out_and_lse(
                out[half_index1], lse[half_index1], block_out, block_lse
            )
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1)  # [num_heads, seqlen]
    return out.unsqueeze(0), lse.unsqueeze(0)


def zigzag_ring_flash_attn_varlen_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens,
    max_seqlen,
    half_index0,
    half_index1,
    softmax_scale,
    causal=True,
    window_size=(-1, -1),
    deterministic=False,
):
    """Backward via the merged-lse trick (zhuzilin): each block's flash backward is fed
    the FINAL merged out/lse, so the joint-softmax gradient (both the output and the
    normalization/lse paths) is captured without a separate merge backward. ms-swift's
    ``lse_grad`` variant drops the lse path, which skews dq/dk.
    """
    assert causal, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq = dk = dv = None
    next_dk = next_dv = None
    next_k = next_v = None
    dout, q, k, v, out, softmax_lse = squeeze_batch(dout, q, k, v, out, softmax_lse)
    q1 = q[half_index1]
    cu_seqlens = cu_seqlens // kv_comm.world_size
    max_seqlen = max_seqlen // kv_comm.world_size
    block_seq_len = q.shape[0] // 2
    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)
        if step == 0:
            _block_backward(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                True,
                cu_seqlens,
                max_seqlen,
                block_seq_len,
                dq_buffer,
                dk_buffer,
                dv_buffer,
                softmax_scale,
                deterministic,
                window_size,
            )
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                _block_backward(
                    dout,
                    q,
                    k[half_index0],
                    v[half_index0],
                    out,
                    softmax_lse,
                    False,
                    cu_seqlens,
                    max_seqlen,
                    block_seq_len,
                    dq_buffer,
                    dk_buffer,
                    dv_buffer,
                    softmax_scale,
                    deterministic,
                    window_size,
                )
                dq += dq_buffer
            else:
                _block_backward(
                    dout[half_index1],
                    q1,
                    k,
                    v,
                    out[half_index1],
                    get_half_lse(softmax_lse, cu_seqlens, front=False),
                    False,
                    cu_seqlens,
                    max_seqlen,
                    block_seq_len,
                    dq_buffer,
                    dk_buffer,
                    dv_buffer,
                    softmax_scale,
                    deterministic,
                    window_size,
                )
                dq[half_index1] += dq_buffer[:block_seq_len]
            d_kv_comm.wait()
            dk_comm_buffer = torch.empty_like(dk)
            dv_comm_buffer = torch.empty_like(dv)
            dk_comm_buffer.copy_(dk)
            dv_comm_buffer.copy_(dv)
            dk, dv = next_dk, next_dv
            if step <= kv_comm.rank:
                dk[half_index0] += dk_buffer[:block_seq_len]
                dv[half_index0] += dv_buffer[:block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        if step == 0:
            next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
        else:
            next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)
    d_kv_comm.wait()
    return dq.to(q.dtype).unsqueeze(0), next_dk.to(q.dtype).unsqueeze(0), next_dv.to(q.dtype).unsqueeze(0)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):
    """Autograd wrapper around the zig-zag ring varlen forward/backward."""

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal, window_size, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k = k.contiguous()
        v = v.contiguous()
        world = dist.get_world_size(group)
        half0 = get_half_index(cu_seqlens // world, front=True)
        half1 = get_half_index(cu_seqlens // world, front=False)
        out, lse = zigzag_ring_flash_attn_varlen_forward(
            group, q, k, v, cu_seqlens, max_seqlen, half0, half1, softmax_scale, causal, window_size
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.half0, ctx.half1 = half0, half1
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.half0,
            ctx.half1,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
        )
        return dq, dk, dv, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_func(
    q, k, v, cu_seqlens, max_seqlen, softmax_scale=None, causal=True, window_size=(-1, -1), group=None
):
    """Load-balanced causal ring FlashAttention. q/k/v are ``[1, S_local, H, D]`` (varlen packed)."""
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal, window_size, group
    )
