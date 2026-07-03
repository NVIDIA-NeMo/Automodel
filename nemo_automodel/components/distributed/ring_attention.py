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
#   Ring attention algorithm from zhuzilin/ring-flash-attention (Apache-2.0) as
#   vendored by sgl-project/SpecForge (specforge/layers/ring/). The FlashAttention
#   kernel dispatch (originally yunchang's ``select_flash_attn_impl``) is replaced
#   here with a direct call into the installed ``flash_attn`` package.

"""Differentiable ring FlashAttention over a context-parallel process group.

Each rank holds a contiguous shard ``S/cp`` of the sequence. K/V are rotated
around the cp ranks p2p (``RingComm``); each incoming K/V block is attended
against the local Q with FlashAttention and merged into the running output via
the online-softmax log-sum-exp identity (``_update_out_and_lse``). The whole
thing is a plain autograd chain (forward rotates K/V, backward rotates the K/V
grads back to their owners), so it composes with FSDP2 like any other module.

The FlashAttention kernels come from the optional ``flash_attn`` package; import
is guarded so this module never breaks import of the package when it is absent.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from nemo_automodel.shared.import_utils import safe_import

HAVE_FLASH_ATTN, _flash_attn = safe_import("flash_attn.flash_attn_interface")


def _fa_forward(q, k, v, softmax_scale, causal, dropout_p=0.0):
    """flash_attn 2.8.3 fwd: returns ``(out[B,S,H,D], lse[B,H,S])``."""
    out, lse, _s_dmask, _rng = _flash_attn._flash_attn_forward(
        q, k, v, dropout_p, softmax_scale, causal, -1, -1, 0.0, None, False
    )
    return out, lse


def _fa_backward(dout, q, k, v, out, lse, dq, dk, dv, softmax_scale, causal, deterministic=False):
    """flash_attn 2.8.3 bwd: writes ``dq/dk/dv`` in place."""
    _flash_attn._flash_attn_backward(
        dout, q, k, v, out, lse, dq, dk, dv, 0.0, softmax_scale, causal, -1, -1, 0.0, None, deterministic, None
    )


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor, lse: torch.Tensor, block_out: torch.Tensor, block_lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax merge of a new attention block into the running ``(out, lse)``.

    ``out`` is ``[B, S, H, D]`` fp32, ``lse`` is ``[B, S, H, 1]`` fp32; ``block_lse``
    arrives as the kernel's ``[B, H, S]`` and is transposed in. Numerically-stable
    sigmoid form (see zhuzilin/ring-flash-attention#34).
    """
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    out = out - torch.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - torch.nn.functional.logsigmoid(lse - block_lse)
    return out, lse


def _init_out_and_lse(block_out: torch.Tensor, block_lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = block_out.to(torch.float32)
    lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1).to(torch.float32)
    return out, lse


class RingComm:
    """One-hop p2p ring: send to ``rank+1``, receive from ``rank-1`` (cp group order)."""

    def __init__(self, process_group: Optional[dist.ProcessGroup]):
        self._pg = process_group
        self._ops: list[dist.P2POp] = []
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

    def commit(self) -> None:
        if self._reqs is not None:
            raise RuntimeError("RingComm.commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        if self._reqs is None:
            raise RuntimeError("RingComm.wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []


def ring_flash_attn_forward(
    process_group, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ring FlashAttention forward. Q/K/V are ``[B, S_local, H, D]``.

    Returns ``(out[B, S_local, H, D], lse[B, H, S_local])``. For ``causal=True`` a
    rank only attends to K/V blocks from itself and earlier ranks (``step <= rank``),
    and applies the causal mask only to its own block (``step == 0``).
    """
    comm = RingComm(process_group)
    out = None
    lse = None
    next_k = next_v = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()
        if not causal or step <= comm.rank:
            block_out, block_lse = _fa_forward(q, k, v, softmax_scale, causal=(causal and step == 0))
            if out is None:
                out, lse = _init_out_and_lse(block_out, block_lse)
            else:
                out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2).contiguous()  # [B, H, S_local]
    return out, lse


def ring_flash_attn_backward(
    process_group,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ring FlashAttention backward. Rotates K/V forward and the K/V grads back to owners."""
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq = dk = dv = None
    next_dk = next_dv = None
    next_k = next_v = None
    dq_buf = torch.empty_like(q)
    dk_buf = torch.empty_like(k)
    dv_buf = torch.empty_like(v)
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()
        if step <= kv_comm.rank or not causal:
            _fa_backward(
                dout, q, k, v, out, softmax_lse, dq_buf, dk_buf, dv_buf, softmax_scale, causal=(causal and step == 0)
            )
            if dq is None:
                dq = dq_buf.to(torch.float32)
                dk = dk_buf.to(torch.float32)
                dv = dv_buf.to(torch.float32)
            else:
                dq += dq_buf
                d_kv_comm.wait()
                dk = dk_buf + next_dk
                dv = dv_buf + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()
    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(k.dtype), next_dv.to(v.dtype)
