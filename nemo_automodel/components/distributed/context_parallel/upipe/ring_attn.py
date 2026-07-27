# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# Adapted from the Untied Ulysses reference implementation
# (https://github.com/togethercomputer/Untied-Ulysses), Apache-2.0, which in
# turn derives from zhuzilin/ring-flash-attention.

"""FlashAttention backends and zigzag ring attention for UPipe.

UPipe calls into this module once per pipeline stage, after the Ulysses
all-to-all has given each rank a small set of heads over a longer slice of the
sequence. When the ring group is trivial (the pure-Ulysses configuration) the
ring machinery is bypassed entirely and FlashAttention is called directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import (
        _flash_attn_backward as _flash_attn_backward_fa2,
    )
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward as _flash_attn_forward_fa2,
    )

    HAS_FA2 = True
except ImportError:  # pragma: no cover - depends on the installed wheel
    _flash_attn_forward_fa2 = None
    _flash_attn_backward_fa2 = None
    HAS_FA2 = False

try:
    from flash_attn_interface import (
        _flash_attn_backward as _flash_attn_backward_fa3,
    )
    from flash_attn_interface import (
        _flash_attn_forward as _flash_attn_forward_fa3,
    )

    HAS_FA3 = True
except ImportError:  # pragma: no cover - depends on the installed wheel
    _flash_attn_forward_fa3 = None
    _flash_attn_backward_fa3 = None
    HAS_FA3 = False


def flash_attn_forward_fa2(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
):
    """FlashAttention-2 forward with a backend-independent signature."""
    return _flash_attn_forward_fa2(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_softmax,
    )


def flash_attn_forward_fa3(
    q,
    k,
    v,
    dropout_p,
    softmax_scale,
    causal=False,
    window_size_left=-1,
    window_size_right=-1,
    softcap=0.0,
    alibi_slopes=None,
    return_softmax=False,
):
    """FlashAttention-3 forward. FA3 supports neither dropout nor ALiBi."""
    result = _flash_attn_forward_fa3(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
    )
    # FA3 returns (out, lse, S_dmask, rng_state) or just (out, lse).
    return result[0], result[1]


def flash_attn_backward_fa2(
    dout,
    q,
    k,
    v,
    out,
    lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    """FlashAttention-2 backward with a backend-independent signature."""
    return _flash_attn_backward_fa2(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
    )


def flash_attn_backward_fa3(
    dout,
    q,
    k,
    v,
    out,
    lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    """FlashAttention-3 backward. FA3 supports neither dropout nor ALiBi."""
    _flash_attn_backward_fa3(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq=dq,
        dk=dk,
        dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
        deterministic=deterministic,
    )


def select_flash_attn_impl(attn_type: str, stage: str = "fwd-only"):
    """Pick the FlashAttention entry point for a backend and direction.

    Args:
        attn_type: ``"fa2"`` or ``"fa3"``.
        stage: ``"fwd-only"`` or ``"bwd-only"``.

    Returns:
        The matching forward or backward callable.

    Raises:
        ValueError: If ``attn_type`` is unknown.
        RuntimeError: If the requested backend is not installed.
    """
    if attn_type == "fa2":
        if not HAS_FA2:
            raise RuntimeError("attn='upipe' requires flash-attn (FlashAttention 2) to be installed.")
        return flash_attn_forward_fa2 if stage == "fwd-only" else flash_attn_backward_fa2
    if attn_type == "fa3":
        if not HAS_FA3:
            raise RuntimeError(
                "attn_type='fa3' requires the FlashAttention 3 'flash_attn_interface' module (Hopper build)."
            )
        return flash_attn_forward_fa3 if stage == "fwd-only" else flash_attn_backward_fa3
    raise ValueError(f"Unknown attn_type: {attn_type}. Use 'fa2' or 'fa3'.")


class RingComm:
    """Ring P2P helper that rotates tensors one hop per step."""

    def __init__(self, process_group: dist.ProcessGroup, recv_buffer: Optional[torch.Tensor] = None):
        self._process_group = process_group
        self._ops = []
        self._reqs = None
        self.recv_buffer = recv_buffer

        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Queue an async send/recv pair; the result is valid after :meth:`wait`."""
        if recv_tensor is None:
            res = self.recv_buffer if self.recv_buffer is not None else torch.empty_like(to_send)
        else:
            res = recv_tensor

        self._ops.append(dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group))
        self._ops.append(dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group))

        return res

    def commit(self):
        """Launch the queued P2P operations."""
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        """Block until every queued P2P operation has completed."""
        if self._reqs is None:
            raise RuntimeError("wait called before commit")

        for req in self._reqs:
            req.wait()

        # barrier is needed to ensure torch mem allocator properly clears the unused tensors
        dist.barrier(group=self._process_group)

        self._reqs.clear()
        self._reqs = None
        self._ops.clear()
        self._ops = []


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge one attention block into the accumulator via a sigmoid-form LSE update.

    Uses ``softmax(a, b) = sigmoid(a-b) * a + sigmoid(b-a) * b`` to avoid the
    explicit ``exp()`` that can overflow.
    """
    block_out = block_out.to(torch.float32)
    if block_lse.ndim == 3:
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_: Optional[Tuple] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate an attention block, optionally into a sub-slice of the output."""
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)

    return out, lse


def zigzag_ring_flash_attn_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zigzag ring attention forward; queries rotate while K/V stay local.

    Args:
        process_group: Ring process group.
        q: Queries ``[batch, seq, heads, head_dim]``.
        k: Keys, same layout as ``q``.
        v: Values, same layout as ``q``.
        softmax_scale: Softmax scaling factor.
        dropout_p: Dropout probability.
        causal: Must be True; zigzag scheduling only makes sense for causal masks.
        window_size: Sliding-window bounds.
        softcap: Attention logit soft cap.
        alibi_slopes: Optional ALiBi slopes.
        attn_type: ``"fa2"`` or ``"fa3"``.

    Returns:
        ``(out, lse)`` with ``out`` shaped like ``q`` and ``lse`` ``[batch, heads, seq]``.
    """
    assert causal, "Zigzag ring attention requires causal=True"

    comm = RingComm(process_group, recv_buffer=torch.empty_like(q))

    block_seq_len = q.shape[1] // 2
    k_first_half = k[:, :block_seq_len]
    v_first_half = v[:, :block_seq_len]

    out = None
    lse = None
    next_q = None

    def forward(q_in, k_in, v_in, is_causal):
        fn = select_flash_attn_impl(attn_type, stage="fwd-only")
        result = fn(
            q_in,
            k_in,
            v_in,
            dropout_p,
            softmax_scale,
            causal=is_causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=(dropout_p > 0),
        )
        return result[0], result[1]

    for step in range(comm.world_size):
        if step + 1 < comm.world_size:
            next_q = comm.send_recv(q)
            comm.commit()

        if step == 0:
            # Local Q against local K/V: the diagonal block, fully causal.
            block_out, block_lse = forward(q, k, v, is_causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            # Q from earlier in the global sequence: only its second half sees our K/V.
            q_second_half = q[:, block_seq_len:]
            block_out, block_lse = forward(q_second_half, k, v, is_causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )
        else:
            # Q from later in the global sequence: all of it sees our first K/V half.
            block_out, block_lse = forward(q, k_first_half, v_first_half, is_causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 < comm.world_size:
            comm.wait()
            q = next_q

    del comm, next_q, k_first_half, v_first_half

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)

    return out, lse


def zigzag_ring_flash_attn_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zigzag ring attention backward; K/V rotate while gradients accumulate.

    Args:
        process_group: Ring process group.
        dout: Gradient of the attention output.
        q: Queries from the forward pass.
        k: Keys from the forward pass.
        v: Values from the forward pass.
        out: Attention output from the forward pass.
        softmax_lse: Log-sum-exp from the forward pass, ``[batch, heads, seq]``.
        softmax_scale: Softmax scaling factor.
        dropout_p: Dropout probability.
        causal: Must be True.
        window_size: Sliding-window bounds.
        softcap: Attention logit soft cap.
        alibi_slopes: Optional ALiBi slopes.
        deterministic: Use deterministic FlashAttention backward kernels.
        attn_type: ``"fa2"`` or ``"fa3"``.

    Returns:
        ``(dq, dk, dv)``.
    """
    assert causal, "Zigzag ring attention requires causal=True"

    kv_comm = RingComm(process_group)
    dkv_comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    dout_second = dout[:, block_seq_len:]
    q_second = q[:, block_seq_len:]
    out_second = out[:, block_seq_len:]
    lse_second = softmax_lse[:, :, block_seq_len:].contiguous()

    dq_buffer = torch.empty_like(q)
    dk_buffer = torch.empty_like(k)
    dv_buffer = torch.empty_like(v)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    def backward(dout_in, q_in, k_in, v_in, out_in, lse_in, is_causal):
        seqlen_q = q_in.shape[1]
        seqlen_kv = k_in.shape[1]
        fn = select_flash_attn_impl(attn_type, stage="bwd-only")

        fn(
            dout_in,
            q_in,
            k_in,
            v_in,
            out_in,
            lse_in,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            rng_state=None,
        )

    for step in range(kv_comm.world_size):
        if step + 1 < kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()

        if step == 0:
            backward(
                dout.contiguous(),
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                out.contiguous(),
                softmax_lse.contiguous(),
                is_causal=True,
            )

            if kv_comm.world_size == 1:
                return dq_buffer, dk_buffer, dv_buffer

            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                # K/V from earlier in the sequence: only their first half is visible.
                k_first = k[:, :block_seq_len]
                v_first = v[:, :block_seq_len]
                backward(dout, q, k_first, v_first, out, softmax_lse, is_causal=False)
                dq += dq_buffer
            else:
                # K/V from later in the sequence: only our second Q half sees them.
                backward(dout_second, q_second, k, v, out_second, lse_second, is_causal=False)
                dq[:, block_seq_len:] += dq_buffer[:, :block_seq_len]

            dkv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[:, :block_seq_len] += dk_buffer[:, :block_seq_len]
                dv[:, :block_seq_len] += dv_buffer[:, :block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 < kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = dkv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = dkv_comm.send_recv(dv, dv_comm_buffer)
        dkv_comm.commit()

    dkv_comm.wait()

    orig_dtype = q.dtype

    return (
        dq.to(orig_dtype).detach(),
        next_dk.to(orig_dtype).detach(),
        next_dv.to(orig_dtype).detach(),
    )


def attn_forward(
    ring_group: Optional[dist.ProcessGroup],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention forward for one UPipe stage, with or without a ring dimension.

    A trivial ring group (the pure-Ulysses configuration) skips the ring
    scheduling and its float32 accumulator entirely and calls FlashAttention
    once.

    Args:
        ring_group: Ring process group, or None for pure Ulysses.
        q: Queries ``[batch, seq, heads, head_dim]``.
        k: Keys, same layout as ``q``.
        v: Values, same layout as ``q``.
        softmax_scale: Softmax scaling factor.
        dropout_p: Dropout probability.
        causal: Whether to apply a causal mask.
        attn_type: ``"fa2"`` or ``"fa3"``.

    Returns:
        ``(out, lse)`` with ``lse`` shaped ``[batch, heads, seq]``.
    """
    if ring_group is None or dist.get_world_size(ring_group) == 1:
        fn = select_flash_attn_impl(attn_type, stage="fwd-only")
        result = fn(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        return result[0], result[1]

    return zigzag_ring_flash_attn_forward(
        ring_group,
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        attn_type=attn_type,
    )


def attn_backward(
    ring_group: Optional[dist.ProcessGroup],
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale: float,
    dropout_p: float = 0.0,
    causal: bool = True,
    deterministic: bool = False,
    attn_type: str = "fa2",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention backward for one UPipe stage, mirroring :func:`attn_forward`.

    Args:
        ring_group: Ring process group, or None for pure Ulysses.
        dout: Gradient of the attention output.
        q: Queries from the forward pass.
        k: Keys from the forward pass.
        v: Values from the forward pass.
        out: Attention output from the forward pass.
        softmax_lse: Log-sum-exp from the forward pass, ``[batch, heads, seq]``.
        softmax_scale: Softmax scaling factor.
        dropout_p: Dropout probability.
        causal: Whether a causal mask was applied.
        deterministic: Use deterministic FlashAttention backward kernels.
        attn_type: ``"fa2"`` or ``"fa3"``.

    Returns:
        ``(dq, dk, dv)``.
    """
    if ring_group is None or dist.get_world_size(ring_group) == 1:
        fn = select_flash_attn_impl(attn_type, stage="bwd-only")
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        fn(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=deterministic,
            rng_state=None,
        )
        return dq, dk, dv

    return zigzag_ring_flash_attn_backward(
        ring_group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        deterministic=deterministic,
        attn_type=attn_type,
    )
