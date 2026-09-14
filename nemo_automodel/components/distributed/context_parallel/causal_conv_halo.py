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

"""Left-halo exchange for causal 1D convolutions under context parallelism.

A causal convolution of kernel size ``K`` needs the ``K - 1`` tokens immediately to the
left of each output position. When the sequence is sharded contiguously across CP ranks,
every rank except the first is missing that left context at the start of its shard and
would otherwise consume zero-padding there, corrupting the first ``K - 1`` outputs of
every shard.

This module exchanges the missing tokens with the left neighbor and routes the halo's
gradient back to the rank that owns those tokens. It is deliberately torch-only, with no
model or framework imports, so it can be unit tested over gloo on a CPU-only machine.

Sharding is assumed **contiguous**: rank ``r`` owns global positions
``[r * L, (r + 1) * L)``, and ``L >= K - 1`` so the halo never spans more than one
neighbor. The layout is ``[batch, sequence, channels]``.

Modelled on :class:`~nemo_automodel.components.distributed.blockdiag_cp.exchange._LeftHaloExchange`,
which does the same neighbor exchange for attention K/V straddle.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _exchange(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    send_peer: int,
    recv_peer: int,
    cp_rank: int,
) -> None:
    """One batched send/recv pair. Peers are global ranks, as ``P2POp`` expects."""
    send_op = dist.P2POp(dist.isend, send_tensor.contiguous(), send_peer, group)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_peer, group)
    ops = [send_op, recv_op] if cp_rank % 2 == 0 else [recv_op, send_op]
    for req in dist.batch_isend_irecv(ops):
        req.wait()


class _CausalConvLeftHalo(torch.autograd.Function):
    """Fetch the left neighbor's trailing ``halo_size`` tokens, differentiably.

    Every rank posts exactly one ``isend`` and one ``irecv``, including the first rank,
    which receives from the last rank around the ring and then discards the payload.
    Uniform participation is what keeps the collective from deadlocking; skipping the
    exchange on the first rank would leave its peer's matching operation unpaired.
    """

    @staticmethod
    def forward(
        ctx,
        x_local: torch.Tensor,
        group: dist.ProcessGroup,
        halo_size: int,
        cp_rank: int,
        prev_peer: int,
        next_peer: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.halo_size = halo_size
        ctx.cp_rank = cp_rank
        ctx.prev_peer = prev_peer
        ctx.next_peer = next_peer
        ctx.is_first = cp_rank == 0
        ctx.shape = tuple(x_local.shape)

        batch, seq_local, channels = x_local.shape
        recv_buf = torch.empty(batch, halo_size, channels, dtype=x_local.dtype, device=x_local.device)
        _exchange(
            x_local[:, seq_local - halo_size :, :],
            recv_buf,
            group=group,
            send_peer=next_peer,
            recv_peer=prev_peer,
            cp_rank=cp_rank,
        )
        if ctx.is_first:
            # True start of the sequence: the convolution's own zero-padding is correct
            # here, so the wrapped-around payload from the last rank is discarded.
            recv_buf.zero_()
        return recv_buf

    @staticmethod
    def backward(ctx, grad_halo: torch.Tensor):
        batch, seq_local, channels = ctx.shape
        halo_size = ctx.halo_size

        send_grad = grad_halo.contiguous()
        if ctx.is_first:
            # The first rank's halo is a constant, so it must contribute nothing to the
            # last rank's tail. It still sends (zeros) to keep participation uniform.
            send_grad = torch.zeros_like(send_grad)

        recv_grad = torch.empty(batch, halo_size, channels, dtype=grad_halo.dtype, device=grad_halo.device)
        _exchange(
            send_grad,
            recv_grad,
            group=ctx.group,
            send_peer=ctx.prev_peer,
            recv_peer=ctx.next_peer,
            cp_rank=ctx.cp_rank,
        )

        grad_x = grad_halo.new_zeros(batch, seq_local, channels)
        grad_x[:, seq_local - halo_size :, :] = recv_grad
        return grad_x, None, None, None, None, None


def causal_conv_left_halo(x_local: torch.Tensor, group: dist.ProcessGroup, halo_size: int) -> torch.Tensor:
    """Return the ``halo_size`` tokens preceding this rank's shard.

    Args:
        x_local: This rank's contiguous shard, ``[batch, seq_local, channels]``.
        group: Context-parallel process group.
        halo_size: Number of left-context tokens, ``kernel_size - 1``.

    Returns:
        torch.Tensor: ``[batch, halo_size, channels]``, the left neighbor's trailing
        tokens, or zeros on the first rank.

    Raises:
        ValueError: If the local shard is shorter than the halo, which would require
            reaching past the immediate neighbor.
    """
    world_size = dist.get_world_size(group)
    batch, seq_local, channels = x_local.shape
    if world_size == 1:
        return x_local.new_zeros(batch, halo_size, channels)
    if seq_local < halo_size:
        raise ValueError(
            f"local sequence shard ({seq_local}) is shorter than the convolution halo ({halo_size}); "
            "the halo would span more than one neighbor. Use a longer sequence or a smaller CP size."
        )

    cp_rank = dist.get_rank(group)
    ranks = dist.get_process_group_ranks(group)
    prev_peer = ranks[(cp_rank - 1) % world_size]
    next_peer = ranks[(cp_rank + 1) % world_size]
    return _CausalConvLeftHalo.apply(x_local, group, halo_size, cp_rank, prev_peer, next_peer)


def prepend_left_halo(x_local: torch.Tensor, group: dist.ProcessGroup, halo_size: int) -> torch.Tensor:
    """Prefix this rank's shard with its left context.

    The caller runs its causal convolution on the returned tensor and drops the first
    ``halo_size`` outputs, which reproduces the single-device result exactly.

    Args:
        x_local: This rank's contiguous shard, ``[batch, seq_local, channels]``.
        group: Context-parallel process group.
        halo_size: Number of left-context tokens, ``kernel_size - 1``.

    Returns:
        torch.Tensor: ``[batch, halo_size + seq_local, channels]``.
    """
    if halo_size == 0:
        return x_local
    return torch.cat([causal_conv_left_halo(x_local, group, halo_size), x_local], dim=1)
