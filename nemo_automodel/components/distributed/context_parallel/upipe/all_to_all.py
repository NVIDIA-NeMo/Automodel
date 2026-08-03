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

"""Sequence <-> head all-to-all redistribution for Ulysses-style context parallelism.

Adapted from the Untied-Ulysses (UPipe) reference implementation, with two changes:
the process group is passed in (from a :class:`~torch.distributed.DeviceMesh`) rather
than read from a module-level singleton, and the collective is autograd-aware so the
surrounding projection / normalization / convolution chain can be differentiated
normally instead of requiring a hand-written backward.

Two layouts are involved. ``cp`` is the sequence-sharded layout every non-attention
part of the model uses (each rank owns ``S / P`` tokens and all heads); ``hp`` is the
head-sharded layout attention runs in (each rank owns the full sequence and ``H / P``
heads). :func:`cp2hp` and :func:`hp2cp` convert between them and are exact inverses.

Sequence sharding is assumed **contiguous**: rank ``r`` owns global positions
``[r * S_local, (r + 1) * S_local)``. The round-robin / DualChunkSwap layout used by
``torch.distributed`` ring attention would interleave the gathered sequence and is not
supported here.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _exchange(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Equal-split all-to-all along dim 0, on a densely packed buffer.

    Both operands must be contiguous. ``torch.empty_like`` defaults to
    ``preserve_format``, so allocating from a permuted view would hand the collective a
    strided destination and silently scatter the payload; the incoming gradient in the
    backward pass is exactly such a view.
    """
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    dist.all_to_all_single(output, tensor, group=group)
    return output


class _AllToAllSingle(torch.autograd.Function):
    """Autograd wrapper around ``torch.distributed.all_to_all_single``.

    The tensor is split into equal chunks along dim 0, one per rank. For equal-sized
    splits an all-to-all is its own inverse, so the backward pass is another all-to-all
    over the same group.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return _exchange(input_, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _exchange(grad_output, ctx.group), None


def all_to_all_single(input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Functional entry point for the autograd-aware all-to-all."""
    return _AllToAllSingle.apply(input_, group)


def cp2hp(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Redistribute from sequence-sharded to head-sharded layout.

    Args:
        x: Tensor of shape ``[B, S_local, H, D]`` holding this rank's contiguous
            sequence shard for all ``H`` heads of the current pipeline stage. ``H``
            must be divisible by the group size.
        group: Context-parallel process group.

    Returns:
        torch.Tensor: Tensor of shape ``[B, S_local * P, H // P, D]`` holding the full
        sequence for this rank's head slice. Rank ``r`` receives heads
        ``[r * H // P, (r + 1) * H // P)``.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x

    batch, seq_local, heads, head_dim = x.shape
    if heads % world_size != 0:
        raise ValueError(f"head count ({heads}) must be divisible by the CP group size ({world_size})")
    heads_local = heads // world_size

    # [B, S_local, P, H_local, Dh] -> [P, S_local, B, H_local, Dh]; chunk p goes to rank p.
    send = x.reshape(batch, seq_local, world_size, heads_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    recv = all_to_all_single(send, group)
    # Chunk j came from rank j, i.e. global positions [j * S_local, (j + 1) * S_local).
    return recv.permute(2, 0, 1, 3, 4).reshape(batch, world_size * seq_local, heads_local, head_dim)


def hp2cp(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Redistribute from head-sharded back to sequence-sharded layout.

    Exact inverse of :func:`cp2hp`.

    Args:
        x: Tensor of shape ``[B, S_full, H_local, D]``. ``S_full`` must be divisible by
            the group size.
        group: Context-parallel process group.

    Returns:
        torch.Tensor: Tensor of shape ``[B, S_full // P, H_local * P, D]``, with heads
        ordered by owning rank so the result matches the pre-:func:`cp2hp` ordering.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return x

    batch, seq_full, heads_local, head_dim = x.shape
    if seq_full % world_size != 0:
        raise ValueError(f"sequence length ({seq_full}) must be divisible by the CP group size ({world_size})")
    seq_local = seq_full // world_size

    # [B, P, S_local, H_local, Dh] -> [P, S_local, B, H_local, Dh]; chunk p goes to rank p.
    send = x.reshape(batch, world_size, seq_local, heads_local, head_dim).permute(1, 2, 0, 3, 4).contiguous()
    recv = all_to_all_single(send, group)
    # Chunk j came from rank j, i.e. heads [j * H_local, (j + 1) * H_local).
    return recv.permute(2, 1, 0, 3, 4).reshape(batch, seq_local, world_size * heads_local, head_dim)
