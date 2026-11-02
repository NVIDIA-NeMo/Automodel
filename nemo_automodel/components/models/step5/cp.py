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

"""Memory-bounded sequence transport for Step-5 context parallelism."""

import torch
import torch.distributed as dist


class _GatherSequence(torch.autograd.Function):
    """Gather directly into one allocation; reduce-scatter the backward gradient."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        local = x.transpose(0, 1).contiguous()
        output = x.new_empty((local.shape[0] * dist.get_world_size(group), *local.shape[1:]))
        dist.all_gather_into_tensor(output, local, group=group)
        ctx.group = group
        ctx.local_shape = local.shape
        return output.transpose(0, 1)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        gradient = grad_output.transpose(0, 1).contiguous()
        local = gradient.new_empty(ctx.local_shape)
        dist.reduce_scatter_tensor(local, gradient, group=ctx.group)
        return local.transpose(0, 1), None


def gather_sequence(x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Gather contiguous sequence shards with their autograd edges.

    Args:
        x: Tensor of shape [batch, local_sequence, ...], contiguous by CP rank.
        group: CP group, or None for a single rank.

    Returns:
        Tensor of shape [batch, global_sequence, ...]. It aliases x without CP.
    """
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _GatherSequence.apply(x, group)


def gather_metadata(x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Gather non-differentiable positions, masks or indexer projections.

    Args:
        x: Tensor of shape [batch, local_sequence, ...] without gradients.
        group: CP group, or None for a single rank.

    Returns:
        Tensor of shape [batch, global_sequence, ...]. It aliases x without CP.
    """
    with torch.no_grad():
        return gather_sequence(x, group)


class _GatherHalo(torch.autograd.Function):
    """Exchange only the preceding window and return its gradients to its owners."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
        halo_size: int,
        left_padding: int,
    ) -> torch.Tensor:
        rank, world, length = dist.get_rank(group), dist.get_world_size(group), x.shape[1]
        hops = min((halo_size + length - 1) // length, world - 1)
        received, sends, operations = [], [], []
        for distance in range(1, hops + 1):
            width = min(length, halo_size - (distance - 1) * length)
            if rank + distance < world:
                outgoing = x[:, -width:].contiguous()
                sends.append(outgoing)
                operations.append(dist.P2POp(dist.isend, outgoing, dist.get_global_rank(group, rank + distance), group))
            if rank >= distance:
                incoming = x.new_empty((x.shape[0], width, *x.shape[2:]))
                received.append(incoming)
                operations.append(dist.P2POp(dist.irecv, incoming, dist.get_global_rank(group, rank - distance), group))
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        ctx.group, ctx.halo_size, ctx.length = group, halo_size, length
        ctx.left_padding = left_padding
        pieces = list(reversed(received))
        if left_padding:
            pieces.insert(0, x.new_zeros((x.shape[0], left_padding, *x.shape[2:])))
        return torch.cat((*pieces, x), dim=1) if pieces else x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        group, halo_size, length = ctx.group, ctx.halo_size, ctx.length
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        prefix = min(rank * length, halo_size) + ctx.left_padding
        local = grad_output[:, prefix:].clone(memory_format=torch.contiguous_format)
        hops = min((halo_size + length - 1) // length, world - 1)
        received, sends, operations = [], [], []
        for distance in range(1, hops + 1):
            width = min(length, halo_size - (distance - 1) * length)
            if rank >= distance:
                end = prefix - (distance - 1) * length
                outgoing = grad_output[:, end - width : end].contiguous()
                sends.append(outgoing)
                operations.append(dist.P2POp(dist.isend, outgoing, dist.get_global_rank(group, rank - distance), group))
            if rank + distance < world:
                incoming = local.new_empty((local.shape[0], width, *local.shape[2:]))
                received.append(incoming)
                operations.append(dist.P2POp(dist.irecv, incoming, dist.get_global_rank(group, rank + distance), group))
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        for incoming in received:
            local[:, -incoming.shape[1] :].add_(incoming)
        return local, None, None, None


def gather_halo(
    x: torch.Tensor,
    group: dist.ProcessGroup | None,
    halo_size: int,
    *,
    left_padding: int = 0,
) -> torch.Tensor:
    """Prepend only the preceding causal window using point-to-point transport.

    Args:
        x: Tensor [batch, local_sequence, ...], with equal contiguous CP shards.
        group: CP process group, or None for a single rank.
        halo_size: Maximum number of preceding tokens, normally window - 1.
        left_padding: Local zero positions before the received halo, for aligning
            an attention tile with global key coordinates. No padding is sent.

    Returns:
        Tensor [batch, left_padding + min(rank * local_sequence, halo_size) + local_sequence, ...].
        Gradients from remote query shards are added back to each source shard.
        Windows spanning several ranks are supported without relaying full shards.
    """
    if halo_size < 0 or left_padding < 0:
        raise ValueError("halo_size and left_padding must be non-negative")
    if group is None or dist.get_world_size(group) == 1 or halo_size == 0:
        if left_padding:
            return torch.cat((x.new_zeros((x.shape[0], left_padding, *x.shape[2:])), x), dim=1)
        return x
    return _GatherHalo.apply(x, group, halo_size, left_padding)
