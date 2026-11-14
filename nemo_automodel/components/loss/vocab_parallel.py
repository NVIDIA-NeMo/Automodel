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

"""Loss reductions over vocabulary-sharded logits that never gather the vocabulary.

Tensor-parallel LM heads (``ColwiseParallel(output_layouts=Shard(-1),
use_local_output=False)``) return logits as a DTensor sharded on the vocabulary
axis: every rank of the TP group holds the same tokens and one slice of the
vocabulary. These helpers reduce such logits where they are. Each rank works on
its local slice and combines per-token statistics with all-reduces, so no rank
allocates ``[tokens, vocab]`` logits.

Gradient convention: an aggregate (a log-sum-exp, a vocabulary sum, a gathered
logit) is an all-reduce whose backward sums the gradient over the group. A loss
built from aggregates has the same value on every rank, and every rank runs
backward on it, so :func:`replicated` divides that loss's gradient by the group
size. The backward sums then count each rank's share exactly once, including
local terms that themselves depend on an aggregate. The per-token results of
:func:`cross_entropy` are already marked with :func:`replicated`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard


@dataclass(frozen=True)
class VocabShard:
    """One rank's slice of logits along the vocabulary axis.

    Attributes:
        local: Tensor of shape [..., local_vocab] holding vocabulary ids
            ``[start, start + local_vocab)``; a DTensor's local tensor keeps the
            DTensor's gradient path.
        group: Process group the vocabulary is sharded over, or None when
            ``local`` holds the whole vocabulary.
        start: First vocabulary id in ``local``.
        vocab: Global vocabulary size.
    """

    local: torch.Tensor
    group: dist.ProcessGroup | None
    start: int
    vocab: int

    def with_local(self, local: torch.Tensor) -> VocabShard:
        """Return the same vocabulary slice holding ``local``.

        Args:
            local: Tensor of shape [..., local_vocab], e.g. a token subset, a
                reshape, or a dtype cast of ``self.local``.

        Returns:
            A shard with the same group and vocabulary range.
        """
        return dataclasses.replace(self, local=local)


def vocab_shard(logits: torch.Tensor) -> VocabShard:
    """Return this rank's vocabulary slice of ``logits`` without communication.

    Args:
        logits: Tensor of global shape [..., vocab]. A DTensor may be sharded on
            its last axis (``Shard(-1)``) over one mesh dimension and must be
            replicated over any others. Shards follow ``torch.chunk`` semantics.

    Returns:
        The local slice. Regular tensors, fully replicated DTensors, and
        DTensors on a one-rank shard dimension hold the whole vocabulary and
        have no group.

    Raises:
        ValueError: If a DTensor has any other placement, or its local shard
            does not match the ``torch.chunk`` layout.
    """
    vocab = logits.shape[-1]
    if not isinstance(logits, DTensor):
        return VocabShard(logits, None, 0, vocab)
    vocab_dim = logits.ndim - 1
    shard_dims = []
    for mesh_dim, placement in enumerate(logits.placements):
        if type(placement) is Shard and placement.dim in (-1, vocab_dim):
            shard_dims.append(mesh_dim)
        elif type(placement) is not Replicate:
            shard_dims = None
            break
    if shard_dims is None or len(shard_dims) > 1:
        raise ValueError(
            "Expected logits sharded on the vocabulary axis (Shard(-1)) over at most one mesh dimension and "
            f"replicated over the others, got placements {tuple(logits.placements)}."
        )
    local = logits.to_local()
    mesh = logits.device_mesh
    if not shard_dims or mesh.size(shard_dims[0]) == 1:
        return VocabShard(local, None, 0, vocab)
    mesh_dim = shard_dims[0]
    rows_per_rank = -(-vocab // mesh.size(mesh_dim))
    start = min(mesh.get_local_rank(mesh_dim) * rows_per_rank, vocab)
    if local.shape[-1] != min(rows_per_rank, vocab - start):
        raise ValueError(
            f"Vocabulary shard holds {local.shape[-1]} ids; Shard(-1) of {vocab} ids over "
            f"{mesh.size(mesh_dim)} ranks expects {min(rows_per_rank, vocab - start)} starting at {start}."
        )
    return VocabShard(local, mesh.get_group(mesh_dim), start, vocab)


class _ReplicatedLoss(torch.autograd.Function):
    """Identity in forward; divides the gradient by the group size in backward."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group_size: int) -> torch.Tensor:
        """Copy a value that every rank of a group computes identically.

        Args:
            ctx: Autograd context.
            value: Tensor of any shape, equal on every rank of the group.
            group_size: Number of ranks in the group.

        Returns:
            A copy of ``value``; the input is not aliased.
        """
        ctx.group_size = group_size
        return value.clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Hand each rank an equal share of the gradient.

        Args:
            ctx: Autograd context.
            grad: Tensor shaped like the forward output.

        Returns:
            ``grad / group_size`` and None for the group size.
        """
        return grad / ctx.group_size, None


def replicated(value: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Mark a loss that every rank of ``group`` computes from the same aggregates.

    Args:
        value: Tensor of any shape with the same value on every rank.
        group: Vocabulary-sharding group, or None for an unsharded vocabulary.

    Returns:
        ``value`` unchanged, or a copy whose backward divides the gradient by
        the group size.
    """
    if group is None:
        return value
    return _ReplicatedLoss.apply(value, dist.get_world_size(group))


def sum_over_group(value: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Sum per-rank partial results into an aggregate.

    Args:
        value: Tensor of any shape with this rank's partial result, e.g. a sum
            over its vocabulary slice.
        group: Vocabulary-sharding group, or None for an unsharded vocabulary.

    Returns:
        Tensor of the same shape with the sum over the group, the same on every
        rank. Its backward sums the gradient over the group, so every rank's
        use of the aggregate reaches every partial result.
    """
    if group is None:
        return value
    return dist_nn.all_reduce(value, op=dist.ReduceOp.SUM, group=group)


def logsumexp(local: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Log-sum-exp over the whole vocabulary.

    Args:
        local: Tensor of shape [..., local_vocab] holding this rank's logits.
        group: Vocabulary-sharding group, or None for an unsharded vocabulary.

    Returns:
        Tensor of shape [...], the same on every rank, in ``local.dtype``.
    """
    if group is None:
        return torch.logsumexp(local, dim=-1)
    if local.shape[-1]:
        shift = local.detach().amax(dim=-1)
    else:
        shift = local.new_full(local.shape[:-1], float("-inf"))
    # The log-sum-exp does not depend on the shift, so it carries no gradient.
    dist.all_reduce(shift, op=dist.ReduceOp.MAX, group=group)
    total = sum_over_group(torch.exp(local - shift.unsqueeze(-1)).sum(dim=-1), group)
    return shift + torch.log(total)


def softmax(local: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """FP32 probabilities of this rank's vocabulary slice.

    Args:
        local: Tensor of shape [..., local_vocab] holding this rank's logits.
        group: Vocabulary-sharding group, or None for an unsharded vocabulary.

    Returns:
        FP32 tensor of shape [..., local_vocab], normalized over the whole
        vocabulary.
    """
    if group is None:
        return F.softmax(local, dim=-1, dtype=torch.float32)
    local = local.float()
    return torch.exp(local - logsumexp(local, group).unsqueeze(-1))


def vocab_sum(values: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Sum over the whole vocabulary.

    Args:
        values: Tensor of shape [..., local_vocab] with this rank's terms.
        group: Vocabulary-sharding group, or None for an unsharded vocabulary.

    Returns:
        Tensor of shape [...], the same on every rank.
    """
    return sum_over_group(values.sum(dim=-1), group)


def gather(local: torch.Tensor, index: torch.Tensor, *, start: int, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Pick one vocabulary entry per position.

    Args:
        local: Tensor of shape [..., local_vocab] holding vocabulary ids
            ``[start, start + local_vocab)``.
        index: Integer tensor of shape [...] with global vocabulary ids in
            ``[0, vocab)``.
        start: First vocabulary id in ``local``.
        group: Vocabulary-sharding group, or None for an unsharded vocabulary.

    Returns:
        Tensor of shape [...] with ``logits[..., index]``, the same on every rank.
    """
    if group is None:
        return local.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    if not local.shape[-1]:
        # Stay connected to the graph so this rank enters the backward all-reduce too.
        return sum_over_group(local.sum(dim=-1), group)
    offset = index - start
    owned = (offset >= 0) & (offset < local.shape[-1])
    picked = local.gather(-1, offset.clamp(0, local.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    return sum_over_group(torch.where(owned, picked, torch.zeros_like(picked)), group)


def argmax(shard: VocabShard) -> torch.Tensor:
    """Index of the largest logit over the whole vocabulary.

    Args:
        shard: Logits of global shape [..., vocab] as this rank's slice.

    Returns:
        Int64 tensor of shape [...], the same on every rank. Ties resolve to the
        first maximal id, as in ``torch.argmax``. Not differentiable.
    """
    local = shard.local.detach()
    if shard.group is None:
        return local.argmax(dim=-1)
    if local.shape[-1]:
        value, index = local.float().max(dim=-1)
        index = index + shard.start
    else:
        value = local.new_full(local.shape[:-1], float("-inf"), dtype=torch.float32)
        index = torch.full(local.shape[:-1], shard.vocab, dtype=torch.long, device=local.device)
    best = value.clone()
    dist.all_reduce(best, op=dist.ReduceOp.MAX, group=shard.group)
    first = torch.where(value == best, index, torch.full_like(index, shard.vocab))
    dist.all_reduce(first, op=dist.ReduceOp.MIN, group=shard.group)
    return first


def cross_entropy(shard: VocabShard, target: torch.Tensor, *, ignore_index: int) -> torch.Tensor:
    """Per-token cross entropy without gathering the vocabulary.

    Args:
        shard: Logits of global shape [..., vocab] as this rank's slice.
        target: Integer tensor of shape [...] with vocabulary ids, or
            ``ignore_index`` for unsupervised positions.
        ignore_index: Target value excluded from the loss.

    Returns:
        Tensor of shape [...] in ``shard.local.dtype``, zero at ignored
        positions and the same on every rank. A sharded vocabulary is reduced in
        FP32; an unsharded one uses ``F.cross_entropy`` in the input dtype.
    """
    if shard.group is None:
        return F.cross_entropy(
            shard.local.reshape(-1, shard.local.shape[-1]),
            target.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(target.shape)
    local = shard.local.float()
    valid = target != ignore_index
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    loss = logsumexp(local, shard.group) - gather(local, safe_target, start=shard.start, group=shard.group)
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    return replicated(loss, shard.group).to(shard.local.dtype)
