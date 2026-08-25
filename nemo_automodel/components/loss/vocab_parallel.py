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

"""Token statistics for dense and vocabulary-sharded logits."""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

_DENSE_TOKEN_CHUNK_SIZE = 256


def _sum_vocab_shards(values: torch.Tensor, logits: DTensor, vocab_mesh_dim: int) -> torch.Tensor:
    """Sum rank-local statistics over the vocabulary-shard mesh dimension.

    Wraps ``values`` as a ``Partial`` DTensor on ``vocab_mesh_dim`` so that
    redistributing to ``Replicate`` performs the all-reduce.
    """
    partial_placements = [Replicate() for _ in logits.placements]
    partial_placements[vocab_mesh_dim] = Partial()
    return (
        DTensor.from_local(values, logits.device_mesh, partial_placements, run_check=False)
        .redistribute(placements=[Replicate() for _ in logits.placements])
        .to_local()
    )


def _shifted_local_logits(
    logits: DTensor,
    temperature: float,
) -> tuple[torch.Tensor, int, int]:
    """Scale local vocabulary-sharded logits and subtract the global max.

    Returns a rank-local fp32 tensor of shape [..., local_vocab], this rank's
    global vocabulary offset, and the vocabulary-shard mesh dimension.
    """
    shard_error = (
        "logits must have exactly one Shard placement on the last vocabulary axis; "
        f"got placements={logits.placements}"
    )
    vocab_mesh_dim = None
    for mesh_dim, placement in enumerate(logits.placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim if placement.dim >= 0 else placement.dim + logits.ndim
            if shard_dim != logits.ndim - 1 or vocab_mesh_dim is not None:
                raise ValueError(shard_error)
            vocab_mesh_dim = mesh_dim
        elif not isinstance(placement, Replicate):
            raise ValueError(
                f"logits must be replicated on every non-vocabulary mesh dimension; got placements={logits.placements}"
            )
    if vocab_mesh_dim is None:
        raise ValueError(shard_error)

    global_vocab_size = int(logits.shape[-1])
    mesh = logits.device_mesh
    shard_count = mesh.size(vocab_mesh_dim)
    shard_rank = mesh.get_local_rank(vocab_mesh_dim)
    chunk_size = (global_vocab_size + shard_count - 1) // shard_count
    shard_offset = min(shard_rank * chunk_size, global_vocab_size)
    shard_size = min(chunk_size, global_vocab_size - shard_offset)
    local_logits = logits.to_local()
    # The shard offset above assumes DTensor's even-chunk layout; this guards
    # that assumption against a differently laid-out local shard.
    if local_logits.shape[-1] != shard_size:
        raise ValueError(
            "logits local vocabulary size does not match its DTensor metadata: "
            f"expected {shard_size}, got {local_logits.shape[-1]}"
        )

    scaled_logits = local_logits.float() / temperature
    if shard_size == 0:
        global_max = torch.full(
            scaled_logits.shape[:-1],
            -torch.inf,
            dtype=scaled_logits.dtype,
            device=scaled_logits.device,
        )
    else:
        global_max = scaled_logits.detach().amax(dim=-1)
    group = mesh.get_group(vocab_mesh_dim)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
    return scaled_logits - global_max.unsqueeze(-1), shard_offset, vocab_mesh_dim


def _vocab_parallel_log_probs(
    logits: DTensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Selected-token log probabilities without gathering the vocabulary.

    Each rank contributes its shard's exp-sum, the selected logit when the
    target falls in its shard, and a target-coverage flag; one all-reduce
    combines the three. Uncovered (out-of-range) targets yield ``NaN``.
    """
    shifted_logits, shard_offset, vocab_mesh_dim = _shifted_local_logits(logits, temperature)
    local_vocab_size = shifted_logits.shape[-1]
    local_denom = shifted_logits.exp().sum(dim=-1)
    in_local_shard = (targets >= shard_offset) & (targets < shard_offset + local_vocab_size)
    if local_vocab_size == 0:
        local_selected = torch.zeros_like(local_denom)
    else:
        local_targets = (targets - shard_offset).clamp(min=0, max=local_vocab_size - 1)
        local_selected = shifted_logits.gather(dim=-1, index=local_targets.unsqueeze(-1)).squeeze(-1)
        local_selected = torch.where(in_local_shard, local_selected, torch.zeros_like(local_selected))

    global_stats = _sum_vocab_shards(
        torch.stack((local_denom, local_selected, in_local_shard.to(local_denom.dtype)), dim=-1),
        logits,
        vocab_mesh_dim,
    )
    log_probs = global_stats[..., 1] - global_stats[..., 0].log()
    return log_probs.masked_fill(global_stats[..., 2] != 1, torch.nan)


def _vocab_parallel_entropy(
    logits: DTensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Categorical entropy without gathering the vocabulary: each rank
    contributes its shard's exp-sum and probability-weighted logit sum."""
    shifted_logits, _, vocab_mesh_dim = _shifted_local_logits(logits, temperature)
    local_weights = shifted_logits.exp()
    local_denom = local_weights.sum(dim=-1)
    local_weighted_logits = (local_weights * shifted_logits).sum(dim=-1)
    global_stats = _sum_vocab_shards(
        torch.stack((local_denom, local_weighted_logits), dim=-1),
        logits,
        vocab_mesh_dim,
    )
    return global_stats[..., 0].log() - global_stats[..., 1] / global_stats[..., 0]


def _validate_logits(logits: torch.Tensor | DTensor, temperature: float) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor or DTensor, got {type(logits).__name__}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating-point dtype, got {logits.dtype}")
    if logits.ndim == 0:
        raise ValueError("logits must have shape [..., vocab]")
    if logits.shape[-1] <= 0:
        raise ValueError(f"logits vocabulary size must be positive, got {logits.shape[-1]}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")


def token_log_probs(
    logits: torch.Tensor | DTensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute selected-token log probabilities for dense or vocabulary-sharded logits.

    Dense logits upcast to fp32 in chunks of at most 256 token rows, so the
    forward peak never holds the full vocabulary tensor in fp32 (with autograd
    on, saved activations still accumulate across chunks). Vocabulary-sharded
    DTensors use distributed reductions without gathering the vocabulary.

    Args:
        logits: Floating-point tensor with global shape [..., vocab], with
            arbitrary leading dimensions. A DTensor must have exactly one
            ``Shard`` placement on the vocabulary axis and ``Replicate`` on
            every other mesh dimension; its per-rank local shape is
            [..., local_vocab].
        targets: Rank-local int64 tensor of shape [...], matching the leading
            dimensions of ``logits`` and containing global vocabulary indices.
            For DTensor logits, targets must be replicated across the
            vocabulary-shard mesh dimension.
        temperature: Positive finite scale applied before normalization.

    Returns:
        Differentiable fp32 tensor of shape [...] containing selected-token log
        probabilities. Invalid target positions contain ``NaN``. The function
        does not mutate or gather ``logits``.

    Raises:
        TypeError: If an input has an invalid type or dtype.
        ValueError: If a shape, device, DTensor placement, vocabulary size, or
            temperature is invalid.
    """
    _validate_logits(logits, temperature)
    if isinstance(targets, DTensor):
        raise TypeError("targets must be a rank-local torch.Tensor, not a DTensor")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets must be a torch.Tensor, got {type(targets).__name__}")
    if targets.dtype != torch.long:
        raise TypeError(f"targets must have dtype torch.int64, got {targets.dtype}")
    if targets.device != logits.device:
        raise ValueError(f"targets and logits must be on the same device, got {targets.device} and {logits.device}")
    if tuple(targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            f"targets shape must match logits leading shape {tuple(logits.shape[:-1])}, got {tuple(targets.shape)}"
        )
    if isinstance(logits, DTensor):
        return _vocab_parallel_log_probs(logits, targets, temperature=temperature)

    leading_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    if flat_targets.numel() == 0:
        # Route through the input so an empty batch still produces a gradient.
        return flat_logits.float().sum(dim=-1).reshape(leading_shape)
    log_probs = torch.empty(flat_targets.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, flat_targets.numel(), _DENSE_TOKEN_CHUNK_SIZE):
        end = min(start + _DENSE_TOKEN_CHUNK_SIZE, flat_targets.numel())
        chunk = flat_logits[start:end].float()
        if temperature != 1.0:
            chunk = chunk / temperature
        chunk_targets = flat_targets[start:end]
        valid_targets = (chunk_targets >= 0) & (chunk_targets < vocab_size)
        safe_targets = chunk_targets.clamp(min=0, max=vocab_size - 1)
        selected_logits = chunk.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        chunk_log_probs = selected_logits - torch.logsumexp(chunk, dim=-1)
        log_probs[start:end] = chunk_log_probs.masked_fill(~valid_targets, torch.nan)
    return log_probs.reshape(leading_shape)


def token_entropy(
    logits: torch.Tensor | DTensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute categorical entropy for dense or vocabulary-sharded logits.

    Dense logits upcast to fp32 in chunks of at most 256 token rows, so the
    forward peak never holds the full vocabulary tensor in fp32 (with autograd
    on, saved activations still accumulate across chunks). Vocabulary-sharded
    DTensors use distributed reductions without gathering the vocabulary.

    Args:
        logits: Floating-point tensor with global shape [..., vocab], with
            arbitrary leading dimensions. A DTensor must have exactly one
            ``Shard`` placement on the vocabulary axis and ``Replicate`` on
            every other mesh dimension; its per-rank local shape is
            [..., local_vocab].
        temperature: Positive finite scale applied before normalization.

    Returns:
        Differentiable fp32 tensor of shape [...] containing categorical entropy.
        The function does not mutate or gather ``logits``.

    Raises:
        TypeError: If ``logits`` has an invalid type or dtype.
        ValueError: If a shape, DTensor placement, vocabulary size, or
            temperature is invalid.
    """
    _validate_logits(logits, temperature)
    if isinstance(logits, DTensor):
        return _vocab_parallel_entropy(logits, temperature=temperature)

    leading_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    if flat_logits.shape[0] == 0:
        # Route through the input so an empty batch still produces a gradient.
        return flat_logits.float().sum(dim=-1).reshape(leading_shape)
    entropy = torch.empty(flat_logits.shape[0], dtype=torch.float32, device=logits.device)
    for start in range(0, flat_logits.shape[0], _DENSE_TOKEN_CHUNK_SIZE):
        end = min(start + _DENSE_TOKEN_CHUNK_SIZE, flat_logits.shape[0])
        chunk = flat_logits[start:end].float()
        if temperature != 1.0:
            chunk = chunk / temperature
        log_distribution = torch.log_softmax(chunk, dim=-1)
        entropy[start:end] = -(log_distribution.exp() * log_distribution).sum(dim=-1)
    return entropy.reshape(leading_shape)
