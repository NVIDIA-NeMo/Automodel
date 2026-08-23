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
    """Sum local token statistics over the vocabulary-shard mesh dimension.

    Args:
        values: Rank-local tensor of shape [..., statistics], with arbitrary
            leading dimensions and one or more statistics on the last axis.
        logits: DTensor with global shape [..., vocab]. Its device mesh owns the
            collective, with ``Shard`` on the last axis at ``vocab_mesh_dim``.
        vocab_mesh_dim: Device-mesh dimension that shards the vocabulary axis.

    Returns:
        Replicated tensor of shape [..., statistics] in the same dtype and on
        the same device as ``values``. The result does not alias ``values``.
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
    """Validate and locally shift vocabulary-sharded logits.

    Args:
        logits: DTensor with global shape [..., vocab] and arbitrary leading
            dimensions. Exactly one device-mesh dimension must use ``Shard`` on
            the last, vocabulary axis; every other mesh dimension must use
            ``Replicate``. Each rank's local shape is [..., local_vocab].
        temperature: Positive finite scale applied to logits before normalization.

    Returns:
        A tuple containing a rank-local fp32 tensor of shape [..., local_vocab],
        with arbitrary leading dimensions, this rank's global vocabulary offset,
        and the vocabulary-shard mesh dimension. The tensor does not alias or
        mutate ``logits``.

    Raises:
        TypeError: If ``logits`` is not a floating-point DTensor.
        ValueError: If the DTensor placement, shape, or temperature is invalid.
    """
    if not isinstance(logits, DTensor):
        raise TypeError(f"logits must be a DTensor, got {type(logits).__name__}")
    if logits.ndim == 0:
        raise ValueError("logits must have shape [..., vocab]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")

    vocab_mesh_dim = None
    for mesh_dim, placement in enumerate(logits.placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim if placement.dim >= 0 else placement.dim + logits.ndim
            if shard_dim != logits.ndim - 1 or vocab_mesh_dim is not None:
                raise ValueError(
                    "logits must have exactly one Shard placement on the last vocabulary axis; "
                    f"got placements={logits.placements}"
                )
            vocab_mesh_dim = mesh_dim
        elif not isinstance(placement, Replicate):
            raise ValueError(
                f"logits must be replicated on every non-vocabulary mesh dimension; got placements={logits.placements}"
            )
    if vocab_mesh_dim is None:
        raise ValueError(
            "logits must have exactly one Shard placement on the last vocabulary axis; "
            f"got placements={logits.placements}"
        )

    global_vocab_size = int(logits.shape[-1])
    if global_vocab_size <= 0:
        raise ValueError(f"logits vocabulary size must be positive, got {global_vocab_size}")

    mesh = logits.device_mesh
    shard_count = mesh.size(vocab_mesh_dim)
    shard_rank = mesh.get_local_rank(vocab_mesh_dim)
    chunk_size = (global_vocab_size + shard_count - 1) // shard_count
    shard_offset = min(shard_rank * chunk_size, global_vocab_size)
    shard_size = min(chunk_size, global_vocab_size - shard_offset)
    local_logits = logits.to_local()
    if not local_logits.is_floating_point():
        raise TypeError(f"logits must have a floating-point dtype, got {local_logits.dtype}")
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
    """Compute selected-token log probabilities without gathering the vocabulary.

    Args:
        logits: DTensor with global shape [..., vocab] and arbitrary leading
            dimensions. Exactly one device-mesh dimension must use ``Shard`` on
            the last, vocabulary axis; every other mesh dimension must use
            ``Replicate``. Each rank's local shape is [..., local_vocab].
        targets: Rank-local int64 tensor of shape [...], matching the leading
            dimensions of ``logits``. It must contain global vocabulary indices
            in ``[0, vocab)`` and be replicated across the vocabulary-shard
            mesh dimension.
        temperature: Positive finite scale applied to logits before normalization.

    Returns:
        Replicated fp32 tensor of shape [...] containing the selected-token log
        probabilities. The output has arbitrary leading dimensions, does not
        gather the vocabulary, and does not alias or mutate an input. Positions
        with targets outside the global vocabulary contain ``NaN``.

    Raises:
        TypeError: If ``logits`` or ``targets`` has an invalid type or dtype.
        ValueError: If a tensor shape, DTensor placement, or temperature is invalid.
    """
    shifted_logits, shard_offset, vocab_mesh_dim = _shifted_local_logits(logits, temperature)
    if isinstance(targets, DTensor):
        raise TypeError("targets must be a rank-local torch.Tensor, not a DTensor")
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"targets must be a torch.Tensor, got {type(targets).__name__}")
    if targets.dtype != torch.long:
        raise TypeError(f"targets must have dtype torch.int64, got {targets.dtype}")
    if targets.device != shifted_logits.device:
        raise ValueError(
            f"targets and logits must be on the same device, got {targets.device} and {shifted_logits.device}"
        )
    if tuple(targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            f"targets shape must match logits leading shape {tuple(logits.shape[:-1])}, got {tuple(targets.shape)}"
        )

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
    """Compute categorical entropy without gathering a sharded vocabulary.

    Args:
        logits: DTensor with global shape [..., vocab] and arbitrary leading
            dimensions. Exactly one device-mesh dimension must use ``Shard`` on
            the last, vocabulary axis; every other mesh dimension must use
            ``Replicate``. Each rank's local shape is [..., local_vocab].
        temperature: Positive finite scale applied to logits before normalization.

    Returns:
        Replicated fp32 tensor of shape [...] containing categorical entropy.
        The output has arbitrary leading dimensions, does not gather the
        vocabulary, and does not alias or mutate ``logits``.

    Raises:
        TypeError: If ``logits`` is not a floating-point DTensor.
        ValueError: If the DTensor placement, shape, or temperature is invalid.
    """
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


def token_log_probs(
    logits: torch.Tensor | DTensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute selected-token log probabilities for dense or vocabulary-sharded logits.

    Dense logits are converted to fp32 in chunks of at most 256 token rows so
    long sequences do not materialize the entire vocabulary tensor in fp32.
    Vocabulary-sharded DTensors use distributed reductions without gathering
    the vocabulary.

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
    if isinstance(logits, DTensor):
        return _vocab_parallel_log_probs(logits, targets, temperature=temperature)
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

    leading_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
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
    if flat_targets.numel() == 0:
        log_probs = flat_logits.float().sum(dim=-1)
    return log_probs.reshape(leading_shape)


def token_entropy(
    logits: torch.Tensor | DTensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute categorical entropy for dense or vocabulary-sharded logits.

    Dense logits are converted to fp32 in chunks of at most 256 token rows so
    long sequences do not materialize the entire vocabulary tensor in fp32.
    Vocabulary-sharded DTensors use distributed reductions without gathering
    the vocabulary.

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
    if isinstance(logits, DTensor):
        return _vocab_parallel_entropy(logits, temperature=temperature)
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

    leading_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    entropy = torch.empty(flat_logits.shape[0], dtype=torch.float32, device=logits.device)
    for start in range(0, flat_logits.shape[0], _DENSE_TOKEN_CHUNK_SIZE):
        end = min(start + _DENSE_TOKEN_CHUNK_SIZE, flat_logits.shape[0])
        chunk = flat_logits[start:end].float()
        if temperature != 1.0:
            chunk = chunk / temperature
        log_distribution = torch.log_softmax(chunk, dim=-1)
        entropy[start:end] = -(log_distribution.exp() * log_distribution).sum(dim=-1)
    if flat_logits.shape[0] == 0:
        entropy = flat_logits.float().sum(dim=-1)
    return entropy.reshape(leading_shape)


# Import-only compatibility for callers of the previously released names.
# New code should use the canonical Tensor-or-DTensor APIs above.
vocab_parallel_log_probs = token_log_probs
vocab_parallel_entropy = token_entropy
