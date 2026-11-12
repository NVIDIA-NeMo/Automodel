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

"""Sketch targets, projections, and the hooks that accumulate projected gradients."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Shard

_DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_DISTRIBUTIONS = ("rademacher", "gaussian")
_RULES = ("row_adam", "sgd")


@dataclass(frozen=True)
class SketchSettings:
    """Validated settings shared by target selection, hooks, and the optimizer.

    Attributes:
        rank: Sketch rank ``r`` (columns of ``S``, rows of ``A``).
        seed: Base seed; each layer and update derives its own seed from it.
        distribution: ``"rademacher"`` (±1/√r) or ``"gaussian"`` (N(0, 1/r)).
        refresh: Redraw ``A`` after every update; ``False`` keeps the update-0 draw.
        optimizer: ``"row_adam"`` or ``"sgd"`` rule applied to the sketch.
        beta2: RowAdam second-moment decay.
        epsilon: RowAdam denominator floor.
        target_modules: Last name segments of ``nn.Linear`` modules to sketch.
        target_parameters: Names of 3-D fused expert parameters to sketch.
        target_prefixes: Restrict selection to modules under these prefixes.
        freeze_non_target: Freeze every parameter that is not a sketch target.
    """

    rank: int = 256
    seed: int = 42
    distribution: str = "rademacher"
    refresh: bool = True
    optimizer: str = "row_adam"
    beta2: float = 0.95
    epsilon: float = 1e-8
    target_modules: tuple[str, ...] = _DEFAULT_TARGET_MODULES
    target_parameters: tuple[str, ...] = ()
    target_prefixes: tuple[str, ...] = ()
    freeze_non_target: bool = False

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"gradient sketch rank must be positive, got {self.rank}")
        if self.distribution not in _DISTRIBUTIONS:
            raise ValueError(f"gradient sketch distribution must be one of {_DISTRIBUTIONS}, got {self.distribution!r}")
        if self.optimizer not in _RULES:
            raise ValueError(f"gradient sketch optimizer must be one of {_RULES}, got {self.optimizer!r}")
        if not 0 <= self.beta2 < 1:
            raise ValueError(f"gradient sketch beta2 must satisfy 0 <= beta2 < 1, got {self.beta2}")
        if self.epsilon <= 0:
            raise ValueError(f"gradient sketch epsilon must be positive, got {self.epsilon}")
        # YAML delivers lists; store tuples so the settings stay hashable and immutable.
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(self, "target_parameters", tuple(self.target_parameters))
        object.__setattr__(self, "target_prefixes", tuple(self.target_prefixes))

    def to_dict(self) -> dict[str, Any]:
        """Plain-value form stored in optimizer ``param_groups`` for checkpoints."""
        return {
            "rank": self.rank,
            "seed": self.seed,
            "distribution": self.distribution,
            "refresh": self.refresh,
            "optimizer": self.optimizer,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "target_modules": list(self.target_modules),
            "target_parameters": list(self.target_parameters),
            "target_prefixes": list(self.target_prefixes),
            "freeze_non_target": self.freeze_non_target,
        }


def projection_seed(base_seed: int, update: int, name: str) -> int:
    """Generate a stable per-layer seed independently of module wrappers.

    Args:
        base_seed: Run-level seed from ``SketchSettings.seed``.
        update: Optimizer update index; changes the seed on every refresh.
        name: Fully qualified module name; FSDP and checkpoint wrapper segments are ignored.

    Returns:
        A non-negative 62-bit seed.
    """
    wrappers = {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
    name = ".".join(part for part in name.split(".") if part not in wrappers)
    digest = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")
    return (base_seed * 1_000_003 + update * 7_919 + digest) & ((1 << 62) - 1)


def fill_projection_(out: Tensor, seed: int, *, distribution: str) -> Tensor:
    """Overwrite ``out`` with a fresh ``[rank, width]`` projection drawn on its own device.

    Drawing on the device keeps the host out of the per-step refresh: no CPU random
    stream, no temporaries, no host-to-device copy. Every data-parallel rank draws from
    the same seed on identical hardware and therefore holds the same matrix; the
    optimizer verifies that once before the first update.

    Args:
        out: Preallocated ``[rank, width]`` tensor to overwrite in place.
        seed: Seed for the device-side generator.
        distribution: ``"rademacher"`` or ``"gaussian"``.

    Returns:
        The same tensor ``out``, now filled.

    Raises:
        ValueError: If a dimension is non-positive or the distribution is unknown.
    """
    rank = out.shape[0]
    if rank <= 0 or out.shape[1] <= 0:
        raise ValueError("Projection dimensions must be positive")
    scale = 1 / math.sqrt(rank)
    generator = torch.Generator(device=out.device).manual_seed(seed & ((1 << 62) - 1))
    if distribution == "rademacher":
        out.random_(0, 2, generator=generator)
        out.mul_(2 * scale).sub_(scale)
    elif distribution == "gaussian":
        out.normal_(0.0, scale, generator=generator)
    else:
        raise ValueError(f"Unsupported projection distribution: {distribution!r}")
    return out


def make_projection(
    rank: int,
    width: int,
    seed: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    distribution: str,
) -> Tensor:
    """Allocate a ``[rank, width]`` projection and fill it on ``device``.

    Args:
        rank: Number of rows ``r`` of the projection.
        width: Number of columns, the weight's input dimension.
        seed: Seed for the device-side generator.
        device: Device to allocate and draw on.
        dtype: Element type of the projection.
        distribution: ``"rademacher"`` or ``"gaussian"``.

    Returns:
        The filled projection tensor.

    Raises:
        ValueError: If ``rank`` or ``width`` is non-positive.
    """
    if rank <= 0 or width <= 0:
        raise ValueError("Projection dimensions must be positive")
    return fill_projection_(torch.empty(rank, width, device=device, dtype=dtype), seed, distribution=distribution)


def local_weight_rows(weight: Tensor) -> tuple[Tensor, int]:
    """Return the local 2-D weight rows and their global row offset.

    Plain tensors are returned whole (3-D expert banks flattened to 2-D). FSDP
    row-sharded ``DTensor``s return the local shard and its offset; any sharding on
    a non-leading dimension (tensor parallelism) is rejected.

    Args:
        weight: A plain or ``DTensor`` weight of 2 or more dimensions.

    Returns:
        A ``(rows, offset)`` pair: the local 2-D rows and the global index of their first row.

    Raises:
        ValueError: If the weight is sharded on a dimension other than the leading one.
    """
    if not isinstance(weight, DTensor):
        return (weight.reshape(-1, weight.shape[-1]) if weight.ndim > 2 else weight), 0
    if any(isinstance(p, Shard) and p.dim != 0 for p in weight.placements):
        raise ValueError("gradient sketch supports FSDP row sharding, not tensor parallelism")
    # Private helper; it is the only torch API that handles uneven and empty FSDP shards.
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(weight.shape, weight.device_mesh, weight.placements)
    local = weight.to_local()
    if weight.ndim > 2:
        return local.reshape(-1, weight.shape[-1]), offset[0] * math.prod(weight.shape[1:-1])
    return local, offset[0]
