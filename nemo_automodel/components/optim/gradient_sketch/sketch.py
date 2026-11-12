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
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Shard
from torch.utils.hooks import RemovableHandle

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


class _AccumulateSketch(torch.autograd.Function):
    """Keep projected activations in autograd-managed storage, freed after backward."""

    @staticmethod
    def forward(ctx: Any, output: Tensor, projected: Tensor, sketch: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(projected)
        ctx.sketch = sketch
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None, None]:  # type: ignore[override]
        (gradient,) = grad_outputs
        (projected,) = ctx.saved_tensors
        with torch.no_grad():
            ctx.sketch.add_((gradient.reshape(-1, gradient.shape[-1]).T @ projected).float())
        return gradient, None, None


class _ProjectWeightGradient(torch.autograd.Function):
    """Project a fused expert gradient before it reaches the stored parameter.

    Unlike the linear hook, this path materializes one transient dense gradient and
    keeps the backend's expert dispatch and matrix multiplication unchanged.
    """

    @staticmethod
    def forward(ctx: Any, weight: Tensor, state: "SketchState", row_offset: int) -> Tensor:  # type: ignore[override]
        ctx.state = state
        ctx.row_offset = row_offset
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[None, None, None]:  # type: ignore[override]
        (gradient,) = grad_outputs
        state: SketchState = ctx.state
        rows = gradient.reshape(-1, gradient.shape[-1])
        with torch.no_grad():
            # Bound the FP32 conversion workspace, including large expert banks.
            for start in range(0, rows.shape[0], 256):
                stop = min(start + 256, rows.shape[0])
                state.sketch[ctx.row_offset + start : ctx.row_offset + stop].add_(
                    rows[start:stop].float() @ state.projection.T
                )
        return None, None, None


@dataclass
class SketchState:
    """Per-weight sketch; model parameter names and checkpoint keys stay unchanged.

    ``sketch`` is a ``[rows, rank]`` view into one buffer shared by all targets (see
    :func:`install_sketches`), so the optimizer can reduce, scale and zero every sketch
    with a single kernel or collective. ``rows`` is the product of all but the last
    weight dimension, so fused 3-D expert banks are flattened.
    """

    name: str
    module: nn.Module
    weight_name: str
    projection: Tensor
    sketch: Tensor
    handles: list[RemovableHandle] = field(default_factory=list)
    swapped_weight: Tensor | None = None
    # Persistent copy of the projection in the activation dtype; refreshed in place so
    # forward hooks never allocate.
    projection_cast: Tensor | None = None

    @property
    def weight(self) -> nn.Parameter:
        weight = self.module._parameters[self.weight_name]
        if weight is None:
            raise RuntimeError(f"Selected parameter disappeared: {self.name}")
        return weight

    def projection_as(self, dtype: torch.dtype) -> Tensor:
        """Return the projection in ``dtype``, reusing one persistent copy."""
        if dtype == self.projection.dtype:
            return self.projection
        if self.projection_cast is None or self.projection_cast.dtype != dtype:
            self.projection_cast = self.projection.to(dtype)
        return self.projection_cast

    def refresh_cast(self) -> None:
        """Mirror a refreshed ``projection`` into the low-precision copy."""
        if self.projection_cast is not None:
            self.projection_cast.copy_(self.projection)

    def forward_hook(self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor | None:
        """Project the layer input and register the sketch accumulation for backward."""
        if not torch.is_grad_enabled() or not output.requires_grad:
            return None
        with torch.no_grad():
            activation = inputs[0]
            projected = activation.reshape(-1, activation.shape[-1]) @ self.projection_as(activation.dtype).T
        return _AccumulateSketch.apply(output, projected, self.sketch)

    def expert_pre_hook(self, module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        """Swap the fused expert weight for a view whose gradient is projected."""
        if not torch.is_grad_enabled():
            return
        if self.swapped_weight is not None:
            raise RuntimeError(f"Reentrant expert forward is unsupported: {self.name}")
        weight = self.weight
        local, offset = local_weight_rows(weight)
        local_shape = weight.to_local().shape if isinstance(weight, DTensor) else weight.shape
        projected = _ProjectWeightGradient.apply(local.reshape(local_shape).detach().requires_grad_(True), self, offset)
        if isinstance(weight, DTensor):
            projected = DTensor.from_local(
                projected,
                weight.device_mesh,
                weight.placements,
                run_check=False,
                shape=weight.shape,
                stride=weight.stride(),
            )
        self.swapped_weight = weight
        cast(dict[str, Tensor | None], module._parameters)[self.weight_name] = projected

    def expert_post_hook(self, module: nn.Module, inputs: tuple[Tensor, ...], output: Any) -> None:
        """Restore the real parameter after the fused expert forward."""
        if self.swapped_weight is not None:
            cast(dict[str, Tensor | None], module._parameters)[self.weight_name] = self.swapped_weight
            self.swapped_weight = None


def select_sketch_targets(model: nn.Module, settings: SketchSettings) -> list[tuple[str, nn.Module, str, bool]]:
    """Return ``(name, module, parameter_name, is_fused_expert)`` for every sketch target."""
    targets: list[tuple[str, nn.Module, str, bool]] = []
    for name, module in model.named_modules():
        if settings.target_prefixes and not any(
            name == prefix or name.startswith(prefix + ".") for prefix in settings.target_prefixes
        ):
            continue
        if name.rsplit(".", 1)[-1] in settings.target_modules:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"gradient sketch requires nn.Linear targets, got {name}: {type(module)}")
            targets.append((name, module, "weight", False))
        for parameter_name in settings.target_parameters:
            parameter = module._parameters.get(parameter_name)
            if parameter is None:
                continue
            if parameter.ndim != 3:
                raise ValueError(f"Expected a 3-D fused expert weight: {name}.{parameter_name}")
            targets.append((f"{name}.{parameter_name}", module, parameter_name, True))
    if not targets:
        raise ValueError("gradient sketch target_modules/target_parameters matched no modules or parameters")
    return targets


def install_sketches(model: nn.Module, settings: SketchSettings) -> list[SketchState]:
    """Install sketch hooks on every target and allocate the shared sketch buffer.

    Linear targets accumulate sketches directly in backward. Fused experts keep their
    forward implementation and project the transient backward gradient. Target weights
    get ``requires_grad=False`` so autograd never allocates their dense gradient.
    """
    targets = select_sketch_targets(model, settings)
    weights: list[Tensor] = []
    for _, module, weight_name, _ in targets:
        weight = module._parameters[weight_name]
        if weight is None:
            raise ValueError(f"Selected parameter is empty: {weight_name}")
        local_weight_rows(weight)  # validates sharding before any hook is installed
        weights.append(weight)
    device = weights[0].device
    if any(weight.device != device for weight in weights):
        raise ValueError("gradient sketch targets must share one device")
    row_counts = [math.prod(weight.shape[:-1]) for weight in weights]
    buffer = torch.zeros(sum(row_counts), settings.rank, device=device, dtype=torch.float32)
    states: list[SketchState] = []
    row = 0
    for (name, module, weight_name, fused), weight, rows in zip(targets, weights, row_counts, strict=True):
        projection = make_projection(
            settings.rank,
            weight.shape[-1],
            projection_seed(settings.seed, 0, name),
            device=device,
            dtype=torch.float32,
            distribution=settings.distribution,
        )
        state = SketchState(name, module, weight_name, projection, buffer[row : row + rows])
        row += rows
        weight.requires_grad_(False)
        if fused:
            state.handles = [
                module.register_forward_pre_hook(state.expert_pre_hook),
                module.register_forward_hook(state.expert_post_hook, always_call=True),
            ]
        else:
            state.handles = [module.register_forward_hook(state.forward_hook)]
        states.append(state)
    return states


def sketch_buffer(states: list[SketchState]) -> Tensor:
    """Return the shared buffer that every ``SketchState.sketch`` is a view of."""
    base = states[0].sketch
    total_rows = sum(state.sketch.shape[0] for state in states)
    if base.storage_offset() != 0 or any(
        state.sketch.untyped_storage().data_ptr() != base.untyped_storage().data_ptr() for state in states
    ):
        raise ValueError("Sketches are not views of one shared buffer")
    return base.as_strided((total_rows, base.shape[1]), base.stride())
