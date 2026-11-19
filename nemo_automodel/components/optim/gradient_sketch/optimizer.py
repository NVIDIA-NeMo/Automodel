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

"""Optimize gradient sketches while delegating ordinary parameters to a native optimizer."""

from __future__ import annotations

from typing import Any, Callable, cast, overload

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.optim.gradient_sketch.row_adam import row_adam_direction
from nemo_automodel.components.optim.gradient_sketch.sketch import (
    SketchSettings,
    SketchState,
    fill_projection_,
    install_sketches,
    local_weight_rows,
    projection_seed,
    sketch_buffer,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

_SETTINGS_KEY = "sketch_settings"
_COUNT_KEY = "update_count"


def _local(tensor: Tensor) -> Tensor:
    """Return the local shard of a ``DTensor``, or the tensor itself."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _parameter_fqn(layer: SketchState) -> str:
    """Canonical parameter FQN of a sketch target (``<module>.<param>``, no leading dot at the root)."""
    # Linear targets are named by their module; fused expert targets already end in ``.<param>``.
    module_fqn = layer.name if isinstance(layer.module, nn.Linear) else layer.name.removesuffix(f".{layer.weight_name}")
    return canonical_parameter_fqn(f"{module_fqn}.{layer.weight_name}" if module_fqn else layer.weight_name)


def _rows(tensor: Tensor) -> Tensor:
    """View the local part of a stored tensor as 2-D ``[rows, width]``, like ``local_weight_rows``."""
    local = _local(tensor)
    return local.reshape(-1, local.shape[-1])


class GradientSketchOptimizer(torch.optim.Optimizer):
    """RowAdam or SGD on gradient sketches, with a native optimizer for everything else.

    ``param_groups[0]`` holds the sketch targets (plus ``update_count`` and the
    settings); the remaining groups are the native optimizer's own group objects (none
    when every non-target parameter is frozen), and ``state`` is shared with it. ``state``
    holds native-parameter state only, so it is empty until the first native step and
    ``torch.distributed.checkpoint`` initializes it for any native optimizer. Sketch-target
    state (flat RowAdam row moments, fp32 masters of low-precision targets) is kept on the
    wrapper and exposed through ``sketch_state_dict`` / ``load_sketch_state_dict``
    (Automodel's ``OptimizerState`` saves it under the ``"sketch"`` key). Construct it before
    the native optimizer's first step: target parameters are removed before it can allocate
    full-size moment buffers.
    """

    def __init__(self, model: nn.Module, native: torch.optim.Optimizer, settings: SketchSettings) -> None:
        if native.state:
            raise ValueError("gradient sketch requires a fresh native optimizer with no state")
        if len(native.param_groups) != 1:
            raise ValueError("gradient sketch requires a native optimizer with one parameter group")
        self.settings = settings
        self.layers: list[SketchState] = install_sketches(model, settings)
        targets = {id(layer.weight) for layer in self.layers}
        group = native.param_groups[0]
        if settings.freeze_non_target:
            for parameter in model.parameters():
                if id(parameter) not in targets:
                    parameter.requires_grad_(False)
        group["params"] = [p for p in group["params"] if id(p) not in targets and p.requires_grad]
        self.native = native
        self.anchor_handle = (
            cast(Any, model).get_input_embeddings().register_forward_hook(self._anchor)
            if hasattr(model, "get_input_embeddings")
            else None
        )
        self.ready = False
        self.projections_verified = False
        self.sketches: Tensor = sketch_buffer(self.layers)
        self.local_rows: list[int] = [local_weight_rows(layer.weight)[1] for layer in self.layers]
        super().__init__(
            [layer.weight for layer in self.layers],
            {
                "lr": group["lr"],
                "weight_decay": group.get("weight_decay", 0.0),
                _COUNT_KEY: 0,
                _SETTINGS_KEY: settings.to_dict(),
            },
        )
        # One state mapping for both halves; one list of group objects for the scheduler. A native
        # group left without parameters is dropped: DCP's flattened state dict indexes the first
        # parameter of every group.
        self.native.state = self.state
        if group["params"]:
            self.param_groups.extend(native.param_groups)
        else:
            self.native.param_groups = []
        self.row_second_moment: Tensor | None = None
        if settings.optimizer == "row_adam":
            self.row_second_moment = torch.zeros(
                self.sketches.shape[0], 1, device=self.sketches.device, dtype=torch.float32
            )
        # fp32 masters for low-precision targets, aligned with ``self.layers``.
        self.masters: list[Tensor | None] = [
            self._new_master(layer.weight) if layer.weight.dtype != torch.float32 else None for layer in self.layers
        ]

    # ------------------------------------------------------------------ bookkeeping
    @staticmethod
    def _new_master(weight: Tensor) -> Tensor:
        """Create an fp32 master copy; a ``DTensor`` with the weight's layout under FSDP so DCP shards it."""
        # Stored in the weight's own (local) shape, 3-D for fused experts; ``_rows`` flattens it on use.
        master = _local(weight).float().clone()
        if isinstance(weight, DTensor):
            return DTensor.from_local(
                master,
                weight.device_mesh,
                weight.placements,
                run_check=False,
                shape=weight.shape,
                stride=weight.stride(),
            )
        return master

    @property
    def update_count(self) -> int:
        """Number of completed optimizer updates, stored in ``param_groups[0]`` for checkpoints."""
        return int(self.param_groups[0][_COUNT_KEY])

    @update_count.setter
    def update_count(self, value: int) -> None:
        self.param_groups[0][_COUNT_KEY] = int(value)

    @torch.no_grad()
    def to(self, device: torch.device | str) -> "GradientSketchOptimizer":
        """Move native state, the flat row moments and the fp32 masters to ``device``."""
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
        if self.row_second_moment is not None:
            self.row_second_moment = self.row_second_moment.to(device)
        self.masters = [None if master is None else master.to(device) for master in self.masters]
        return self

    @staticmethod
    def _anchor(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        # A frozen embedding would make every downstream activation requires_grad=False
        # and silently disable the sketch hooks; force a differentiable, non-leaf output.
        if torch.is_grad_enabled():
            output.requires_grad_(True)
            if output.is_leaf:
                output = output.clone()
        return output

    # ------------------------------------------------------------------ training step
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients of every parameter group and zero the sketch buffer.

        Args:
            set_to_none: Set gradients to ``None`` instead of zero, as in ``torch.optim.Optimizer``.
        """
        super().zero_grad(set_to_none=set_to_none)
        self.sketches.zero_()
        self.ready = False

    @torch.no_grad()
    def synchronize_and_clip(self, process_group: Any, max_norm: float | None) -> Tensor:
        """Average sketches across data-parallel ranks, then clip the implied gradient globally.

        After the all-reduce every rank holds the full averaged sketch, so the sketch part of the
        norm is computed locally over all rows; only the (sharded) native gradient norm is reduced.
        ``process_group`` must be the FSDP shard group; HSDP replicate groups are unsupported.

        Args:
            process_group: FSDP shard group, or ``None`` for the default group / a single process.
            max_norm: Clip the global gradient norm to this value, or ``None`` to only measure it.

        Returns:
            The pre-clip global norm as a 0-d tensor on the sketch device.

        Raises:
            RuntimeError: If sketches were already synchronized this step, targets are not
                FSDP-sharded on more than one rank, or projections differ across ranks.
            FloatingPointError: If the norm is not finite.
        """
        if self.ready:
            raise RuntimeError("Sketches have already been synchronized for this step")
        world = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if world > 1:
            self._check_sharded_targets()
        if world > 1 and not self.projections_verified:
            self._verify_shared_projections(process_group)
        if world > 1:
            dist.all_reduce(self.sketches, group=process_group)
            self.sketches.div_(world)
        # ||S Aᵀ||² = <S (A Aᵀ), S>: a rank x rank Gram matrix per layer; the dense gradient is never formed.
        sketch_terms: list[Tensor] = []
        for layer in self.layers:
            gram = layer.projection @ layer.projection.T
            sketch_terms.append(((layer.sketch @ gram) * layer.sketch).sum())
        sketch_norm2 = torch.stack(sketch_terms).sum()
        # Local shards as stored (any layout): the norm needs no row view, and scaling them in place
        # scales the real gradients.
        native_grads = [_local(p.grad) for g in self.native.param_groups for p in g["params"] if p.grad is not None]
        native_norm2 = torch.zeros((), device=self.sketches.device, dtype=torch.float32)
        if native_grads:
            native_norm2 = torch.stack(
                [n.square() for n in torch._foreach_norm(native_grads, 2, dtype=torch.float32)]
            ).sum()
        if world > 1:
            dist.all_reduce(native_norm2, group=process_group)
        norm = (sketch_norm2 + native_norm2).sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient sketch norm")
        if max_norm is not None:
            scale = (max_norm / (norm + 1e-6)).clamp(max=1)
            self.sketches.mul_(scale)
            if native_grads:
                torch._foreach_mul_(native_grads, scale)
        self.ready = True
        return norm

    def _check_sharded_targets(self) -> None:
        for layer in self.layers:
            weight = layer.weight
            if not isinstance(weight, DTensor):
                raise RuntimeError(
                    "gradient sketch requires FSDP-sharded (DTensor) targets when running on more than one rank"
                )
            placements = weight.placements
            if any(isinstance(p, Replicate) for p in placements) or sum(isinstance(p, Shard) for p in placements) > 1:
                raise RuntimeError(
                    f"gradient sketch requires targets sharded once along rows by FSDP; {layer.name} has "
                    f"placements {placements}"
                )

    @torch.no_grad()
    def _verify_shared_projections(self, process_group: Any) -> None:
        fingerprint = torch.stack([layer.projection.double().sum() for layer in self.layers])
        lowest, highest = fingerprint.clone(), fingerprint.clone()
        dist.all_reduce(lowest, op=dist.ReduceOp.MIN, group=process_group)
        dist.all_reduce(highest, op=dist.ReduceOp.MAX, group=process_group)
        if not torch.equal(lowest, highest):
            raise RuntimeError(
                "gradient sketch projections differ across data-parallel ranks; ranks must run identical "
                "hardware so a shared seed draws one matrix"
            )
        self.projections_verified = True

    # Mirror ``Optimizer.step``'s overloads so the override stays type-compatible.
    @overload
    def step(self, closure: None = None, *, scale: float = 1.0) -> None: ...

    @overload
    def step(self, closure: Callable[[], float], *, scale: float = 1.0) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None, *, scale: float = 1.0) -> float | None:
        """Apply the sketch update to targets and the native step to everything else.

        An unready call with every group's ``lr`` equal to 0 is the state-initialization probe
        used by ``torch.distributed.checkpoint``: only the native optimizer runs, so its state
        is created; sketches, projections and ``update_count`` are untouched.

        Args:
            closure: Unsupported; must be ``None``.
            scale: Multiplies only the sketch-target update, not its weight decay or the native step.

        Returns:
            ``None``; closures are unsupported, so there is no loss to return.

        Raises:
            RuntimeError: If ``synchronize_and_clip`` has not run this step (and this is not the
                zero-lr probe), or a closure is given.
        """
        if closure is None and not self.ready and all(g["lr"] == 0 for g in self.param_groups):
            self.native.step()
            return None
        if closure is not None or not self.ready:
            raise RuntimeError("Call synchronize_and_clip before step; closures are unsupported")
        group = self.param_groups[0]
        if self.row_second_moment is not None:
            row_adam_direction(
                self.sketches,
                self.row_second_moment,
                step=self.update_count + 1,
                beta2=self.settings.beta2,
                epsilon=self.settings.epsilon,
                inplace=True,
            )
        decay = 1 - group["lr"] * group["weight_decay"]
        alpha = -group["lr"] * scale
        for layer, offset, stored in zip(self.layers, self.local_rows, self.masters, strict=True):
            local, _ = local_weight_rows(layer.weight)
            master = None if stored is None else _rows(stored)
            destination = local if master is None else master
            destination.addmm_(
                layer.sketch[offset : offset + local.shape[0]], layer.projection, beta=decay, alpha=alpha
            )
            if master is not None:
                local.copy_(master)
        self.native.step()
        self.update_count += 1
        self.ready = False
        self.refresh_projection()
        return None

    @torch.no_grad()
    def reset_master_weights(self) -> None:
        """Rebase fp32 masters after loading model weights without optimizer state."""
        for layer, master in zip(self.layers, self.masters, strict=True):
            if master is not None:
                _rows(master).copy_(local_weight_rows(layer.weight)[0])

    def refresh_projection(self) -> None:
        """Redraw every projection on its device from ``(seed, update_count, layer name)``."""
        update = self.update_count if self.settings.refresh else 0
        for layer in self.layers:
            fill_projection_(
                layer.projection,
                projection_seed(self.settings.seed, update, layer.name),
                distribution=self.settings.distribution,
            )
            layer.refresh_cast()

    # ------------------------------------------------------------------ checkpointing
    def sketch_state_dict(self) -> dict[str, Tensor]:
        """Live tensors holding the state of sketch-target parameters.

        Returns:
            ``row_second_moment`` (the flat ``[rows, 1]`` fp32 buffer, identical on every rank, only
            for RowAdam) and ``master_weight.<parameter fqn>`` for each target that keeps an fp32 master
            (``DTensor`` under FSDP). Tensors are not cloned so DCP can use them as a load skeleton.
        """
        sketch_state: dict[str, Tensor] = {}
        if self.row_second_moment is not None:
            sketch_state["row_second_moment"] = self.row_second_moment
        for layer, master in zip(self.layers, self.masters, strict=True):
            if master is not None:
                sketch_state[f"master_weight.{_parameter_fqn(layer)}"] = master
        return sketch_state

    @torch.no_grad()
    def load_sketch_state_dict(self, sketch_state: dict[str, Tensor]) -> None:
        """Restore state produced by :meth:`sketch_state_dict`.

        Args:
            sketch_state: Mapping with the same keys as :meth:`sketch_state_dict`.

        Raises:
            ValueError: If a key is missing or unexpected, or a shape differs.
        """
        expected = self.sketch_state_dict()
        for key in expected:
            if key not in sketch_state:
                raise ValueError(f"gradient sketch checkpoint is missing {key}")
        for key in sketch_state:
            if key not in expected:
                raise ValueError(f"gradient sketch checkpoint has unexpected key {key}")
        for key, target in expected.items():
            saved = sketch_state[key]
            if saved.shape != target.shape:
                raise ValueError(
                    f"gradient sketch checkpoint shape mismatch for {key}: {tuple(saved.shape)} vs "
                    f"{tuple(target.shape)} (resume requires the same data-parallel size for plain-tensor masters)"
                )
        for key, target in expected.items():
            _rows(target).copy_(_rows(sketch_state[key]).to(torch.float32))

    def state_dict(self) -> dict[str, Any]:
        """Pack ``param_groups[0]`` (settings, ``update_count``) with the native optimizer's own state dict.

        The native half comes from ``native.state_dict()`` so an optimizer with its own
        serialization (TE ``FusedAdam`` unscales low-precision moments to fp32) is honoured;
        its parameter indices are shifted past the sketch targets so the layout matches
        :class:`torch.optim.Optimizer`'s packing of all groups in order.
        """
        packed = super().state_dict()
        offset = len(self.param_groups[0]["params"])
        native_state = self.native.state_dict()["state"] if self.native.param_groups else {}
        packed["state"] = {index + offset: value for index, value in native_state.items()}
        return packed

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load native state, group settings and ``update_count`` from :meth:`state_dict`.

        The native half is handed to ``native.load_state_dict`` (``torch.optim.Optimizer``
        casts loaded state to the parameter dtype; TE ``FusedAdam`` restores fp32 there).
        Sketch-target state is not read here; use :meth:`load_sketch_state_dict`.

        Args:
            state_dict: A dict produced by :meth:`state_dict`.

        Raises:
            ValueError: If the saved sketch settings differ from the configured ones, or the
                saved group layout differs from the current one.
        """
        groups = state_dict["param_groups"]
        if groups[0].get(_SETTINGS_KEY) != self.settings.to_dict():
            raise ValueError("gradient sketch checkpoint settings differ from the configured settings")
        if len(groups) != len(self.param_groups) or len(groups[0]["params"]) != len(self.param_groups[0]["params"]):
            raise ValueError("gradient sketch checkpoint parameter groups differ from the configured ones")
        # Parameters are matched by position within each group, so the saved keys may be the indices
        # written by :meth:`state_dict` or the FQNs DCP's flattened path substitutes for them.
        target_keys = set(groups[0]["params"])
        if any(key in target_keys for key in state_dict["state"]):
            raise ValueError("gradient sketch checkpoint holds state for sketch targets; use load_sketch_state_dict")
        self.param_groups[0].update({key: value for key, value in groups[0].items() if key != "params"})
        if self.native.param_groups:
            native_keys = {key for group in groups[1:] for key in group["params"]}
            self.native.load_state_dict(
                {
                    "state": {key: value for key, value in state_dict["state"].items() if key in native_keys},
                    "param_groups": groups[1:],
                }
            )
        # Torch rebuilt the native state mapping and group objects: share them again.
        self.state = self.native.state
        self.param_groups = [self.param_groups[0], *self.native.param_groups]
        self.refresh_projection()
