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

"""Optimize gradient sketches while delegating ordinary parameters to a native optimizer."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn

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

_SETTINGS_KEY = "sketch_settings"
_COUNT_KEY = "update_count"


class GradientSketchOptimizer(torch.optim.Optimizer):
    """RowAdam or SGD on gradient sketches, with a native optimizer for everything else.

    ``param_groups[0]`` holds the sketch targets (plus ``update_count`` and the
    settings); the remaining groups are the native optimizer's own group objects, and
    ``state`` is shared with it, so the standard ``state_dict`` covers both and DCP
    checkpointing needs no special handling. Construct it before the native
    optimizer's first step: target parameters are removed before it can allocate
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
        # One state mapping for both halves; one list of group objects for the scheduler.
        self.native.state = self.state
        self.param_groups.extend(native.param_groups)
        self.row_second_moment: Tensor | None = None
        if settings.optimizer == "row_adam":
            self.row_second_moment = torch.zeros(
                self.sketches.shape[0], 1, device=self.sketches.device, dtype=torch.float32
            )
            self._link_row_moments()
        for layer in self.layers:
            if layer.weight.dtype != torch.float32:
                local, _ = local_weight_rows(layer.weight)
                self.state[layer.weight]["master_weight"] = local.float().clone()

    # ------------------------------------------------------------------ bookkeeping
    @property
    def update_count(self) -> int:
        return int(self.param_groups[0][_COUNT_KEY])

    @update_count.setter
    def update_count(self, value: int) -> None:
        self.param_groups[0][_COUNT_KEY] = int(value)

    def _link_row_moments(self) -> None:
        moments = self.row_second_moment
        assert moments is not None
        row = 0
        for layer in self.layers:
            rows = layer.sketch.shape[0]
            self.state[layer.weight]["row_second_moment"] = moments[row : row + rows]
            row += rows

    def _gather_row_moments_from_state(self) -> None:
        """Copy per-parameter moment tensors (restored by load or offload) into the flat buffer."""
        moments = self.row_second_moment
        if moments is None:
            return
        row = 0
        for layer in self.layers:
            rows = layer.sketch.shape[0]
            restored = self.state[layer.weight].get("row_second_moment")
            if restored is None:
                moments[row : row + rows].zero_()
            else:
                moments[row : row + rows].copy_(restored)
            row += rows
        self._link_row_moments()

    @torch.no_grad()
    def to(self, device: torch.device | str) -> "GradientSketchOptimizer":
        """Move all optimizer state to ``device`` and restore the flat-buffer views."""
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
        if self.row_second_moment is not None:
            self.row_second_moment = self.row_second_moment.to(device)
            self._gather_row_moments_from_state()
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
        super().zero_grad(set_to_none=set_to_none)
        self.native.zero_grad(set_to_none=set_to_none)
        self.sketches.zero_()
        self.ready = False

    @torch.no_grad()
    def synchronize_and_clip(self, process_group: Any, max_norm: float | None) -> Tensor:
        """Average sketches across data-parallel ranks, then clip the implied gradient globally.

        Returns the pre-clip global norm as a 0-d tensor on the sketch device.
        """
        if self.ready:
            raise RuntimeError("Sketches have already been synchronized for this step")
        world = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if world > 1 and not self.projections_verified:
            self._verify_shared_projections(process_group)
        if world > 1:
            dist.all_reduce(self.sketches, group=process_group)
            self.sketches.div_(world)
        # ||S Aᵀ||² = <S (A Aᵀ), S>: a rank x rank Gram matrix per layer; the dense gradient is never formed.
        partial: list[Tensor] = []
        for layer, offset in zip(self.layers, self.local_rows, strict=True):
            local, _ = local_weight_rows(layer.weight)
            rows = layer.sketch[offset : offset + local.shape[0]]
            gram = layer.projection @ layer.projection.T
            partial.append(((rows @ gram) * rows).sum())
        native_grads = [
            local_weight_rows(p.grad)[0] for g in self.native.param_groups for p in g["params"] if p.grad is not None
        ]
        if native_grads:
            partial.extend(n.square() for n in torch._foreach_norm(native_grads, 2, dtype=torch.float32))
        norm2 = torch.stack(partial).sum()
        if world > 1:
            dist.all_reduce(norm2, group=process_group)
        norm = norm2.sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient sketch norm")
        if max_norm is not None:
            scale = (max_norm / (norm + 1e-6)).clamp(max=1)
            self.sketches.mul_(scale)
            if native_grads:
                torch._foreach_mul_(native_grads, scale)
        self.ready = True
        return norm

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

    @torch.no_grad()
    def step(self, closure: Any = None, *, scale: float = 1.0) -> None:  # type: ignore[override]
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
        for layer, offset in zip(self.layers, self.local_rows, strict=True):
            local, _ = local_weight_rows(layer.weight)
            master = self.state[layer.weight].get("master_weight")
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

    @torch.no_grad()
    def reset_master_weights(self) -> None:
        """Rebase fp32 masters after loading model weights without optimizer state."""
        for layer in self.layers:
            master = self.state[layer.weight].get("master_weight")
            if master is not None:
                master.copy_(local_weight_rows(layer.weight)[0])

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
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Standard torch format, with fp32 kept for sketch-target state.

        ``torch.optim.Optimizer.load_state_dict`` casts floating state to the parameter
        dtype, which would round fp32 moments and masters of bf16 targets. Native
        parameters go through the parent implementation; target state is copied back
        explicitly in fp32.
        """
        saved_settings = state_dict["param_groups"][0].get(_SETTINGS_KEY)
        if saved_settings != self.settings.to_dict():
            raise ValueError("gradient sketch checkpoint settings differ from the configured settings")
        target_ids = set(state_dict["param_groups"][0]["params"])
        native_state = {k: v for k, v in state_dict["state"].items() if k not in target_ids}
        super().load_state_dict({"state": native_state, "param_groups": state_dict["param_groups"]})
        # Torch replaced ``self.state`` with a new mapping; re-share it with the native optimizer.
        self.native.state = self.state
        for layer, saved_id in zip(self.layers, state_dict["param_groups"][0]["params"], strict=True):
            saved = state_dict["state"].get(saved_id, {})
            self.state[layer.weight] = {
                name: value.to(device=layer.weight.device, dtype=torch.float32, copy=True)
                for name, value in saved.items()
            }
        self._gather_row_moments_from_state()
        self.native.param_groups = self.param_groups[1:]
        self.refresh_projection()
