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

"""A small training engine for an already-distributed eager model."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.utils import (
    get_expert_tp_replication_factor,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.shared.import_utils import MISSING_TORCHAO_MSG, safe_import_from


def _resolve_fp8_scale_precompute(module: nn.Module) -> Callable[[nn.Module], None] | None:
    """Resolve torchao's post-step helper when the module opts into it."""
    if not getattr(module, "precompute_float8_dynamic_scale_for_fsdp", False):
        return None

    available, precompute = safe_import_from(
        "torchao.float8",
        "precompute_float8_dynamic_scale_for_fsdp",
        msg=MISSING_TORCHAO_MSG,
    )
    if not available:
        raise ImportError(MISSING_TORCHAO_MSG)
    return precompute


def _uses_summed_gradient_reduction(module: nn.Module) -> bool:
    """Detect MegatronFSDP's summed-gradient reduction contract.

    MegatronFSDP publishes ``calculate_per_token_loss`` on its wrapper. Walk
    the module tree as well so ordinary wrapper layers do not hide that
    declaration. Modules without a declaration use PyTorch's usual averaged
    distributed-gradient contract.
    """
    sentinel = object()
    declared_modes: set[bool] = set()
    for child in module.modules():
        mode = getattr(child, "calculate_per_token_loss", sentinel)
        if mode is not sentinel:
            declared_modes.add(bool(mode))

    if len(declared_modes) > 1:
        raise ValueError(
            "model mixes calculate_per_token_loss=True and False; "
            "Engine requires one gradient-reduction mode per optimizer window"
        )
    return declared_modes == {True}


class Engine(nn.Module):
    """Train an already-distributed eager model.

    The interface intentionally mirrors a regular PyTorch module plus the
    training operations exposed by DeepSpeed: callers prepare batches and
    losses, while the engine owns backward, gradient accumulation, distributed
    gradient finalization, optimizer updates, and scheduler advancement. The
    engine preserves the usual averaged-gradient backward semantics, while
    compensating when a backend instead sums distributed gradients.

    Args:
        module: Model whose forward method is exposed by this engine. It must
            already be parallelized and wrapped for distributed training.
        optimizer: Optimizer for ``module``. A forward-only engine may omit it.
        lr_scheduler: Optional scheduler advanced after each optimizer update.
        mesh_context: Runtime topology used to finalize distributed gradients.
        max_grad_norm: Maximum global gradient norm. ``None`` disables clipping
            while preserving distributed expert-gradient finalization.
        gradient_accumulation_steps: Number of forward/backward microsteps per
            optimizer update.
        defer_fsdp_grad_sync: Defer FSDP gradient synchronization until the
            final microstep in an accumulation window. Wrappers exposing a
            ``no_sync`` context use it on non-final microsteps.

    Note:
        Batch collation, device transfer, context-parallel input sharding,
        packing, loss normalization, and output restoration belong to the
        caller. Pipeline execution has a different scheduling contract and is
        not supported by this eager engine.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: OptimizerParamScheduler | torch.optim.lr_scheduler.LRScheduler | None = None,
        mesh_context: MeshContext | None = None,
        max_grad_norm: float | None = 1.0,
        gradient_accumulation_steps: int = 1,
        defer_fsdp_grad_sync: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(module, AutoPipeline):
            raise NotImplementedError(
                "Engine supports eager modules only; execute AutoPipeline with its pipeline schedule"
            )
        if mesh_context is not None and getattr(mesh_context, "pp_enabled", False):
            raise NotImplementedError(
                "Engine supports eager modules only; execute pipeline stages with their pipeline schedule"
            )
        if not isinstance(module, nn.Module):
            raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
        if max_grad_norm is not None and max_grad_norm < 0:
            raise ValueError(f"max_grad_norm must be non-negative or None, got {max_grad_norm}")

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mesh_context = mesh_context
        self.max_grad_norm = max_grad_norm
        self.defer_fsdp_grad_sync = defer_fsdp_grad_sync

        self._gradient_accumulation_steps = 1
        self._micro_step = 0
        self._backward_context: AbstractContextManager[Any] | None = None
        self._global_grad_norm: torch.Tensor | float | None = None
        self._precompute_fp8_scale_fn = _resolve_fp8_scale_precompute(module)
        self._summed_gradient_reduction = _uses_summed_gradient_reduction(module)
        self.set_gradient_accumulation_steps(gradient_accumulation_steps)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate a raw forward call to :attr:`module`."""
        if not self._will_run_backward():
            return self.module(*args, **kwargs)
        if self._backward_context is not None:
            raise RuntimeError("call Engine.backward() before starting another training forward")

        if self._micro_step == 0:
            prepare_for_grad_accumulation([self.module], pp_enabled=False)
        if self.is_gradient_accumulation_boundary():
            prepare_for_final_backward([self.module], pp_enabled=False)

        context = get_sync_ctx(
            self.module,
            self.is_gradient_accumulation_boundary(),
            defer_fsdp_grad_sync=self.defer_fsdp_grad_sync,
        )
        context.__enter__()
        try:
            output = self.module(*args, **kwargs)
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        self._backward_context = context
        return output

    def backward(
        self,
        loss: torch.Tensor,
        retain_graph: bool = False,
        scale_wrt_gas: bool = True,
    ) -> None:
        """Backpropagate a scalar loss.

        Args:
            loss: Scalar loss to backpropagate. Its scale follows the usual
                averaged-gradient distributed-training convention.
            retain_graph: Preserve the autograd graph after backward.
            scale_wrt_gas: Divide ``loss`` by the configured gradient
                accumulation steps. Disable this when the caller has already
                normalized the complete accumulation window.
        """
        if self.optimizer is None:
            self._close_backward_context()
            raise RuntimeError("Engine.backward requires an optimizer")
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            self._close_backward_context()
            raise ValueError(f"loss must be a scalar Tensor, got {loss!r}")

        gradient_group_size = self._gradient_group_size()
        reduction_compensation = self._gradient_reduction_compensation(gradient_group_size)
        scale = reduction_compensation
        if scale_wrt_gas:
            scale /= self._gradient_accumulation_steps
        # The auxiliary loss is injected by its own autograd node on every
        # microstep, so it always needs accumulation scaling even when the
        # caller has already normalized the main loss for the full window. Its
        # local mean also needs the same reducer compensation.
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
            self._context_parallel_size() * reduction_compensation / self._gradient_accumulation_steps,
            device=loss.device,
        )
        try:
            (loss * scale).backward(retain_graph=retain_graph)
        finally:
            self._close_backward_context(*sys.exc_info())

        if self._micro_step == 0:
            prepare_after_first_microbatch()

    @torch.no_grad()
    def step(self) -> None:
        """Advance one microstep and update parameters at its GAS boundary."""
        if self.optimizer is None:
            raise RuntimeError("Engine.step requires an optimizer")
        if self._backward_context is not None:
            raise RuntimeError("call Engine.backward() before Engine.step()")

        if not self.is_gradient_accumulation_boundary():
            self._micro_step += 1
            return

        device_mesh = self.mesh_context.device_mesh if self.mesh_context is not None else None
        moe_mesh = self.mesh_context.moe_mesh if self.mesh_context is not None else None
        self._global_grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=self.max_grad_norm,
            model_parts=[self.module],
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            ep_axis_name="ep" if moe_mesh is not None and "ep" in (moe_mesh.mesh_dim_names or ()) else None,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=None,
            dp_group_size=self._gradient_group_size(),
            expert_tp_replication_factor=get_expert_tp_replication_factor([self.module], device_mesh),
        )

        self.optimizer.step()
        self.zero_grad()

        update_moe_gate_bias = getattr(self.module, "update_moe_gate_bias", None)
        if callable(update_moe_gate_bias):
            update_moe_gate_bias()

        if self._precompute_fp8_scale_fn is not None:
            self._precompute_fp8_scale_fn(self.module)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._micro_step = 0

    def zero_grad(self) -> None:
        """Clear gradients through the optimizer, or through the module if absent."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.module.zero_grad(set_to_none=True)

    def set_gradient_accumulation_steps(self, steps: int) -> None:
        """Set the number of microsteps in the next optimizer window."""
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(f"gradient accumulation steps must be a positive integer, got {steps!r}")
        if self._micro_step != 0:
            raise RuntimeError("gradient accumulation steps cannot change during an active window")
        self._gradient_accumulation_steps = steps

    def is_gradient_accumulation_boundary(self) -> bool:
        """Return whether the current microstep completes the optimizer window."""
        return self._micro_step + 1 == self._gradient_accumulation_steps

    def get_global_grad_norm(self) -> torch.Tensor | float | None:
        """Return the global gradient norm measured at the latest optimizer step."""
        return self._global_grad_norm

    def _will_run_backward(self) -> bool:
        return self.optimizer is not None and self.module.training and torch.is_grad_enabled()

    def _close_backward_context(self, *exc_info: Any) -> None:
        context, self._backward_context = self._backward_context, None
        if context is not None:
            context.__exit__(*exc_info if exc_info else (None, None, None))

    def _context_parallel_size(self) -> int:
        return self.mesh_context.cp_size if self.mesh_context is not None else 1

    def _gradient_group_size(self) -> int:
        if self.mesh_context is not None and self.mesh_context.device_mesh is not None:
            axis = "dp_cp" if self._context_parallel_size() > 1 else "dp"
            return int(get_flat_mesh(self.mesh_context.device_mesh, axis).size())

        group = self.mesh_context.process_group if self.mesh_context is not None else None
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(group=group)
        return 1

    def _gradient_reduction_compensation(self, gradient_group_size: int) -> float:
        """Preserve averaged-reducer semantics for summed-gradient backends."""
        return 1.0 / gradient_group_size if self._summed_gradient_reduction else 1.0
