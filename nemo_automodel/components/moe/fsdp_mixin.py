# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Iterator, Optional

import torch
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining._backward import stage_backward, stage_backward_input, stage_backward_weight
from torch.nn.parallel import DistributedDataParallel

IS_OPTIM_STEP = False


def set_is_optim_step(value: bool) -> None:
    """Set the global IS_OPTIM_STEP flag.

    Args:
        value: Whether we are in an optimization step.
    """
    global IS_OPTIM_STEP
    IS_OPTIM_STEP = value


def get_is_optim_step() -> bool:
    """Get the global IS_OPTIM_STEP flag.

    Returns:
        Whether we are in an optimization step.
    """
    return IS_OPTIM_STEP


def _iter_fsdp_modules(module: torch.nn.Module) -> Iterator[FSDPModule]:
    # Check main model
    _model = module.model if hasattr(module, "model") else module
    if isinstance(_model, FSDPModule):
        yield _model

    # Check embeddings
    if hasattr(_model, "embed_tokens") and isinstance(_model.embed_tokens, FSDPModule):
        yield _model.embed_tokens

    # Check lm_head
    if hasattr(module, "lm_head") and isinstance(module.lm_head, FSDPModule):
        yield module.lm_head

    # Check experts in each layer
    if hasattr(_model, "layers"):
        for _, block in _model.layers.named_children():
            if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
                experts = block.mlp.experts
                if isinstance(experts, FSDPModule):
                    yield experts


def _configure_fsdp_module(
    fsdp_module: FSDPModule,
    *,
    is_last_backward: bool,
    reshard_after_backward: bool,
    requires_gradient_sync: bool,
) -> None:
    fsdp_module.set_is_last_backward(is_last_backward)
    fsdp_module.set_reshard_after_backward(reshard_after_backward)
    fsdp_module.set_requires_gradient_sync(requires_gradient_sync)


def _run_post_backward_hooks(fsdp_module: FSDPModule) -> Callable:
    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
    for state in fsdp_state._state_ctx.all_states:
        if state._fsdp_param_group:
            state._fsdp_param_group.post_backward()
    return fsdp_state._root_post_backward_final_callback


class MoEFSDPSyncMixin:
    """
    Mixin for managing FSDP synchronization state during MoE model training.

    Controls gradient sync and resharding for FSDP-wrapped modules to optimize
    performance during gradient accumulation steps.

    Usage differs based on parallelism strategy:
    - Without pipeline parallelism (PP): prepare_for_grad_accumulation() defers sync and
      resharding at the start of gradient accumulation. prepare_for_final_backward() enables
      sync and resharding before the last backward pass. FSDP's autograd hooks automatically
      handle post-backward synchronization and resharding.
    - With pipeline parallelism (PP): FSDP state management is handled by patching
      _PipelineStageBase.backward_maybe_with_nosync (see patched_backward_maybe_with_nosync
      below). The patch disables sync/resharding for all backwards except the last
      one before optimizer step, where it manually triggers post-backward hooks and resharding.
    """

    def prepare_for_grad_accumulation(self, pp_enabled: bool = False) -> None:
        """Prepare FSDP states before starting gradient accumulation.

        Args:
            pp_enabled: Whether pipeline parallelism is enabled.

        Note:
            When PP is enabled, FSDP state management is handled by the patched
            _PipelineStageBase.backward_maybe_with_nosync method.
            This method only applies optimizations for non-PP cases.
        """
        if not self.backend.enable_fsdp_optimizations:
            return

        for fsdp_module in _iter_fsdp_modules(self):
            _configure_fsdp_module(
                fsdp_module,
                is_last_backward=False,
                reshard_after_backward=False,
                requires_gradient_sync=False,
            )

    def prepare_for_final_backward(self, pp_enabled: bool = False) -> None:
        """Enable gradient sync and resharding for the final backward pass.

        Args:
            pp_enabled: Whether pipeline parallelism is enabled.

        Note:
            When PP is enabled, FSDP state management is handled by the patched
            _PipelineStageBase.backward_maybe_with_nosync method.
            This method only applies optimizations for non-PP cases.
        """
        if not self.backend.enable_fsdp_optimizations:
            return

        for fsdp_module in _iter_fsdp_modules(self):
            _configure_fsdp_module(
                fsdp_module,
                is_last_backward=True,
                reshard_after_backward=True,
                requires_gradient_sync=True,
            )


#############################################################
# MoE-specific FSDP state management for pipeline parallelism
#############################################################


def _disable_fsdp_for_moe_module(module: torch.nn.Module) -> None:
    for fsdp_module in _iter_fsdp_modules(module):
        _configure_fsdp_module(
            fsdp_module,
            is_last_backward=False,
            reshard_after_backward=False,
            requires_gradient_sync=False,
        )


def _run_post_backward_for_moe_module(module: torch.nn.Module) -> None:
    fsdp_modules = list(_iter_fsdp_modules(module))

    # Enable sync for all modules
    for fsdp_module in fsdp_modules:
        _configure_fsdp_module(
            fsdp_module,
            is_last_backward=True,
            reshard_after_backward=True,
            requires_gradient_sync=True,
        )

    # Run post-backward hooks
    root_callbacks = []
    for fsdp_module in fsdp_modules:
        root_callbacks.append(_run_post_backward_hooks(fsdp_module))

    for root_callback in root_callbacks:
        root_callback()


def patched_backward_maybe_with_nosync(
    self,
    backward_type,
    bwd_kwargs: dict,
    last_backward: bool = False,
) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
    """
    Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
    other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
    there are additional state-variables and performance considerations depending on the data parallelism used.
    This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
    """

    def perform_backward(
        backward_type,
    ) -> Callable[
        [],
        tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
    ]:
        if backward_type == "full":
            return lambda: (
                stage_backward(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                ),
                None,
            )
        elif backward_type == "input":
            return lambda: stage_backward_input(
                bwd_kwargs["stage_output"],
                bwd_kwargs["output_grads"],
                bwd_kwargs["input_values"],
                self.submod.parameters(),
            )
        elif backward_type == "weight":
            return lambda: (
                stage_backward_weight(self.submod.parameters(), bwd_kwargs["param_groups"]),
                None,
            )
        else:
            raise RuntimeError(f"Unknown backward type: {backward_type}")

    # If submod is wrapped by DDP
    if isinstance(self.submod, DistributedDataParallel):
        if last_backward:
            # Last chunk, prepare for gradient reduction
            # HACK: reaching into DDP implementation details here. Is there a better way?
            self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                list(
                    torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                        bwd_kwargs["stage_output"]
                    )
                )
            )
            result = perform_backward(backward_type)()
        else:
            with self.submod.no_sync():  # type: ignore[operator]
                result = perform_backward(backward_type)()
    # If submod is a FSDP module
    elif isinstance(self.submod, FSDPModule):
        self.submod.set_is_last_backward(False)
        self.submod.set_reshard_after_backward(False)
        self.submod.set_requires_gradient_sync(False)
        result = perform_backward(backward_type)()
        if last_backward:
            # Manually call post backward for FSDP
            def run_post_backward(fsdp_module: FSDPModule) -> None:
                fsdp_module.set_is_last_backward(True)
                fsdp_module.set_reshard_after_backward(True)
                fsdp_module.set_requires_gradient_sync(True)
                fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
                for state in fsdp_state._state_ctx.all_states:
                    if state._fsdp_param_group:
                        state._fsdp_param_group.post_backward()

                # it would be much better if pipelining backward invoked .backward so autograd hooks
                # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
                # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
                fsdp_state._root_post_backward_final_callback()

            run_post_backward(self.submod)
    # If submod is a MoEFSDPSyncMixin, use the MoE-specific FSDP functions
    elif isinstance(self.submod, MoEFSDPSyncMixin):
        _disable_fsdp_for_moe_module(self.submod)
        result = perform_backward(backward_type)()
        if last_backward and get_is_optim_step():
            _run_post_backward_for_moe_module(self.submod)
    else:
        # Non-DP submodule, regular backward
        result = perform_backward(backward_type)()

    grads, param_groups = result
    return grads, param_groups
