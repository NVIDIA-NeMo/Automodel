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

import logging
import os
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import torch
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.distributed.multimodal_fsdp import iter_multimodal_modules
from nemo_automodel.components.models.common.utils import get_is_optim_step

logger = logging.getLogger(__name__)


@contextmanager
def _one_graph_task_for_checkpoint_recompute() -> Iterator[None]:
    """Report one graph-task id to activation checkpointing for every ``autograd.grad`` call inside.

    EXPERIMENT. PyTorch's ``stage_backward_weight`` runs one ``autograd.grad`` per parameter group,
    each a new graph task, and a non-reentrant checkpoint recomputes its region once per graph task
    that unpacks it: a checkpointed block with N parameter groups is recomputed N times in the weight
    pass. With one id for the whole weight pass the first group recomputes the block and the others
    read the tensors it stored (each saved tensor is still unpacked once). One ``autograd.grad`` over
    all groups is not equivalent: a group's intermediate also feeds the intermediates below it.
    """
    fake_gid = uuid.uuid4().int >> 65
    real = torch._C._current_graph_task_id
    torch._C._current_graph_task_id = lambda: fake_gid
    try:
        yield
    finally:
        torch._C._current_graph_task_id = real


def stage_backward_weight_one_recompute(
    weights: Iterator[torch.nn.Parameter], param_groups: list[dict[str, Any]]
) -> tuple[torch.Tensor | None, ...]:
    """``stage_backward_weight`` that recomputes each checkpointed region once (see above)."""
    with _one_graph_task_for_checkpoint_recompute():
        return stage_backward_weight(weights, param_groups)


def _iter_fsdp_modules(module: torch.nn.Module) -> Iterator[FSDPModule]:
    """Yield every FSDP unit whose backward state this mixin must drive, deduplicated.

    A unit that is not yielded here never receives ``set_is_last_backward`` /
    ``set_reshard_after_backward`` / ``set_requires_gradient_sync``, so it drops
    out of the gradient-accumulation state machine and, under pipeline
    parallelism, out of ``patched_backward_maybe_with_nosync``.
    """
    seen: set[int] = set()
    for candidate in _iter_fsdp_module_candidates(module):
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate


def _iter_fsdp_module_candidates(module: torch.nn.Module) -> Iterator[FSDPModule]:
    if isinstance(module, FSDPModule):
        yield module

    # Check main model
    _model = module.model if hasattr(module, "model") else module
    if _model is not module and isinstance(_model, FSDPModule):
        yield _model

    # Check embeddings
    if hasattr(_model, "embed_tokens") and isinstance(_model.embed_tokens, FSDPModule):
        yield _model.embed_tokens

    # Check lm_head
    if hasattr(module, "lm_head") and isinstance(module.lm_head, FSDPModule):
        yield module.lm_head

    # Multimodal towers/projectors. ``moe/parallelizer.apply_fsdp`` gives every
    # recognized *trainable* multimodal module its own FSDP unit, and under the
    # ``per_layer`` frozen policy it shards their layer containers instead of the
    # module itself -- so walk the module and its descendants rather than testing
    # only the top-level attribute. Uses the same shared taxonomy as the
    # parallelizer so unit creation and unit enumeration cannot drift apart:
    # e.g. Kimi-K2.5-VL LoRA (`examples/vlm_benchmark/kimi/kimi25vl_lora.yaml`,
    # target_modules ["*"] -> trainable vision_tower, wrap_outer_model=false, pp_size 4)
    # previously created a vision_tower root that this function never found.
    for _, mm_module in iter_multimodal_modules(module):
        subs = mm_module.modules() if isinstance(mm_module, torch.nn.Module) else (mm_module,)
        for sub in subs:
            if isinstance(sub, FSDPModule):
                yield sub

    # Check experts in each layer (Qwen-style: block.mlp.experts; Gemma4-style: block.moe.experts)
    if hasattr(_model, "layers"):
        for _, block in _model.layers.named_children():
            for attr in ("mlp", "moe"):
                mod = getattr(block, attr, None)
                if mod is not None and hasattr(mod, "experts"):
                    experts = mod.experts
                    if isinstance(experts, FSDPModule):
                        yield experts

    # Check MTP blocks. They are registered on the outer model rather than the
    # text backbone but participate in the same backward pass and FSDP state
    # transitions.
    mtp = getattr(module, "mtp", None)
    mtp_layers = getattr(mtp, "layers", None)
    if mtp_layers is not None:
        for _, block in mtp_layers.named_children():
            if isinstance(block, FSDPModule):
                yield block
            for attr in ("mlp", "moe"):
                mod = getattr(block, attr, None)
                if mod is not None and hasattr(mod, "experts"):
                    experts = mod.experts
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


def _fsdp_backward_finalized(fsdp_state: Any) -> bool:
    """Return whether FSDP has no backward iteration left to finalize for this root state context.

    PyTorch 2.15 (NGC 26.11) records the backward finalization mode when FSDP sees a backward
    start and clears it when the root post-backward final callback ends the iteration; the
    callback asserts that the mode is set. The FSDP units of one model share their root's
    state context, so once one unit's callback has finalized, the others find the mode
    cleared, as does a manual callback after FSDP's own autograd callback finalized a
    synchronized last backward. A root that no backward reached has nothing to reduce either.
    Earlier PyTorch has no such field and accepts repeated callbacks.

    Args:
        fsdp_state: The ``FSDPState`` of an FSDP unit.

    Returns:
        True when calling ``_root_post_backward_final_callback`` must be skipped.
    """
    return getattr(fsdp_state._state_ctx, "manual_backward_finalization_active", False) is None


def _run_post_backward_hooks(fsdp_module: FSDPModule) -> Callable:
    """Run post-backward for every FSDP state and return the root final callback.

    Both the per-state ``post_backward()`` calls and the returned callback
    tolerate FSDP modules that never ran forward in this step. With
    ``FusedLinearCrossEntropy`` under pipeline parallelism the ``lm_head``
    weight is consumed inside the loss function, so its FSDP module never
    runs forward and its ``FSDPCommContext`` is never lazily initialized;
    ``post_backward()`` then raises ``AttributeError`` (``'FSDPCommContext'
    object has no attribute 'post_forward_order'``). A never-forwarded group
    has no gradients to reduce, so it is safe to skip.
    """
    fsdp_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]
    for state in fsdp_state._state_ctx.all_states:
        if state._fsdp_param_group:
            try:
                state._fsdp_param_group.post_backward()
            except AttributeError as exc:
                if "post_forward_order" not in str(exc):
                    raise
                logger.debug("Skipping post_backward for never-forwarded FSDP group: %s", exc)
                continue

    def _final_callback() -> None:
        if _fsdp_backward_finalized(fsdp_state):
            return
        try:
            fsdp_state._root_post_backward_final_callback()
        except AttributeError as exc:
            if "post_forward_order" not in str(exc):
                raise
            logger.debug("Skipping root post-backward callback for never-forwarded FSDP root: %s", exc)

    return _final_callback


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
      below). By default (``distributed.defer_fsdp_grad_sync: true``) the patch disables
      sync/resharding for all backwards except the last one before the optimizer step, where
      it manually triggers post-backward hooks and resharding. With
      ``defer_fsdp_grad_sync: false`` every micro-batch reduce-scatters its own gradients
      (after its full or weight backward); the input backward of a split schedule never
      reduces or reshards, and resharding/finalization is deferred to the last backward so
      that pending split weight backwards still find the unsharded parameters.
    """

    # fp32 holders whose forward ignores its input (dotted names relative to a decoder block, e.g.
    # "attn.sinks_param"): one FSDP unit across all blocks instead of one per block — see
    # parallelizer_utils.fully_shard_across_blocks. Models override it; the parallelizer reads it with getattr so
    # models that do not inherit this mixin keep working.
    _fsdp_shard_across_blocks: tuple[str, ...] = ()

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


def _reduce_fsdp_for_moe_module(module: torch.nn.Module, *, last_backward: bool) -> None:
    """Let FSDP's hooks reduce-scatter this micro-batch's gradients; reshard and finalize only on the last backward.

    Used for a full backward under ``distributed.defer_fsdp_grad_sync: false``. A split
    weight backward for an earlier micro-batch may still be pending (mixed schedules such
    as ``F0 I0 F1 B1 W0``) and looks the parameters up by their unsharded objects, so the
    parameters must stay unsharded until the actual last backward.
    """
    for fsdp_module in _iter_fsdp_modules(module):
        _configure_fsdp_module(
            fsdp_module,
            is_last_backward=last_backward,
            reshard_after_backward=last_backward,
            requires_gradient_sync=True,
        )


def _run_post_backward_for_moe_module(module: torch.nn.Module, *, last_backward: bool = True) -> None:
    """Run FSDP's post-backward (reduce-scatter) manually for every FSDP unit of a MoE module.

    With ``last_backward`` (the default) the units are also resharded and the root final
    callback finalizes the step; otherwise the parameters stay unsharded for the backwards
    still to come and only the gradients are reduced.
    """
    fsdp_modules = list(_iter_fsdp_modules(module))

    # Enable sync for all modules
    for fsdp_module in fsdp_modules:
        _configure_fsdp_module(
            fsdp_module,
            is_last_backward=last_backward,
            reshard_after_backward=last_backward,
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
) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None]:
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
        tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None],
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
                (
                    stage_backward_weight_one_recompute
                    if os.environ.get("EXP_ZB_SINGLE_W", "0") == "1"
                    else stage_backward_weight
                )(self.submod.parameters(), bwd_kwargs["param_groups"]),
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
        if getattr(self, "_reduce_grad_per_microbatch", False):
            self.submod.set_is_last_backward(True)
            self.submod.set_reshard_after_backward(True)
            self.submod.set_requires_gradient_sync(True)
        else:
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
                if not _fsdp_backward_finalized(fsdp_state):
                    fsdp_state._root_post_backward_final_callback()

            run_post_backward(self.submod)
    # If submod is a MoEFSDPSyncMixin, use the MoE-specific FSDP functions
    elif isinstance(self.submod, MoEFSDPSyncMixin):
        if getattr(self, "_reduce_grad_per_microbatch", False):
            # Honour ``distributed.defer_fsdp_grad_sync: false``: every micro-batch
            # reduce-scatters its own gradients, so no unsharded ``reduce_dtype``
            # gradient accumulator survives across micro-batches (the deferred branch
            # below keeps one alive for the whole step).
            if backward_type == "input":
                # Split backward (zero-bubble schedules): the input backward produces no
                # parameter gradients yet, and ``stage_backward_input`` has recorded the
                # ``AccumulateGrad`` nodes of the *unsharded* parameters for the matching
                # weight backward. Keep sync and resharding off so FSDP neither reduces
                # nor replaces those parameters before ``stage_backward_weight`` runs.
                _disable_fsdp_for_moe_module(self.submod)
                result = perform_backward(backward_type)()
            elif backward_type == "weight":
                # Keep FSDP out of the whole weight helper: ``stage_backward_weight`` runs
                # one ``autograd.grad`` per parameter group, and an activation-checkpoint
                # recompute inside a later group must still see the unsharded parameters,
                # so nothing may reduce or reshard until the helper returns. Then
                # reduce-scatter this micro-batch's gradients manually (FSDP's own hooks
                # hang off the forward graph and do not fire for a weight-only backward);
                # resharding waits for the last backward because the weight backwards
                # still queued for the other micro-batches look the parameters up by the
                # same unsharded objects.
                _disable_fsdp_for_moe_module(self.submod)
                result = perform_backward(backward_type)()
                _run_post_backward_for_moe_module(self.submod, last_backward=last_backward)
            else:
                # ``full``: reduce-scatter through FSDP's hooks. Resharding and finalizing
                # wait for the last backward, since a split weight backward of an earlier
                # micro-batch may still be pending; on the last backward run the
                # post-backward manually so the reduction is synced back to the default
                # stream, as in the FSDPModule branch above.
                _reduce_fsdp_for_moe_module(self.submod, last_backward=last_backward)
                result = perform_backward(backward_type)()
                if last_backward:
                    _run_post_backward_for_moe_module(self.submod)
        else:
            _disable_fsdp_for_moe_module(self.submod)
            result = perform_backward(backward_type)()
            if last_backward and get_is_optim_step():
                _run_post_backward_for_moe_module(self.submod)
    else:
        # Non-DP submodule, regular backward
        result = perform_backward(backward_type)()

    grads, param_groups = result
    return grads, param_groups
