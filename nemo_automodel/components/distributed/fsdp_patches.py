# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

"""PyTorch FSDP compatibility patches owned by distributed infrastructure.

Every patch here is required on PyTorch 2.12 and returns without installing
once upstream FSDP2 covers the same case; each one checks exactly one upstream
attribute as its version boundary.
"""

from __future__ import annotations

import functools


def patch_fsdp_uniform_reduce_dtype() -> None:
    """Give every FSDP2 reduce-scatter group local gradients of one dtype.

    PyTorch 2.12 ``foreach_reduce`` asserts ``FSDP reduce-scatter expects
    uniform gradient dtype``. Gradient accumulation leaves a group holding
    ``reduce_dtype`` accumulations for the parameters used so far, while any
    parameter whose gradient joins later -- a locally unused parameter
    zero-filled by :func:`patch_fsdp_unused_param_reduction`, or one whose
    gradient lands after its group's post-backward already ran -- contributes
    ``param_dtype``. Widen the minority gradients to the promoted dtype right
    before ``foreach_reduce`` copies them into its ``reduce_dtype`` buffer: no
    value changes and FSDP2's own bookkeeping is left exactly as upstream leaves
    it. Uniform groups pass straight through.

    PyTorch >= 2.15 promotes the group's dtypes itself
    (``FSDPParamGroup._get_reduce_dtype``) and packs mixed gradients with
    ``chunk_cat_mixed_dtype``, so the patch returns without installing. The
    patch is process-global and idempotent.
    """
    try:
        import torch
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group
    except ImportError:
        return
    if hasattr(param_group.FSDPParamGroup, "_get_reduce_dtype"):
        return

    original_foreach_reduce = collectives.foreach_reduce
    if getattr(original_foreach_reduce, "_automodel_uniform_reduce_dtype", False):
        return
    # Anything outside these (integers, fp8) is passed through untouched so
    # PyTorch's own uniformity assertion still reports inconsistent gradients.
    widenable = {torch.float16, torch.bfloat16, torch.float32}

    def foreach_reduce_uniform_dtype(fsdp_params, unsharded_grads, *args, **kwargs):
        dtypes = {grad.dtype for grad in unsharded_grads}
        if len(dtypes) > 1 and dtypes <= widenable:
            target = functools.reduce(torch.promote_types, dtypes)
            # Mutate in place: ``foreach_reduce`` frees the gradients by clearing
            # this list, and that must still release the caller's references.
            unsharded_grads[:] = [grad if grad.dtype is target else grad.to(target) for grad in unsharded_grads]
        return original_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    foreach_reduce_uniform_dtype._automodel_uniform_reduce_dtype = True
    collectives.foreach_reduce = foreach_reduce_uniform_dtype
    param_group.foreach_reduce = foreach_reduce_uniform_dtype


def patch_fsdp_unused_param_reduction() -> None:
    """Backport FSDP2 unused-parameter reduction when the public API is absent.

    Fill a missing local gradient with zeros immediately before FSDP2
    post-backward reduction, so ranks that skipped a parameter still
    participate in the same collective as ranks that used it. PyTorch >= 2.15
    provides ``FSDPModule.set_reduce_scatter_unused_params`` (its zero goes
    through ``FSDPParam.unsharded_zero_grad_data``), so the patch returns
    without installing and callers enable the public API per unit instead. The
    patch is process-global and idempotent.

    Raises:
        RuntimeError: If the installed PyTorch exposes neither the public API
            nor the compatible private FSDP2 implementation.
    """
    try:
        import torch
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except ImportError as error:
        raise RuntimeError(
            "Context parallelism requires FSDP unused-parameter reduction, but this PyTorch "
            "version provides neither the public API nor the compatible FSDP2 implementation."
        ) from error
    if hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        return

    original_post_backward = FSDPParamGroup.post_backward
    if getattr(original_post_backward, "_automodel_reduce_scatter_unused_params", False):
        return

    def _post_backward_with_unused_param_reduction(self, *args, **kwargs):
        if self.reduce_grads and self._training_state == TrainingState.PRE_BACKWARD:
            for fsdp_param in self.fsdp_params:
                # Same lazy-state check as upstream post_backward: a unit that
                # never ran forward has no unsharded parameter to zero-fill.
                if not hasattr(fsdp_param, "_unsharded_param"):
                    continue
                if fsdp_param.unsharded_accumulated_grad is not None:
                    continue
                param = fsdp_param.unsharded_param
                if param.requires_grad and param.grad is None:
                    # ``zeros_like`` on the *unsharded* parameter is the dtype
                    # autograd would have produced under the unit's precision
                    # policy; ``patch_fsdp_uniform_reduce_dtype`` widens it if
                    # the group is already accumulating in ``reduce_dtype``.
                    param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        return original_post_backward(self, *args, **kwargs)

    _post_backward_with_unused_param_reduction._automodel_reduce_scatter_unused_params = True
    FSDPParamGroup.post_backward = _post_backward_with_unused_param_reduction


def patch_fsdp_accumulated_grad_guard() -> None:
    """Guard FSDP2 post-backward against params that were never unsharded.

    PyTorch 2.12 creates ``_unsharded_param`` lazily from an FSDP unit's
    forward pre-hook, and ``FSDPParam.to_accumulated_grad_if_needed``
    dereferences it unconditionally. If a separately wrapped unit is skipped
    by the batch (for example a vision tower on text-only data), deferred
    post-backward raises ``AttributeError`` there. Missing lazy state means
    there is no unsharded grad to upcast, so that exact case returns early.

    PyTorch >= 2.15 removed ``to_accumulated_grad_if_needed`` and reads the
    lazy field defensively, so the patch returns without installing.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except ImportError:
        return
    if not hasattr(FSDPParam, "to_accumulated_grad_if_needed"):
        return

    orig = FSDPParam.to_accumulated_grad_if_needed
    if getattr(orig, "_nemo_automodel_guarded", False):
        return

    def guarded(self):
        try:
            return orig(self)
        except AttributeError as exc:
            if "_unsharded_param" not in str(exc) or hasattr(self, "_unsharded_param"):
                raise
            return None

    setattr(guarded, "_nemo_automodel_guarded", True)
    # Preserve the previous Gemma4 marker name for callers/tests that only need
    # idempotency and do not care which entry point installed the patch.
    setattr(guarded, "_gemma4_guarded", True)
    FSDPParam.to_accumulated_grad_if_needed = guarded


__all__ = [
    "patch_fsdp_accumulated_grad_guard",
    "patch_fsdp_uniform_reduce_dtype",
    "patch_fsdp_unused_param_reduction",
]
