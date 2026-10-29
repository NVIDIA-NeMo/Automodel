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

"""PyTorch FSDP compatibility patches owned by distributed infrastructure."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from nemo_automodel.shared.import_utils import get_torch_version

# torch 2.15 removed FSDPParam.to_accumulated_grad_if_needed: FSDP2 sets ``grad_dtype`` on the
# unsharded param, so autograd accumulates in the reduce dtype and there is no grad to upcast.
_FSDP_ACCUMULATED_GRAD_REMOVED_VERSION = (2, 15)


def _widest_float_dtype(dtypes: Iterable[Any]) -> Any:
    """Return the float dtype among ``dtypes`` that every other one converts into losslessly.

    Args:
        dtypes: Gradient dtypes from a single reduce-scatter group.

    Returns:
        The dtype with the largest element size; ties resolve to float32 over
        the 2-byte float types.
    """
    import torch

    return max(dtypes, key=lambda dtype: (torch.finfo(dtype).bits, dtype is torch.float32))


def patch_fsdp_uniform_reduce_dtype() -> None:
    """Give every FSDP2 reduce-scatter group local gradients of one dtype.

    Gradient accumulation leaves a group holding ``reduce_dtype`` accumulations
    for the parameters used so far, while any parameter whose gradient joins
    later -- a locally unused parameter zero-filled by PyTorch's public API or
    :func:`patch_fsdp_unused_param_reduction`, or one whose gradient lands after
    its group's post-backward already ran -- contributes ``param_dtype``.
    ``foreach_reduce`` then aborts with ``FSDP reduce-scatter expects uniform
    gradient dtype``.

    Normalize and widen gradients at the last possible moment, inside
    ``foreach_reduce`` itself. That placement matters:

    * ``FSDPParam`` normally unwraps gradients through
      ``_get_grad_inner_tensor``. PyTorch versions whose public unused-parameter
      API appends ``zeros_like(unsharded_param)`` directly can still leave a
      ``DTensor`` in this list, so unwrap that residual value before sizing the
      reduce-scatter buffer;
    * FSDP2's own bookkeeping (``unsharded_param.grad`` /
      ``unsharded_accumulated_grad``) is left exactly as upstream leaves it, so
      no later reader of that state sees anything unusual;
    * ``foreach_reduce`` immediately copies these gradients into a
      ``reduce_dtype`` buffer anyway, so widening first changes no value.

    Uniform groups are passed straight through, so the upstream assertion still
    fires for genuinely inconsistent gradients such as fp8 weights that fail to
    produce higher-precision ones. The patch is process-global and idempotent.
    """
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group
    except ImportError:
        return

    original_foreach_reduce = collectives.foreach_reduce
    if getattr(original_foreach_reduce, "_automodel_uniform_reduce_dtype", False):
        return

    def foreach_reduce_uniform_dtype(fsdp_params, unsharded_grads, *args, **kwargs):
        from torch.distributed.tensor import DTensor

        # PyTorch 2.13a0's unused-parameter branch can append a DTensor zero
        # directly, while gradients from used parameters are already local
        # tensors. Besides making ``fsdp.chunk_cat`` reject the mixed list, the
        # DTensor's global numel makes FSDP size a global-shape staging buffer.
        # Current PyTorch routes the zero through ``_get_grad_inner_tensor``;
        # localizing here is the equivalent compatibility path for that build.
        unsharded_grads[:] = [grad.to_local() if isinstance(grad, DTensor) else grad for grad in unsharded_grads]
        dtypes = {grad.dtype for grad in unsharded_grads}
        if len(dtypes) > 1 and all(dtype.is_floating_point for dtype in dtypes):
            target = _widest_float_dtype(dtypes)
            # Mutate in place: ``foreach_reduce`` frees the gradients by clearing
            # this list, and that must still release the caller's references.
            unsharded_grads[:] = [grad if grad.dtype is target else grad.to(target) for grad in unsharded_grads]
        return original_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    foreach_reduce_uniform_dtype._automodel_uniform_reduce_dtype = True
    collectives.foreach_reduce = foreach_reduce_uniform_dtype
    param_group.foreach_reduce = foreach_reduce_uniform_dtype


def _has_local_unsharded_grad(fsdp_param: Any) -> bool:
    """Whether this rank's backward produced a gradient for ``fsdp_param`` since its last reduction."""
    if not hasattr(fsdp_param, "_unsharded_param"):
        return False
    return fsdp_param.unsharded_accumulated_grad is not None or fsdp_param.unsharded_param.grad is not None


def patch_fsdp_unused_param_reduction() -> None:
    """Backport FSDP2 unused-parameter reduction when the public API is absent.

    The patch is process-global and idempotent. It only fills a missing local
    gradient with zeros immediately before FSDP2 post-backward reduction, so
    ranks that skipped a parameter still participate in the same collective as
    ranks that used it. Callers must first prefer the public
    ``FSDPModule.set_reduce_scatter_unused_params`` API.

    Raises:
        RuntimeError: If the installed PyTorch exposes neither the public API
            nor the compatible private FSDP2 implementation.
    """
    try:
        import torch
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except ImportError as error:
        raise RuntimeError(
            "Context parallelism requires FSDP unused-parameter reduction, but this PyTorch "
            "version provides neither the public API nor the compatible FSDP2 implementation."
        ) from error

    original_post_backward = FSDPParamGroup.post_backward
    if getattr(original_post_backward, "_automodel_reduce_scatter_unused_params", False):
        return
    if getattr(original_post_backward, "_automodel_unused_param_probe", False):
        raise RuntimeError(
            "patch_fsdp_unused_param_reduction must be installed before patch_fsdp_restore_unused_param_grads: "
            "a zero fill wrapped outside the usage probe would make every parameter look used."
        )

    def _post_backward_with_unused_param_reduction(self, *args, **kwargs):
        if self.reduce_grads and self._training_state == TrainingState.PRE_BACKWARD:
            for fsdp_param in self.fsdp_params:
                if not hasattr(fsdp_param, "_unsharded_param") or _has_local_unsharded_grad(fsdp_param):
                    continue
                param = fsdp_param.unsharded_param
                if param.requires_grad:
                    # ``zeros_like`` on the *unsharded* parameter is the dtype
                    # autograd would have produced under any precision policy
                    # (bf16 compute over fp32 storage, an fp32-pinned unit, or no
                    # casting at all). ``_align_accumulated_grad_dtype`` then
                    # promotes it to ``reduce_dtype`` if the group is accumulating.
                    param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        return original_post_backward(self, *args, **kwargs)

    _post_backward_with_unused_param_reduction._automodel_reduce_scatter_unused_params = True
    FSDPParamGroup.post_backward = _post_backward_with_unused_param_reduction


def _fsdp_state_param_groups(state: Any) -> list[Any]:
    """Return an FSDP2 state's parameter groups across PyTorch versions that keep one or several."""
    groups = getattr(state, "_fsdp_param_groups", None)
    if groups is not None:
        return list(groups)
    group = getattr(state, "_fsdp_param_group", None)
    return [group] if group is not None else []


def _restore_unused_param_grads(param_groups: list[Any]) -> None:
    """Reset the gradients that a reduction created for parameters no rank of its domain used.

    Args:
        param_groups: FSDP2 parameter groups holding a usage probe from this backward, in the same
            order on every rank of their reduction domains.
    """
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo, FSDPMeshInfo

    entries_by_domain: dict[tuple[Any, Any, Any], list[tuple[Any, bool, bool]]] = {}
    for group in param_groups:
        reduce_scatter_group = (
            group._reduce_scatter_process_group if isinstance(group.mesh_info, FSDPMeshInfo) else None
        )
        all_reduce_group = group._all_reduce_process_group if isinstance(group.mesh_info, DDPMeshInfo) else None
        domain = (reduce_scatter_group, all_reduce_group, group.device)
        entries_by_domain.setdefault(domain, []).extend(group._automodel_unused_param_probe)
        group._automodel_unused_param_probe = None

    # Issue every domain's reduction before reading any back, so the backward pays one host sync.
    used_by_domain = []
    for (reduce_scatter_group, all_reduce_group, device), entries in entries_by_domain.items():
        used = torch.tensor([used_locally for _, used_locally, _ in entries], dtype=torch.int32, device=device)
        for process_group in (reduce_scatter_group, all_reduce_group):
            if process_group is not None:
                dist.all_reduce(used, op=dist.ReduceOp.MAX, group=process_group)
        used_by_domain.append((entries, used))
    for entries, used in used_by_domain:
        for (fsdp_param, _, grad_was_none), used_anywhere in zip(entries, used.tolist()):
            if grad_was_none and not used_anywhere:
                fsdp_param.sharded_param.grad = None


def patch_fsdp_restore_unused_param_grads() -> None:
    """Keep ``grad is None`` for parameters that no rank of their reduction used.

    The unused-parameter fill (PyTorch's public API or
    :func:`patch_fsdp_unused_param_reduction`) gives a locally unused parameter a
    zero contribution so every rank issues the same reduce-scatter. When no rank
    of the parameter's reduction domain used it, that reduction turns a missing
    gradient into an exact zero, which an optimizer treats as a real gradient:
    AdamW advances its step, applies decoupled weight decay and keeps moving the
    weight with residual momentum. Without the fill the gradient stays ``None``
    and the optimizer skips the parameter.

    Each reducing post-backward records which parameters had a local gradient,
    and the root's final backward callback reduces those flags over the reduction
    domain and resets the sharded gradients this reduction created for parameters
    no rank used. Usage comes from gradient presence, never from a zero value.
    Install it after the fill so the probe runs before the zero fill; the patch is
    process-global and idempotent. Under HSDP with a deferred all-reduce, a reduction
    that folds into a pending partial reduce is not treated as creating the gradient,
    so a window in which no rank used the parameter in any micro-batch keeps the
    reduced zero there.

    Raises:
        RuntimeError: If the installed PyTorch lacks the FSDP2 internals the patch wraps.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
        from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    except ImportError as error:
        raise RuntimeError(
            "FSDP unused-parameter reduction keeps gradients of parameters no rank used as None, but this "
            "PyTorch "
            "version does not provide the compatible FSDP2 implementation."
        ) from error

    original_final_callback = FSDPState._root_post_backward_final_callback
    if getattr(original_final_callback, "_automodel_restore_unused_param_grads", False):
        return
    original_post_backward = FSDPParamGroup.post_backward
    legacy_fill = getattr(original_post_backward, "_automodel_reduce_scatter_unused_params", False)

    def _post_backward_recording_param_usage(self, *args, **kwargs):
        zero_fills = legacy_fill or getattr(self, "reduce_scatter_unused_params", False)
        if zero_fills and self.reduce_grads and self._training_state != TrainingState.POST_BACKWARD:
            # An HSDP reduction whose all-reduce was deferred holds earlier micro-batches' real
            # contributions outside ``sharded_param.grad``, so this reduction does not create the gradient.
            pending_partial_reduce = self._partial_reduce_output is not None
            self._automodel_unused_param_probe = [
                (
                    fsdp_param,
                    _has_local_unsharded_grad(fsdp_param),
                    fsdp_param.sharded_param.grad is None and not pending_partial_reduce,
                )
                for fsdp_param in self.fsdp_params
                if fsdp_param.sharded_param.requires_grad
            ]
        try:
            return original_post_backward(self, *args, **kwargs)
        except BaseException:
            # A post-backward that failed reduced nothing; callers that tolerate the failure (a
            # never-forwarded group) must not leave this probe for a later backward.
            self._automodel_unused_param_probe = None
            raise

    def _root_post_backward_final_callback_restoring_unused_grads(self, *args, **kwargs):
        try:
            return original_final_callback(self, *args, **kwargs)
        finally:
            # Also on failure: callers that tolerate a never-forwarded root still step the optimizer.
            _restore_unused_param_grads(
                [
                    group
                    for state in self._state_ctx.all_states
                    for group in _fsdp_state_param_groups(state)
                    if getattr(group, "_automodel_unused_param_probe", None) is not None
                ]
            )

    # Keep the fill's marker visible so a repeated fill install stays a no-op under this wrapper, and mark
    # the probe so a fill installed only now refuses to wrap outside it.
    _post_backward_recording_param_usage._automodel_reduce_scatter_unused_params = legacy_fill
    _post_backward_recording_param_usage._automodel_unused_param_probe = True
    _root_post_backward_final_callback_restoring_unused_grads._automodel_restore_unused_param_grads = True
    FSDPParamGroup.post_backward = _post_backward_recording_param_usage
    FSDPState._root_post_backward_final_callback = _root_post_backward_final_callback_restoring_unused_grads


def patch_fsdp_accumulated_grad_guard() -> None:
    """Guard FSDP2 post-backward against params that were never unsharded.

    PyTorch FSDP2 creates ``_unsharded_param`` lazily from an FSDP unit's
    forward pre-hook. If a separately wrapped unit is skipped by the batch
    (for example a vision tower on text-only data), deferred post-backward can
    dereference that missing field. Missing lazy state means there is no
    unsharded grad to upcast, so the exact missing-field case can return early.
    Applies only before torch 2.15, which removed the upcast step this guards.
    """
    if get_torch_version().release >= _FSDP_ACCUMULATED_GRAD_REMOVED_VERSION:
        return
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
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
    "patch_fsdp_restore_unused_param_grads",
    "patch_fsdp_uniform_reduce_dtype",
    "patch_fsdp_unused_param_reduction",
]
