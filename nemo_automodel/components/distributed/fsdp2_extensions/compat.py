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

"""Compatibility hooks for PyTorch FSDP2 behavior not yet available publicly."""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any


def compiled_autograd_active() -> bool:
    """Return whether PyTorch compiled autograd is enabled or currently executing."""
    try:
        import torch._dynamo.compiled_autograd as compiled_autograd

        return bool(
            compiled_autograd.compiled_autograd_enabled
            or compiled_autograd.compiled_autograd_enabled_force_eager
            or compiled_autograd.in_compiled_autograd_region
        )
    except (ImportError, AttributeError):
        return False


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

    @functools.wraps(original_foreach_reduce)
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


def patch_fsdp_accumulated_grad_bucketing() -> None:
    """Coalesce FSDP2's first deferred-accumulation dtype conversion.

    With BF16 compute and FP32 reduction, a backward executed under
    ``set_requires_gradient_sync(False)`` calls ``grad.to(reduce_dtype)`` once
    per parameter from ``FSDPParamGroup.post_backward``. Later microbatches
    accumulate into those FP32 tensors, and the synchronized backward reduces
    them in FP32.

    Keep that conversion where upstream performs it -- in ``post_backward``,
    after the group has resharded -- but cast the group's eligible gradients
    into one flat ``reduce_dtype`` bucket with FSDP's own ``chunk_cat`` operator
    and install the resulting views as the existing
    ``unsharded_accumulated_grad`` state. Every later accumulation/reduction
    transition is unchanged, and the conversion does not depend on
    ``set_is_last_backward``.

    Two upstream invariants are preserved explicitly:

    * a unit that did not run forward has no lazily created
      ``_unsharded_param`` and therefore nothing to convert, so it is skipped
      instead of dereferenced;
    * a DTensor gradient (tensor parallelism) is re-wrapped as a DTensor over
      the same mesh and placements, because FSDP later accumulates into it
      in place with the next microbatch's DTensor gradient.

    The patch is process-global and idempotent, including when stacked with the
    other FSDP2 patches in this module.
    """
    try:
        import torch
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
        from torch.distributed.tensor import DTensor
    except ImportError:
        return

    original_to_accumulated = FSDPParam.to_accumulated_grad_if_needed
    original_post_backward = FSDPParamGroup.post_backward
    if getattr(original_post_backward, "_automodel_bucket_accumulated_grads", False):
        return

    def pending_conversion(fsdp_param) -> Any:
        """Return the gradient awaiting its first ``reduce_dtype`` conversion, else ``None``."""
        if getattr(fsdp_param, "offload_to_cpu", False) or fsdp_param.reduce_dtype is None:
            return None
        # ``_unsharded_param`` is created lazily by the unit's first forward; a
        # unit skipped by this batch has neither the field nor a gradient.
        unsharded_param = getattr(fsdp_param, "_unsharded_param", None)
        grad = None if unsharded_param is None else unsharded_param.grad
        if grad is None or grad.dtype is fsdp_param.reduce_dtype or fsdp_param.unsharded_accumulated_grad is not None:
            return None
        return grad

    @functools.wraps(original_to_accumulated)
    def defer_accumulated_grad_conversion(self) -> None:
        if compiled_autograd_active():
            return original_to_accumulated(self)
        if not hasattr(self, "_unsharded_param"):
            return  # never unsharded, so there is no gradient to convert
        if pending_conversion(self) is None:
            return original_to_accumulated(self)
        # ``post_backward`` converts this gradient together with the rest of its group.

    @functools.wraps(original_post_backward)
    def post_backward_with_bucketed_accumulation(self, *args, **kwargs):
        result = original_post_backward(self, *args, **kwargs)
        if compiled_autograd_active() or self.reduce_grads:
            return result

        with torch.no_grad():
            grouped: dict[tuple[torch.device, torch.dtype, torch.dtype], list[tuple[Any, Any, torch.Tensor]]] = {}
            for fsdp_param in self.fsdp_params:
                grad = pending_conversion(fsdp_param)
                if grad is None:
                    continue
                local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
                grouped.setdefault((local_grad.device, local_grad.dtype, fsdp_param.reduce_dtype), []).append(
                    (fsdp_param, grad, local_grad)
                )

            for (device, _grad_dtype, reduce_dtype), entries in grouped.items():
                numels = [local_grad.numel() for _, _, local_grad in entries]
                bucket = torch.empty(sum(numels), device=device, dtype=reduce_dtype)
                torch.ops.fsdp.chunk_cat(
                    [local_grad for _, _, local_grad in entries],
                    dim=0,
                    num_chunks=1,
                    out=bucket.view(1, -1),
                )
                for (fsdp_param, grad, local_grad), flat_view in zip(entries, bucket.split(numels)):
                    converted = flat_view.view(local_grad.shape)
                    if isinstance(grad, DTensor):
                        converted = DTensor.from_local(
                            converted,
                            grad.device_mesh,
                            grad.placements,
                            run_check=False,
                            shape=grad.shape,
                            stride=grad.stride(),
                        )
                    fsdp_param._unsharded_param.grad = None
                    fsdp_param.unsharded_accumulated_grad = converted
        return result

    defer_accumulated_grad_conversion._automodel_bucket_accumulated_grads = True
    post_backward_with_bucketed_accumulation._automodel_bucket_accumulated_grads = True
    FSDPParam.to_accumulated_grad_if_needed = defer_accumulated_grad_conversion
    FSDPParamGroup.post_backward = post_backward_with_bucketed_accumulation


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

    @functools.wraps(original_post_backward)
    def _post_backward_with_unused_param_reduction(self, *args, **kwargs):
        if self.reduce_grads and self._training_state == TrainingState.PRE_BACKWARD:
            for fsdp_param in self.fsdp_params:
                if not hasattr(fsdp_param, "_unsharded_param"):
                    continue
                if fsdp_param.unsharded_accumulated_grad is not None:
                    continue
                param = fsdp_param.unsharded_param
                if param.requires_grad and param.grad is None:
                    # ``zeros_like`` on the *unsharded* parameter is the dtype
                    # autograd would have produced under any precision policy
                    # (bf16 compute over fp32 storage, an fp32-pinned unit, or no
                    # casting at all). ``_align_accumulated_grad_dtype`` then
                    # promotes it to ``reduce_dtype`` if the group is accumulating.
                    param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        return original_post_backward(self, *args, **kwargs)

    _post_backward_with_unused_param_reduction._automodel_reduce_scatter_unused_params = True
    FSDPParamGroup.post_backward = _post_backward_with_unused_param_reduction


def patch_fsdp_accumulated_grad_guard() -> None:
    """Guard FSDP2 post-backward against params that were never unsharded.

    PyTorch FSDP2 creates ``_unsharded_param`` lazily from an FSDP unit's
    forward pre-hook. If a separately wrapped unit is skipped by the batch
    (for example a vision tower on text-only data), deferred post-backward can
    dereference that missing field. Missing lazy state means there is no
    unsharded grad to upcast, so the exact missing-field case can return early.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return

    orig = FSDPParam.to_accumulated_grad_if_needed
    if getattr(orig, "_nemo_automodel_guarded", False):
        return

    @functools.wraps(orig)
    def guarded(self):
        try:
            return orig(self)
        except AttributeError as exc:
            if "_unsharded_param" not in str(exc) or hasattr(self, "_unsharded_param"):
                raise
            return None

    setattr(guarded, "_nemo_automodel_guarded", True)
    FSDPParam.to_accumulated_grad_if_needed = guarded
