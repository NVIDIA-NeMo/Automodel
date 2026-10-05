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

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo


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


class _PostForwardMeshInfoCache:
    """Reuse partial-reshard mesh metadata within one physical source mesh."""

    def __init__(
        self,
        original: Callable[[bool | int, FSDPMeshInfo], FSDPMeshInfo | None],
    ) -> None:
        self._original = original
        # Keep the source mesh while the weak value lives, and include identity:
        # DeviceMesh equality does not distinguish live process groups.
        self._cache: WeakValueDictionary[tuple[int, DeviceMesh, int], FSDPMeshInfo] = WeakValueDictionary()

    def __call__(self, reshard_after_forward: bool | int, mesh_info: FSDPMeshInfo) -> FSDPMeshInfo | None:
        size = mesh_info.shard_mesh_size
        if (
            isinstance(reshard_after_forward, bool)
            or not isinstance(reshard_after_forward, int)
            or not 1 < reshard_after_forward < size
            or size % reshard_after_forward
        ):
            # Preserve upstream bool, size-one/full-size normalization and every
            # invalid-input error, including a direct unsupported None argument.
            return self._original(reshard_after_forward, mesh_info)
        key = (reshard_after_forward, mesh_info.mesh, id(mesh_info.mesh))
        result = self._cache.get(key)
        if result is None:
            result = self._original(reshard_after_forward, mesh_info)
            if result is not None:
                self._cache[key] = result
        return result


def patch_fsdp_post_forward_mesh_cache() -> None:
    """Deduplicate NCCL groups for integer resharding on affected PyTorch builds.

    PyTorch 2.13.0a0+8145d630e8.nv26.06 constructs a fresh DeviceMesh for
    every FSDP unit when 1 < reshard_after_forward < shard_world_size.
    Duplicate NCCL/NVLS allocations can exhaust device memory outside the
    PyTorch allocator. See https://github.com/pytorch/pytorch/issues/187155.

    This compatibility patch follows the weak-value cache proposed in open PR
    https://github.com/pytorch/pytorch/pull/187161, with source-mesh identity
    added to avoid sharing across distinct process groups with equal layouts.
    It changes neither collective membership nor bool/no-reshard behavior.
    Both imported aliases are updated; installation is process-local and
    idempotent. It does not modify installed PyTorch files.
    """
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_init as fsdp_init
        import torch.distributed.fsdp._fully_shard._fully_shard as fully_shard_module
    except ImportError:
        return
    original = fsdp_init._get_post_forward_mesh_info
    cached = original if isinstance(original, _PostForwardMeshInfoCache) else _PostForwardMeshInfoCache(original)
    fsdp_init._get_post_forward_mesh_info = cached
    fully_shard_module._get_post_forward_mesh_info = cached
