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

"""FSDP2 ownership with per-parameter transient compute dtypes."""

from types import MethodType
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.fsdp2_extensions.compat import (
    compiled_autograd_active,
    patch_fsdp_accumulated_grad_bucketing,
    patch_fsdp_uniform_reduce_dtype,
)
from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    fully_shard_by_dtype,
    make_parameter_compute_dtype_resolver,
)

_ComputeDtypeMetadata = tuple[torch.dtype, torch.Size, tuple[int, ...]]
# ``(owning module, local parameter name) -> (parameter, compute dtype)`` for
# every parameter one candidate FSDP unit would own.
_ComputeDtypePlan = dict[tuple[nn.Module, str], tuple[nn.Parameter, torch.dtype]]


def _child_fsdp_parameters(module: nn.Module) -> set[nn.Parameter]:
    """Return parameter tensors of arbitrary shape owned by nested FSDP units."""
    return {
        parameter
        for child in module.modules()
        if child is not module and isinstance(child, FSDPModule)
        for parameter in child.parameters()
    }


def _plan_per_param_compute_dtypes(
    module: nn.Module,
    *,
    fp32_compute_module_names: tuple[str, ...],
    mp_policy: MixedPrecisionPolicy | None,
    ignored_params: set[nn.Parameter],
) -> _ComputeDtypePlan:
    """Resolve the compute dtype of every parameter ``module`` would own as one FSDP unit.

    Args:
        module: Candidate FSDP unit containing parameter tensors of arbitrary shape.
        fp32_compute_module_names: Name fragments selecting parameters that must
            materialize in FP32 compute.
        mp_policy: Default parameter-compute and gradient-reduction dtypes.
        ignored_params: Parameter tensors of arbitrary shape owned outside the
            candidate unit, including those of nested FSDP units. DTensors retain
            their global shapes and placements.

    Returns:
        Mapping from ``(owner, name)`` to the owned parameter and its compute dtype.
    """
    ignored_param_ids = {id(parameter) for parameter in ignored_params}
    compute_dtype_of = make_parameter_compute_dtype_resolver(
        module,
        mp_policy,
        fp32_compute_module_names,
        ignored_params=ignored_params,
    )
    return {
        (owner, name): (parameter, compute_dtype_of(parameter))
        for owner in module.modules()
        for name, parameter in owner.named_parameters(recurse=False)
        if id(parameter) not in ignored_param_ids
    }


def _overrides_unit_policy(
    resident_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    mp_policy: MixedPrecisionPolicy | None,
) -> bool:
    """Return whether a floating parameter computes in a dtype other than the unit's default."""
    default_dtype = getattr(mp_policy, "param_dtype", None) or resident_dtype
    return resident_dtype.is_floating_point and compute_dtype is not default_dtype


def _supports_per_param_compute_dtype_extension(
    plan: _ComputeDtypePlan,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
) -> bool:
    """Return whether one candidate unit can use the optimized tensor extension.

    Args:
        plan: Per-parameter compute dtypes from :func:`_plan_per_param_compute_dtypes`.
        mesh: FSDP or HSDP mesh. HSDP shards on the last mesh dimension.
        mp_policy: Default parameter-compute and gradient-reduction dtypes.
        offload_policy: Optional FSDP parameter offload policy.

    Returns:
        ``True`` only when every resident floating parameter is FP32, at least one
        parameter overrides the unit policy, and each overriding tensor's rank-local
        dim 0 shards evenly over the FSDP shard mesh.
    """
    if isinstance(offload_policy, CPUOffloadPolicy) or compiled_autograd_active():
        return False
    floating = [parameter for parameter, _ in plan.values() if parameter.dtype.is_floating_point]
    if not floating or any(parameter.dtype is not torch.float32 for parameter in floating):
        return False
    overrides = [
        parameter.to_local() if isinstance(parameter, DTensor) else parameter
        for parameter, compute_dtype in plan.values()
        if _overrides_unit_policy(parameter.dtype, compute_dtype, mp_policy)
    ]
    shard_size = mesh.shape[-1]
    return bool(overrides) and all(local.ndim > 0 and local.shape[0] % shard_size == 0 for local in overrides)


def _fsdp_pre_all_gather_in_compute_dtype(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    outer_size: torch.Size,
    outer_stride: tuple[int, ...],
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy,
) -> tuple[tuple[torch.Tensor, ...], _ComputeDtypeMetadata]:
    """Create a transient compute-precision all-gather input from an FP32 master shard.

    Args:
        tensor: Per-rank local parameter shard of shape ``[local_shard_numel]``.
        mesh: FSDP device mesh that shards the parameter on mesh dimension 0.
        outer_size: Global unsharded parameter shape.
        outer_stride: Global unsharded parameter stride.
        module: Module that owns ``tensor``; unused by this extension.
        mp_policy: Mixed-precision policy for the enclosing FSDP unit; unused by
            this per-parameter override.

    Returns:
        A one-element tuple containing the local gather input of shape
        ``[local_shard_numel]`` and metadata describing the global parameter.
    """
    del module, mp_policy
    compute_dtype = tensor._compute_dtype
    if outer_size[0] % mesh.size() != 0:
        raise NotImplementedError(
            "per-parameter FSDP compute casting requires even dim-0 sharding; "
            f"got shape {tuple(outer_size)} over {mesh.size()} ranks"
        )
    # Pinned FP32 parameters already have the requested compute dtype. Reuse
    # their resident shard directly instead of dispatching a redundant cast.
    all_gather_input = tensor if tensor.dtype is compute_dtype else tensor.to(compute_dtype)
    metadata = (compute_dtype, outer_size, outer_stride)
    return (all_gather_input,), metadata


@torch.no_grad()
def _fsdp_post_all_gather_in_compute_dtype(
    tensor: torch.Tensor,
    all_gather_outputs: tuple[torch.Tensor, ...],
    metadata: _ComputeDtypeMetadata,
    param_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
    """Expose one gathered parameter in its checkpoint-defined compute dtype.

    Args:
        tensor: Per-rank local FP32 master shard of shape ``[local_shard_numel]``;
            unused after the all-gather.
        all_gather_outputs: One-element tuple containing the flattened global
            parameter of shape ``[global_numel]`` in its compute dtype.
        metadata: Compute dtype, global parameter shape, and global parameter stride
            returned by :func:`_fsdp_pre_all_gather_in_compute_dtype`.
        param_dtype: FSDP-unit parameter dtype; unused because metadata owns the
            per-parameter compute dtype.
        out: Optional unsharded parameter tensor with the global parameter shape.
            When provided, it is updated in place and retains its storage identity.

    Returns:
        ``None`` when ``out`` is updated in place. Otherwise, the flattened gathered
        tensor of shape ``[global_numel]`` and an empty auxiliary-output tuple. The
        returned tensor aliases the all-gather output.
    """
    del tensor, param_dtype
    compute_dtype, outer_size, outer_stride = metadata
    (all_gather_output,) = all_gather_outputs
    if all_gather_output.dtype is not compute_dtype:
        raise AssertionError(f"expected {compute_dtype} all-gather output, got {all_gather_output.dtype}")
    if out is not None:
        if out.dtype is not compute_dtype:
            raise AssertionError(f"expected {compute_dtype} unsharded parameter, got {out.dtype}")
        source = torch.as_strided(all_gather_output, outer_size, outer_stride)
        with torch.autograd._unsafe_preserve_version_counter(out):
            out.copy_(source)
        return None
    return all_gather_output, ()


def _install_per_param_compute_dtypes(
    module: nn.Module,
    compute_dtypes: dict[tuple[nn.Module, str], torch.dtype],
    mp_policy: MixedPrecisionPolicy | None,
) -> int:
    """Install FSDP tensor extensions on the unit's parameters that override the unit policy."""
    get_fsdp_state = getattr(module, "_get_fsdp_state", None)
    if get_fsdp_state is None:
        raise RuntimeError("per-parameter FSDP compute casting requires a PyTorch FSDPModule")
    param_group = getattr(get_fsdp_state(), "_fsdp_param_group", None)
    if param_group is None:
        return 0

    installed = 0
    for fsdp_param in param_group.fsdp_params:
        module_info = fsdp_param._module_info
        compute_dtype = compute_dtypes[(module_info.module, module_info.param_name)]
        if not _overrides_unit_policy(fsdp_param.sharded_param.dtype, compute_dtype, mp_policy):
            continue
        local_tensor = fsdp_param._sharded_local_tensor
        local_tensor._compute_dtype = compute_dtype
        local_tensor.fsdp_pre_all_gather = MethodType(_fsdp_pre_all_gather_in_compute_dtype, local_tensor)
        local_tensor.fsdp_post_all_gather = MethodType(_fsdp_post_all_gather_in_compute_dtype, local_tensor)
        fsdp_param._init_extensions()
        installed += 1
    return installed


def _fully_shard_with_plan(
    module: nn.Module,
    plan: _ComputeDtypePlan,
    *,
    mp_policy: MixedPrecisionPolicy | None,
    ignored_params: set[nn.Parameter],
    fully_shard_fn: Callable[..., nn.Module],
    **fully_shard_kwargs,
) -> nn.Module:
    """Establish one FSDP unit for ``module`` and install the planned per-parameter extensions.

    ``fully_shard_kwargs`` (mesh, offload and reshard policies) pass straight to
    ``fully_shard_fn``. The extensions live on the sharded local tensors, which
    checkpoint loading may replace, so they are reinstalled from a
    ``load_state_dict`` post-hook.
    """
    compute_dtypes = {key: compute_dtype for key, (_, compute_dtype) in plan.items()}
    wrapped = fully_shard_fn(module, mp_policy=mp_policy, ignored_params=ignored_params or None, **fully_shard_kwargs)

    def install_extensions(*_args) -> None:
        if _install_per_param_compute_dtypes(module, compute_dtypes, mp_policy):
            patch_fsdp_accumulated_grad_bucketing()
            patch_fsdp_uniform_reduce_dtype()

    install_extensions()
    module.register_load_state_dict_post_hook(install_extensions)
    return wrapped


def fully_shard_with_per_param_compute_dtypes(
    module: nn.Module,
    *,
    fp32_compute_module_names: tuple[str, ...],
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    fully_shard_fn: Callable[..., nn.Module] = fully_shard,
) -> nn.Module:
    """Fully shard one FP32-master unit with per-parameter transient compute dtypes.

    FSDP retains one ownership and collective boundary for ``module``. Its normal
    mixed-precision policy casts ordinary FP32 master shards to ``param_dtype``;
    parameters resolved to another compute dtype use PyTorch's per-tensor FSDP
    all-gather extension. Mixed all-gather inputs are packed into one collective.

    Args:
        module: Module whose FP32 resident parameters form one FSDP ownership unit.
        fp32_compute_module_names: Parameter-name fragments whose weights must also
            compute in FP32.
        mesh: FSDP device mesh that owns the unit's sharding collective.
        mp_policy: Mixed-precision policy defining the default compute and reduction
            dtypes.
        offload_policy: FSDP offload policy. CPU offload is not supported by this
            per-parameter extension.
        reshard_after_forward: Optional FSDP reshard behavior for the unit.
        ignored_params: Parameters owned by another sharding or replication policy.
        fully_shard_fn: FSDP implementation used to establish the ownership unit.

    Returns:
        The FSDP-wrapped ``module`` returned by ``fully_shard_fn``.

    Raises:
        NotImplementedError: Under CPU offload or compiled autograd, which the
            per-tensor extension does not support.
        ValueError: If a resident floating parameter the unit would own is not FP32.
    """
    if isinstance(offload_policy, CPUOffloadPolicy):
        raise NotImplementedError("per-parameter FSDP compute casting does not support CPU offload")
    if compiled_autograd_active():
        raise NotImplementedError("per-parameter FSDP compute casting is incompatible with compiled autograd")

    ignored_params = set(ignored_params or ()) | _child_fsdp_parameters(module)
    plan = _plan_per_param_compute_dtypes(
        module,
        fp32_compute_module_names=fp32_compute_module_names,
        mp_policy=mp_policy,
        ignored_params=ignored_params,
    )
    for (owner, name), (parameter, _) in plan.items():
        if parameter.dtype.is_floating_point and parameter.dtype is not torch.float32:
            raise ValueError(
                "per-parameter FSDP compute casting requires FP32 resident/master weights; "
                f"{type(owner).__name__}.{name} is {parameter.dtype}"
            )
    return _fully_shard_with_plan(
        module,
        plan,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
        ignored_params=ignored_params,
        fully_shard_fn=fully_shard_fn,
    )


def fully_shard_with_compute_dtype_fallback(
    module: nn.Module,
    *,
    fp32_compute_module_names: tuple[str, ...],
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    fully_shard_fn: Callable[..., nn.Module] = fully_shard,
) -> nn.Module:
    """Fully shard one unit with the most efficient supported dtype ownership.

    The single-owner tensor extension is used only for uniform FP32 resident
    weights, mixed compute dtypes, and a supported runtime/shape. Every other
    layout delegates to the established storage-and-compute dtype grouping,
    which also collapses uniform layouts to ordinary one-unit FSDP.

    Args:
        module: Candidate FSDP ownership unit.
        fp32_compute_module_names: Parameter-name fragments pinned to FP32 compute.
        mesh: FSDP or HSDP device mesh.
        mp_policy: Default FSDP compute and reduction policy.
        offload_policy: Optional FSDP offload policy.
        reshard_after_forward: Optional FSDP reshard behavior.
        ignored_params: Parameter tensors of arbitrary shape owned by another FSDP
            or replication policy. DTensors retain their global shapes and placements.
        fully_shard_fn: FSDP implementation used to establish ownership units.

    Returns:
        The input module after FSDP ownership has been established.
    """
    # Deferred accumulation is a group-level FSDP2 concern, including the
    # common uniform BF16-compute/FP32-reduce case. Install it before selecting
    # either the per-tensor extension or dtype-grouped compatibility path.
    patch_fsdp_accumulated_grad_bucketing()
    ignored_params = set(ignored_params or ()) | _child_fsdp_parameters(module)
    plan = _plan_per_param_compute_dtypes(
        module,
        fp32_compute_module_names=fp32_compute_module_names,
        mp_policy=mp_policy,
        ignored_params=ignored_params,
    )
    if _supports_per_param_compute_dtype_extension(plan, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy):
        return _fully_shard_with_plan(
            module,
            plan,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params,
            fully_shard_fn=fully_shard_fn,
        )

    fully_shard_by_dtype(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        fp32_compute_module_names=fp32_compute_module_names,
        reshard_after_forward=reshard_after_forward,
        ignored_params=ignored_params or None,
        fully_shard_fn=fully_shard_fn,
    )
    return module
