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

from nemo_automodel.components.distributed.fsdp2_extensions.compat import patch_fsdp_uniform_reduce_dtype
from nemo_automodel.components.distributed.fsdp2_extensions.utils import (
    fully_shard_by_dtype,
    make_parameter_compute_dtype_resolver,
)

_ComputeDtypeMetadata = tuple[torch.dtype, torch.Size, tuple[int, ...]]


def _compiled_autograd_is_enabled() -> bool:
    import torch._dynamo.compiled_autograd as compiled_autograd

    return bool(
        compiled_autograd.compiled_autograd_enabled
        or compiled_autograd.compiled_autograd_enabled_force_eager
        or compiled_autograd.in_compiled_autograd_region
    )


def _fsdp_shard_mesh_size(mesh: DeviceMesh) -> int:
    """Return the dimension that shards parameters for FSDP or HSDP."""
    return mesh.size() if mesh.ndim == 1 else mesh.shape[-1]


def _parameters_owned_by_candidate_unit(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
) -> tuple[nn.Parameter, ...]:
    """Return parameters not already owned by child FSDP units or another policy.

    Args:
        module: Candidate FSDP unit containing parameter tensors of arbitrary shape.
        ignored_params: Parameter tensors of arbitrary shape whose ownership is
            already established outside ``module``.

    Returns:
        Parameter tensors of arbitrary shape that the candidate unit would own.
        DTensors retain their existing global shape, device mesh, and placements.
    """
    excluded_ids = {id(parameter) for parameter in ignored_params}
    for child in module.modules():
        if child is not module and isinstance(child, FSDPModule):
            excluded_ids.update(id(parameter) for parameter in child.parameters())
    return tuple(parameter for parameter in module.parameters() if id(parameter) not in excluded_ids)


def _child_fsdp_parameters(module: nn.Module) -> set[nn.Parameter]:
    """Return parameter tensors of arbitrary shape owned by nested FSDP units."""
    return {
        parameter
        for child in module.modules()
        if child is not module and isinstance(child, FSDPModule)
        for parameter in child.parameters()
    }


def _supports_per_param_compute_dtype_extension(
    module: nn.Module,
    *,
    fp32_compute_module_names: tuple[str, ...],
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    ignored_params: set[nn.Parameter],
) -> bool:
    """Return whether one candidate unit can use the optimized tensor extension.

    Args:
        module: Candidate FSDP unit containing parameter tensors of arbitrary shape.
        fp32_compute_module_names: Name fragments selecting parameters that must
            materialize in FP32 compute.
        mesh: FSDP or HSDP mesh. HSDP shards on the last mesh dimension.
        mp_policy: Default parameter-compute and gradient-reduction dtypes.
        offload_policy: Optional FSDP parameter offload policy.
        ignored_params: Parameter tensors of arbitrary shape owned outside the
            candidate unit. DTensors retain their global shapes and placements.

    Returns:
        ``True`` only when every resident floating parameter is FP32 and each
        overridden tensor's rank-local dim 0 shards evenly over the FSDP shard mesh.
    """
    if isinstance(offload_policy, CPUOffloadPolicy) or _compiled_autograd_is_enabled():
        return False

    parameters = _parameters_owned_by_candidate_unit(module, ignored_params)
    floating_parameters = tuple(parameter for parameter in parameters if parameter.dtype.is_floating_point)
    if not floating_parameters or any(parameter.dtype is not torch.float32 for parameter in floating_parameters):
        return False

    compute_dtype_of = make_parameter_compute_dtype_resolver(
        module,
        mp_policy,
        fp32_compute_module_names,
        ignored_params=ignored_params,
    )
    policy_dtype = getattr(mp_policy, "param_dtype", None)
    overrides = tuple(
        parameter
        for parameter in floating_parameters
        if compute_dtype_of(parameter) is not (policy_dtype or parameter.dtype)
    )
    if not overrides:
        return False

    shard_size = _fsdp_shard_mesh_size(mesh)
    local_overrides = tuple(
        parameter.to_local() if isinstance(parameter, DTensor) else parameter for parameter in overrides
    )
    return all(parameter.ndim > 0 and parameter.shape[0] % shard_size == 0 for parameter in local_overrides)


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
    compute_dtype_by_owner: dict[tuple[int, str], torch.dtype],
    policy_dtype: torch.dtype | None,
) -> int:
    """Install FSDP tensor extensions on parameters overriding the unit policy."""
    if _compiled_autograd_is_enabled():
        raise NotImplementedError("per-parameter FSDP compute casting is incompatible with compiled autograd")
    get_fsdp_state = getattr(module, "_get_fsdp_state", None)
    if get_fsdp_state is None:
        raise RuntimeError("per-parameter FSDP compute casting requires a PyTorch FSDPModule")
    fsdp_state = get_fsdp_state()
    param_group = getattr(fsdp_state, "_fsdp_param_group", None)
    if param_group is None:
        return 0

    installed = 0
    for fsdp_param in param_group.fsdp_params:
        module_info = fsdp_param._module_info
        compute_dtype = compute_dtype_by_owner[(id(module_info.module), module_info.param_name)]
        default_dtype = policy_dtype or fsdp_param.sharded_param.dtype
        if compute_dtype is default_dtype:
            continue
        local_tensor = fsdp_param._sharded_local_tensor
        local_tensor._compute_dtype = compute_dtype
        local_tensor.fsdp_pre_all_gather = MethodType(_fsdp_pre_all_gather_in_compute_dtype, local_tensor)
        local_tensor.fsdp_post_all_gather = MethodType(_fsdp_post_all_gather_in_compute_dtype, local_tensor)
        fsdp_param._init_extensions()
        installed += 1
    return installed


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
    """
    if isinstance(offload_policy, CPUOffloadPolicy):
        raise NotImplementedError("per-parameter FSDP compute casting does not support CPU offload")

    ignored_params = set(ignored_params or ()) | _child_fsdp_parameters(module)
    ignored_param_ids = {id(parameter) for parameter in ignored_params}
    compute_dtype_of = make_parameter_compute_dtype_resolver(
        module,
        mp_policy,
        fp32_compute_module_names,
        ignored_params=ignored_params,
    )
    compute_dtype_by_owner: dict[tuple[int, str], torch.dtype] = {}
    for owner in module.modules():
        for name, parameter in owner.named_parameters(recurse=False):
            if id(parameter) in ignored_param_ids:
                continue
            if parameter.dtype.is_floating_point and parameter.dtype is not torch.float32:
                raise ValueError(
                    "per-parameter FSDP compute casting requires FP32 resident/master weights; "
                    f"{type(owner).__name__}.{name} is {parameter.dtype}"
                )
            compute_dtype_by_owner[(id(owner), name)] = compute_dtype_of(parameter)

    wrapped = fully_shard_fn(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
        ignored_params=ignored_params or None,
    )
    policy_dtype = getattr(mp_policy, "param_dtype", None)

    def install_extensions(*_args) -> None:
        installed = _install_per_param_compute_dtypes(module, compute_dtype_by_owner, policy_dtype)
        if installed:
            patch_fsdp_uniform_reduce_dtype()

    install_extensions()
    module.register_load_state_dict_post_hook(install_extensions)
    return wrapped


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
    ignored_params = set(ignored_params or ()) | _child_fsdp_parameters(module)
    if _supports_per_param_compute_dtype_extension(
        module,
        fp32_compute_module_names=fp32_compute_module_names,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        ignored_params=ignored_params,
    ):
        return fully_shard_with_per_param_compute_dtypes(
            module,
            fp32_compute_module_names=fp32_compute_module_names,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params or None,
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
