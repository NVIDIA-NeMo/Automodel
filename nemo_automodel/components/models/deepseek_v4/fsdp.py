# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""FSDP2 primitive shared by DeepSeek-V4 and DeepSeek-V4.1.

Every unit the parallelizer hands over (decoder block, vision block or tower,
model root, ``lm_head``) becomes exactly one FSDP unit. Parameters named in the
model's ``_keep_in_fp32_modules_strict`` keep fp32 compute inside that unit
through PyTorch's per-parameter ``param_dtype_override_fn`` (see
``with_fp32_compute_override``); a unit made only of such parameters, such as
``lm_head``, computes in fp32 outright and returns fp32 activations.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed.parallelizer_utils import with_fp32_compute_override
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Compressor
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn


def _hca_param_sync_group_from_1d_mesh(mesh):
    """Return the 1D PyTorch FSDP2 group used for HCA graph alignment.

    HCA graph alignment is an FSDP/FSDP2 parameter-sync invariant: ranks that
    synchronize the same sharded HCA parameters must agree on whether the HCA
    compressor path participates in backward. This DeepSeek-V4 wrapper gets
    that domain from its 1D PyTorch FSDP2 mesh. The mesh may be named or
    unnamed; multi-dimensional meshes need an explicit owner dimension to avoid
    reducing across unrelated parallel groups. Until that is available, disable
    HCA graph alignment instead of using a broader or wrong group.
    """
    if mesh is None:
        return None

    mesh_ndim = getattr(mesh, "ndim", None)
    mesh_shape = getattr(mesh, "shape", None)
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if mesh_ndim is not None:
        is_1d_mesh = mesh_ndim == 1
    elif mesh_shape is not None:
        is_1d_mesh = len(mesh_shape) == 1
    elif mesh_dim_names is not None:
        is_1d_mesh = len(mesh_dim_names) == 1
    else:
        return None
    if not is_1d_mesh:
        return None

    try:
        if mesh.size() <= 1:
            return None
        return mesh.get_group()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _attach_hca_param_sync_group(module: nn.Module, mesh: DeviceMesh | None) -> None:
    """Bind the FSDP mesh's parameter-sync group to every HCA compressor in ``module``.

    The FSDP2 mesh is only known while wrapping, so the group is attached here
    instead of through public model configuration.
    """
    process_group = _hca_param_sync_group_from_1d_mesh(mesh)
    for submodule in module.modules():
        if isinstance(submodule, DeepseekV4Compressor):
            submodule._set_hca_param_sync_group(process_group)


def deepseek_v4_unit_mp_policy(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy | None,
    fp32_compute_module_names: tuple[str, ...],
    ignored_params: set[nn.Parameter] | None = None,
    *,
    module_name: str = "",
) -> MixedPrecisionPolicy | None:
    """Return the mixed-precision policy for one DeepSeek-V4 FSDP unit.

    The strict names are model-level tokens (``lm_head``, ``vision.norm.weight``,
    ``attn_hc.fn``). They are matched against ``module_name`` joined with each
    parameter's unit-relative name, so a unit such as ``lm_head`` whose only
    parameter is ``weight`` still resolves its contract. Without ``module_name``
    the unit-relative names are matched directly, which is sufficient for
    decoder and vision blocks.

    Args:
        module: Module about to become one FSDP unit.
        mp_policy: Policy of the enclosing FSDP boundary, or ``None``.
        fp32_compute_module_names: The model's ``_keep_in_fp32_modules_strict``.
        ignored_params: Parameters owned by another FSDP or parallelism unit.
        module_name: Model-level name of ``module``; empty for the model itself
            or when unknown.

    Returns:
        ``mp_policy`` itself when it already computes in fp32 or when no
        parameter of the unit has an fp32 contract. A full fp32 policy (fp32
        parameters, reductions, inputs and outputs) when every parameter of the
        unit has an fp32 contract, e.g. ``lm_head``. Otherwise ``mp_policy``
        with ``param_dtype_override_fn`` pinning the fp32-contract parameters.

    """
    if mp_policy is None or mp_policy.param_dtype in (None, torch.float32):
        return mp_policy
    ignored_param_ids = {id(param) for param in ignored_params or ()}
    owned = [
        canonical_parameter_fqn(name)
        for name, param in module.named_parameters()
        if id(param) not in ignored_param_ids and param.dtype.is_floating_point
    ]
    prefix = f"{module_name}." if module_name else ""
    pinned = [name for name in owned if any(token in prefix + name for token in fp32_compute_module_names)]
    if not pinned:
        return mp_policy
    if len(pinned) == len(owned):
        return replace(
            mp_policy,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )
    return with_fp32_compute_override(
        module, mp_policy, fp32_compute_module_names, ignored_params, module_name=module_name
    )


def fully_shard_deepseek_v4(
    module: nn.Module,
    *,
    fp32_compute_module_names: tuple[str, ...],
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    module_name: str = "",
    **fsdp_kwargs,
) -> nn.Module:
    """Shard ``module`` as one FSDP2 unit with DeepSeek-V4's fp32 compute contract.

    Args:
        module: Module to shard in place.
        fp32_compute_module_names: The model's ``_keep_in_fp32_modules_strict``.
        mesh: Runtime FSDP device mesh; its 1D group also drives HCA graph alignment.
        mp_policy: Mixed-precision policy of the enclosing boundary.
        module_name: Model-level name of ``module`` (see
            :func:`deepseek_v4_unit_mp_policy`).
        **fsdp_kwargs: Remaining ``fully_shard`` keyword arguments
            (``offload_policy``, ``reshard_after_forward``, ``ignored_params``).

    Returns:
        The input module with FSDP applied.
    """
    _attach_hca_param_sync_group(module, mesh)
    return fully_shard(
        module,
        mesh=mesh,
        mp_policy=deepseek_v4_unit_mp_policy(
            module,
            mp_policy,
            fp32_compute_module_names,
            fsdp_kwargs.get("ignored_params"),
            module_name=module_name,
        ),
        **fsdp_kwargs,
    )


__all__ = ["deepseek_v4_unit_mp_policy", "fully_shard_deepseek_v4"]
