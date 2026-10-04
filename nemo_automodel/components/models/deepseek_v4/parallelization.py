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

"""Model-owned distributed parallelization for DeepSeek-V4 and DeepSeek-V4.1.

The shared :class:`ModelParallelizer` already keeps the parameters named in the
model's ``_keep_in_fp32_modules_strict`` in fp32 inside each unit and runs the
all-fp32 ``lm_head`` unit in fp32. DeepSeek-V4 only adds the HCA parameter-sync
group, which is known only once the FSDP mesh is.
"""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Compressor


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


class DeepseekV4ModelParallelizer(ModelParallelizer):
    """Shared parallelization plus the HCA parameter-sync group per FSDP unit."""

    def _fully_shard_module(self, module: nn.Module, **kwargs) -> nn.Module:
        _attach_hca_param_sync_group(module, kwargs["mesh"])
        return super()._fully_shard_module(module, **kwargs)


__all__ = ["DeepseekV4ModelParallelizer"]
