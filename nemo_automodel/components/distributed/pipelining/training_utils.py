# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import math
from typing import Any, Iterable, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import DTensor


def validate_batch_shapes(batch: dict[str, Any], *, must_have: Optional[list[str]] = None) -> None:
    if must_have:
        for key in must_have:
            if key not in batch:
                raise ValueError(f"Missing required batch key: {key}")


@torch.no_grad()
def pp_scale_grads_by_divisor(
    stages: list[PipelineStage],
    divisor: int,
) -> None:
    for stage in stages:
        if hasattr(stage, "scale_grads"):
            stage.scale_grads(divisor)


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
    ep_axis_name: str,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        if ep_axis_name not in p.device_mesh.mesh_dim_names:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
        else:
            ep_params.append(p)
            ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(ep_grads, norm_type, error_if_nonfinite, foreach).full_tensor()
    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    ).full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm


@torch.no_grad()
def pp_clip_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_axis_name: str | None = None,
) -> torch.Tensor:
    if ep_axis_name:
        return _clip_grad_norm_with_ep(
            parameters, max_norm, norm_type, error_if_nonfinite, foreach, pp_mesh, ep_axis_name
        )

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
