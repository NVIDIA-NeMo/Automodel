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

from typing import Any, Callable, Iterable, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torch.distributed.pipelining.stage import PipelineStage
from torch.distributed.tensor import DTensor


def validate_batch_shapes(batch: dict[str, Any], *, must_have: Optional[list[str]] = None) -> None:
    if must_have:
        for key in must_have:
            if key not in batch:
                raise ValueError(f"Missing required batch key: {key}")


def pp_forward_backward_step(
    pp_schedule: _PipelineSchedule,
    pp_has_first_stage: bool,
    pp_has_last_stage: bool,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    train_ctx: Callable,
    device: torch.device,
) -> torch.Tensor:
    with train_ctx():
        losses = [] if pp_has_last_stage else None
        if pp_has_last_stage:
            masked_labels = labels.clone()
            if loss_mask is not None:
                masked_labels[loss_mask == 0] = -100
            targets = masked_labels
        else:
            targets = None

        input_ids = batch.pop("input_ids")
        if pp_has_first_stage:
            pp_schedule.step(input_ids, target=targets, losses=losses, **batch)
        else:
            pp_schedule.step(target=targets, losses=losses, **batch)

    if pp_has_last_stage:
        loss = torch.sum(torch.stack(losses))
    else:
        loss = torch.tensor(0.0, device=device)
    return loss


@torch.no_grad()
def pp_scale_grads_by_divisor(
    stages: list[PipelineStage],
    divisor: int,
) -> None:
    for stage in stages:
        if hasattr(stage, "scale_grads"):
            stage.scale_grads(divisor)


@torch.no_grad()
def pp_clip_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_dense_params_mesh_ndim: int | None = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if torch.isinf(torch.tensor(norm_type)):
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
