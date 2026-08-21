# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import types
from dataclasses import dataclass
from typing import Callable

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import _PipelineSchedule

from nemo_automodel.components.distributed.pipelining import schedules
from nemo_automodel.components.distributed.pipelining.model_parts import PipelineModelPart

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRuntime:
    """PyTorch runtime objects built around materialized model parts."""

    schedule: _PipelineSchedule
    stages: list[PipelineStage]


def create_pipeline_stages(
    parts: list[PipelineModelPart],
    pp_mesh: DeviceMesh,
    pp_axis_name: str,
    device: torch.device,
) -> list[PipelineStage]:
    """Create PyTorch stages from already-parallelized model parts.

    Stage-boundary tensor metadata is deliberately not supplied. ``PipelineStage``
    infers it on the first ``schedule.step()`` by running each stage forward once
    under ``torch.no_grad()`` and forwarding the observed output metadata to the
    next stage. That measured metadata reflects the real dtype, context-parallel
    sharding, and number of boundary tensors, which hand-derived metadata did not.

    Args:
        parts: Materialized, already-parallelized model parts owned by this rank.
        pp_mesh: Device mesh containing the pipeline axis.
        pp_axis_name: Name of the pipeline axis within ``pp_mesh``.
        device: Device on which the local stages execute.

    Returns:
        One ``PipelineStage`` per model part, in the order the parts were given.
    """
    group = pp_mesh.get_group(pp_axis_name)
    return [PipelineStage(part.module, part.stage_index, part.num_stages, device, group=group) for part in parts]


def configure_pipeline_stage_backward(
    stages: list[PipelineStage],
    *,
    patch_stage_backward_maybe_with_nosync: bool,
    reduce_grad_per_microbatch: bool,
) -> None:
    """Configure the pipeline-stage backward policy for MoE data-parallel sync.

    ``torch.distributed.pipelining`` cannot know about Automodel's MoE FSDP
    wrapper, so expert-parallel stages need a backward that defers gradient
    reduction to the final microbatch. The replacement delegates every
    non-MoE case back to the stage's own implementation.

    Args:
        stages: Local pipeline stages to configure.
        patch_stage_backward_maybe_with_nosync: Whether to install the MoE-aware
            backward.
        reduce_grad_per_microbatch: Whether each microbatch reduces gradients
            instead of deferring reduction to the final backward.
    """
    if not patch_stage_backward_maybe_with_nosync and not reduce_grad_per_microbatch:
        return

    from nemo_automodel.components.moe.fsdp_mixin import patched_backward_maybe_with_nosync

    for stage in stages:
        if not hasattr(stage, "backward_maybe_with_nosync"):
            raise RuntimeError(
                "MoE pipeline gradient synchronization requires "
                "torch.distributed.pipelining.PipelineStage.backward_maybe_with_nosync, "
                f"which is absent from torch {torch.__version__}."
            )
        stage.backward_maybe_with_nosync = types.MethodType(patched_backward_maybe_with_nosync, stage)
        stage._reduce_grad_per_microbatch = reduce_grad_per_microbatch

    logger.info(
        "Patched pipeline stages with backward_maybe_with_nosync (reduce_grad_per_microbatch=%s)",
        reduce_grad_per_microbatch,
    )


def build_pipeline_runtime(
    parts: list[PipelineModelPart],
    pp_mesh: DeviceMesh,
    pp_axis_name: str,
    device: torch.device,
    *,
    microbatch_size: int,
    local_batch_size: int,
    schedule_name: str | None,
    schedule_csv: str | None,
    loss_fn: Callable,
    scale_grads: bool,
    patch_stage_backward_maybe_with_nosync: bool,
    reduce_grad_per_microbatch: bool,
) -> PipelineRuntime:
    """Build stages and their schedule around existing model parts.

    Args:
        parts: Materialized, already-parallelized model parts owned by this rank.
        pp_mesh: Device mesh containing the pipeline axis.
        pp_axis_name: Name of the pipeline axis within ``pp_mesh``.
        device: Device on which the local stages execute.
        microbatch_size: Samples per pipeline microbatch.
        local_batch_size: Samples per pipeline schedule step on this rank.
        schedule_name: Registered pipeline schedule name.
        schedule_csv: Path to a CSV schedule, overriding ``schedule_name``.
        loss_fn: Loss callable invoked by the schedule on the last stage.
        scale_grads: Whether the schedule divides gradients by the microbatch count.
        patch_stage_backward_maybe_with_nosync: Whether to install the MoE-aware backward.
        reduce_grad_per_microbatch: Whether each microbatch reduces gradients.

    Returns:
        The schedule and the local stages it drives.
    """
    stages = create_pipeline_stages(parts, pp_mesh, pp_axis_name, device)
    schedule = schedules.build_pipeline_schedule(
        schedule_csv,
        schedule_name,
        microbatch_size,
        local_batch_size,
        stages,
        loss_fn,
        scale_grads=scale_grads,
    )
    configure_pipeline_stage_backward(
        stages,
        patch_stage_backward_maybe_with_nosync=patch_stage_backward_maybe_with_nosync,
        reduce_grad_per_microbatch=reduce_grad_per_microbatch,
    )
    return PipelineRuntime(schedule=schedule, stages=stages)
