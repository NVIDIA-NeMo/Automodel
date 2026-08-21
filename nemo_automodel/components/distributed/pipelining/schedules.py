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
import os
from typing import Callable

from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
)

logger = logging.getLogger(__name__)

#: Pipeline schedules this package supports, mapped to the stage-assignment style
#: :func:`nemo_automodel.components.distributed.pipelining.module_plan.stage_ids_this_rank`
#: must use for them. PyTorch registers additional schedules (notably
#: ``DualPipeV``) whose stage assignment is not implemented here; those are
#: rejected rather than split with the wrong mapping.
PIPELINE_SCHEDULE_STAGE_STYLES: dict[str, str] = {
    "1F1B": "loop",
    "GPipe": "loop",
    "Interleaved1F1B": "loop",
    "LoopedBFS": "loop",
    "InterleavedZeroBubble": "loop",
    "PipelineScheduleSingle": "loop",
    "PipelineScheduleMulti": "loop",
    "ZBVZeroBubble": "v",
}


def resolve_pipeline_schedule(pp_schedule: str | None) -> tuple[type[_PipelineSchedule], str]:
    """Resolve a schedule name to its class and its stage-assignment style.

    Args:
        pp_schedule: Case-insensitive schedule name, e.g. ``"interleaved1f1b"``.

    Returns:
        The schedule class and the stage-assignment style ``"loop"`` or ``"v"``.

    Raises:
        ValueError: If the name is empty, unknown to PyTorch, or names a
            schedule whose stage assignment this package does not implement.
    """
    supported = list(PIPELINE_SCHEDULE_STAGE_STYLES)
    if not pp_schedule:
        raise ValueError(f"pp_schedule must be a schedule name; supported schedules are {supported}")

    style = next(
        (
            stage_style
            for name, stage_style in PIPELINE_SCHEDULE_STAGE_STYLES.items()
            if name.lower() == pp_schedule.lower()
        ),
        None,
    )
    try:
        schedule_class = get_schedule_class(pp_schedule)
    except ValueError as error:
        raise ValueError(f"Unknown pipeline schedule {pp_schedule!r}; supported schedules are {supported}") from error
    if style is None:
        raise ValueError(
            f"Pipeline schedule {pp_schedule!r} is registered by PyTorch but its pipeline stage assignment is not "
            f"implemented here; supported schedules are {supported}"
        )
    return schedule_class, style


def build_pipeline_schedule(
    pipeline_parallel_schedule_csv: str | None,
    pipeline_parallel_schedule: str | None,
    microbatch_size: int,
    local_batch_size: int,
    stages: list[PipelineStage],
    loss_fn: Callable,
    scale_grads: bool = False,
) -> _PipelineSchedule:
    """Build a PyTorch pipeline schedule for materialized local stages."""
    if pipeline_parallel_schedule_csv:
        if not os.path.isfile(pipeline_parallel_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pipeline_parallel_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class, _ = resolve_pipeline_schedule(pipeline_parallel_schedule)

    if microbatch_size <= 0:
        raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
    if local_batch_size % microbatch_size != 0:
        raise ValueError(f"Batch size {local_batch_size} must be divisible by microbatch size {microbatch_size}.")

    n_microbatches = local_batch_size // microbatch_size
    num_local_stages = len(stages)
    if n_microbatches < num_local_stages:
        logger.warning(
            "Number of microbatches (%d) is less than the number of local stages (%d), which may create a bubble",
            n_microbatches,
            num_local_stages,
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=scale_grads,
    )
    logger.info(
        "Using pipeline schedule %s with %d microbatches and %d local stages",
        pipeline_parallel_schedule,
        n_microbatches,
        num_local_stages,
    )

    if pipeline_parallel_schedule_csv:
        schedule._load_csv(pipeline_parallel_schedule_csv)

    return schedule
