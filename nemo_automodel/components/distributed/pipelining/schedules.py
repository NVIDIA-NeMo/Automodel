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
        schedule_class = get_schedule_class(pipeline_parallel_schedule)

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
