# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Compatibility exports for the pipeline helpers split into focused modules."""

from nemo_automodel.components.distributed.pipelining.model_parts import (
    ParallelizeFnProtocol,
    PipelineModelPart,
    split_model_into_parts,
)
from nemo_automodel.components.distributed.pipelining.module_plan import (
    calculate_virtual_stages,
    generate_hf_model_fqn_per_model_part,
    stage_ids_this_rank,
)
from nemo_automodel.components.distributed.pipelining.schedules import build_pipeline_schedule
from nemo_automodel.components.distributed.pipelining.stage_runtime import (
    configure_pipeline_stage_backward,
    create_pipeline_stages,
    scale_grads_by_divisor,
    warmup_pipeline_stage_neighbors,
)

__all__ = [
    "ParallelizeFnProtocol",
    "PipelineModelPart",
    "build_pipeline_schedule",
    "calculate_virtual_stages",
    "configure_pipeline_stage_backward",
    "create_pipeline_stages",
    "generate_hf_model_fqn_per_model_part",
    "scale_grads_by_divisor",
    "split_model_into_parts",
    "stage_ids_this_rank",
    "warmup_pipeline_stage_neighbors",
]
