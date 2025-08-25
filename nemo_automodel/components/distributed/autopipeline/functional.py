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

import copy
import logging
import math
import os
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
)

from nemo_automodel.components.distributed.autopipeline.hf_utils import patch_hf_model_for_pp

logger = logging.getLogger(__name__)


def stage_ids_this_rank(pp_rank: int, pp_size: int, num_stages: int, style: str = "loop") -> tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert num_stages % pp_size == 0, f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert stages_per_rank == 2, f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1)))
        return stage_v_pairs[pp_rank]


def generate_hf_model_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    include_embeddings: bool = True,
    include_lm_head: bool = True,
    include_rotary_emb: bool = True,
    fqn_prefix: str = "model.",
) -> list[list[str]]:
    """
    Generates module names for each pipeline stage for HuggingFace models.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        include_embeddings: Whether to include embedding layer in first stage
        include_lm_head: Whether to include lm_head in last stage (for CausalLM models)

    Returns:
        List of lists containing module names for each stage

    Example:
        generate_hf_model_split(4, 32) might return:
        [
            ["model.embed_tokens", "model.layers.0", ..., "model.layers.7"],
            ["model.layers.8", ..., "model.layers.15"],
            ["model.layers.16", ..., "model.layers.23"],
            ["model.layers.24", ..., "model.layers.31", "model.norm", "lm_head"]
        ]
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages > num_layers:
        raise ValueError(f"Number of stages ({num_stages}) cannot exceed number of layers ({num_layers})")

    # Calculate base layers per stage and remainder
    layers_per_stage = num_layers // num_stages
    extra_layers = num_layers % num_stages

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate number of layers for this stage
        stage_layer_count = layers_per_stage
        if stage_idx < extra_layers:
            stage_layer_count += 1

        # First stage: add embeddings if requested
        if stage_idx == 0 and include_embeddings:
            stage_modules.append(f"{fqn_prefix}embed_tokens")

        # Add transformer layers for this stage
        for _ in range(stage_layer_count):
            stage_modules.append(f"{fqn_prefix}layers.{current_layer}")
            current_layer += 1

        # Last stage: add norm and lm_head if requested
        if stage_idx == num_stages - 1:
            stage_modules.append(f"{fqn_prefix}norm")
            if include_lm_head:
                stage_modules.append("lm_head")

        if include_rotary_emb:
            # Always include rotary_emb in all stages (it's needed for position embeddings)
            stage_modules.append(f"{fqn_prefix}rotary_emb")

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def calculate_virtual_stages(
    num_layers: int,
    layers_per_stage: Optional[int],
    pp_size: int,
    is_single_stage_schedule: bool,
    round_to_pp_multiple: str | None = None,
) -> tuple[int, int]:
    if layers_per_stage is not None:
        # Calculate number of virtual stages needed (using ceiling division)
        # This allows for unequal distribution where stages can differ by at most 1 layer
        num_virtual_stages = math.ceil(num_layers / layers_per_stage) + 1

        # Validation: check stages per rank based on schedule type
        # Common error message components to reduce duplication
        model_config_info = f"Model has {num_layers} layers with pipeline_parallel_layers_per_stage={layers_per_stage}"
        stage_distribution_info = f"resulting in {num_virtual_stages=} across {pp_size} PP ranks"

        if num_virtual_stages % pp_size != 0:
            # Rename arg to round_virtual_stages_to_pp_multiple for clarity
            if round_to_pp_multiple is not None:
                if round_to_pp_multiple == "up":
                    if num_virtual_stages % pp_size != 0:
                        num_virtual_stages += pp_size - (num_virtual_stages % pp_size)
                elif round_to_pp_multiple == "down":
                    if num_virtual_stages % pp_size != 0:
                        num_virtual_stages -= num_virtual_stages % pp_size
                else:
                    raise ValueError(
                        f"Invalid value for round_to_pp_multiple: {round_to_pp_multiple}. Use 'up' or 'down'."
                    )
            else:
                raise ValueError(
                    f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                    f"pipeline parallel size ({pp_size}). "
                    f"{model_config_info}. "
                    f"Please adjust pipeline_parallel_layers_per_stage to a value that results in a number of stages "
                    f"divisible by {pp_size}."
                )

        stages_per_rank = num_virtual_stages // pp_size

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please increase pipeline_parallel_layers_per_stage to {num_layers // pp_size} or higher "
                f"to achieve 1 stage per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, but got {stages_per_rank} stages per rank. "
                f"{model_config_info}, {stage_distribution_info}. "
                f"Please decrease pipeline_parallel_layers_per_stage to {num_layers // (2 * pp_size)} or lower "
                f"to achieve at least 2 stages per rank."
            )
    else:
        # Fallback to default behavior when layers_per_stage is not provided
        # For multi-stage schedules, default is 2 virtual stages per rank
        # For single-stage schedules, default is 1 virtual stage per rank
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = pp_size * stages_per_rank

    return num_virtual_stages, stages_per_rank


def split_model_into_stages(
    model: torch.nn.Module,
    pp_mesh: DeviceMesh,
    pp_axis_name: str,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: Optional[list[list[str]]] = None,
    layers_per_stage: Optional[int] = None,
    patch_inner_model: bool = True,
    patch_causal_lm_model: bool = True,
    round_to_pp_multiple: str | None = None,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    Splits a HuggingFace model for pipeline parallelism.

    Args:
        model: The HuggingFace model to split
        pp_mesh: Pipeline parallel device mesh
        pp_schedule: Name of pipeline parallelism schedule
        device: Device to place stages on
        module_names_per_stage: Optional manual specification of modules per stage
        num_stages: Number of pipeline stages (used if module_names_per_stage not provided)

    Returns:
        Tuple of (stages, models) where stages are PipelineStage objects and models are the
        corresponding model chunks
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    # Detect model structure
    has_model_attr = hasattr(model, "model")
    has_rotary_emb = hasattr(model.model, "rotary_emb") if has_model_attr else hasattr(model, "rotary_emb")
    has_lm_head = hasattr(model, "lm_head")

    if has_model_attr:
        # Models like LlamaForCausalLM have model.layers
        num_layers = len(model.model.layers)
    else:
        # Direct model access
        num_layers = len(model.layers)

    schedule_class = get_schedule_class(pp_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)

    # Calculate number of virtual stages
    num_virtual_stages, _ = calculate_virtual_stages(
        num_layers=num_layers,
        layers_per_stage=layers_per_stage,
        pp_size=pp_size,
        is_single_stage_schedule=is_single_stage_schedule,
        round_to_pp_multiple=round_to_pp_multiple,
    )

    # Auto-generate module split if not provided
    if module_names_per_stage is None:
        module_names_per_stage = generate_hf_model_fqn_per_model_part(
            num_stages=num_virtual_stages,
            num_layers=num_layers,
            include_embeddings=True,
            include_lm_head=has_lm_head,
            include_rotary_emb=has_rotary_emb,
            fqn_prefix="model." if has_model_attr else "",
        )

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        """Build a pipeline stage from specified module names."""
        # Deep copy the model
        stage_model = copy.deepcopy(model)
        patch_hf_model_for_pp(
            stage_model, patch_inner_model=patch_inner_model, patch_causal_lm_model=patch_causal_lm_model
        )
        # Create a set of modules to keep
        modules_to_keep = set(module_names)
        logger.info(
            f"PP Rank {pp_rank}: Stage {stage_idx}: Keeping modules: {sorted(modules_to_keep, key=lambda x: x.split('.')[-1])}"
        )

        # Helper function to handle nested module removal
        def _process_module(parent_module, parent_name=""):
            for name, module in list(parent_module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name

                # Special handling for layers (ModuleList)
                if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
                    # Determine which layers to keep
                    layers_to_keep = {
                        name.split(".")[-1] for name in modules_to_keep if name.startswith(f"{full_name}.")
                    }
                    # Create new ModuleList with only kept layers
                    if layers_to_keep:
                        # Keep only specified layers
                        if isinstance(module, nn.ModuleDict):
                            for layer_name in list(module.keys()):
                                if layer_name not in layers_to_keep:
                                    del module[layer_name]
                        elif isinstance(module, nn.ModuleList):
                            indices_to_keep = {int(idx) for idx in layers_to_keep if idx.isdigit()}
                            new_layers = nn.ModuleDict(
                                {str(i): layer for i, layer in enumerate(module) if i in indices_to_keep}
                            )
                            setattr(parent_module, name, new_layers)
                    else:
                        # No layers from this structure needed, set to empty structure
                        if isinstance(module, nn.ModuleDict):
                            setattr(parent_module, name, nn.ModuleDict())
                        elif isinstance(module, nn.ModuleList):
                            setattr(parent_module, name, nn.ModuleDict())

                # Handle other modules
                elif full_name not in modules_to_keep and not any(
                    kept_name.startswith(full_name + ".") for kept_name in modules_to_keep
                ):
                    # This module and its children are not needed
                    setattr(parent_module, name, None)
                else:
                    # Recursively process children
                    _process_module(module, full_name)

        # Process the model
        _process_module(stage_model)

        # Create pipeline stage
        stage = PipelineStage(
            stage_model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group(pp_axis_name),
        )

        return stage, stage_model

    # Determine which stages this rank will handle
    schedule_class = get_schedule_class(pp_schedule)
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    stages = []
    models = []

    total_stages = len(module_names_per_stage)
    assert total_stages % pp_size == 0, f"Total stages {total_stages} must be divisible by PP size {pp_size}"
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, total_stages, style=style):
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            total_stages,
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models


def build_pipeline_schedule(
    pipeline_parallel_schedule_csv: str | None,
    pipeline_parallel_schedule: str | None,
    microbatch_size: int,
    local_batch_size: int,
    stages: list[PipelineStage],
    loss_fn: Callable,
    scale_grads: bool = False,
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        pipeline_parallel_schedule_csv (str | None): The path to the pipeline parallel schedule csv file.
        pipeline_parallel_schedule (str | None): The name of the pipeline parallel schedule.
        microbatch_size (int): The microbatch size.
        local_batch_size (int): The local batch size.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(f"The specified path {pp_schedule_csv} does not exist or is not a file.")
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(pipeline_parallel_schedule)

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    n_microbatches = local_batch_size // microbatch_size
    # validate that the batch size is divisible by the microbatch_size otherwise we'll hang or error during training
    if local_batch_size % microbatch_size != 0:
        raise ValueError(
            f"Batch size {local_batch_size} must be divisible by number of microbatches {n_microbatches}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatch_size."
        )

    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = len(stages)
    if n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=scale_grads,
    )
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        schedule._load_csv(pp_schedule_csv)

    return schedule


def pipeline_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh,
    *,
    pp_axis_name: str,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    layers_per_stage: int | None,
    pipeline_parallel_schedule_csv: str | None,
    pipeline_parallel_schedule: str | None,
    microbatch_size: int,
    local_batch_size: int,
    device: torch.device,
    loss_fn: Callable = None,
    parallelize_fn: Callable | None = None,
    module_fqns_per_model_part: list[list[str]] | None = None,
    patch_inner_model: bool = True,
    patch_causal_lm_model: bool = True,
    scale_grads: bool = False,
    round_to_pp_multiple: str | None = None,
) -> tuple[_PipelineSchedule, list[torch.nn.Module], bool, bool, list[PipelineStage]]:
    """HF-specific pipeline model splitting."""
    pp_size = world_mesh[pp_axis_name].size()
    assert pp_size > 1, "Pipeline parallelism is not enabled"

    # Use HF-specific pipeline split
    stages, model_parts = split_model_into_stages(
        model,
        world_mesh[pp_axis_name],
        pp_axis_name,
        pipeline_parallel_schedule,
        device,
        module_fqns_per_model_part,
        layers_per_stage=layers_per_stage,
        patch_inner_model=patch_inner_model,
        patch_causal_lm_model=patch_causal_lm_model,
        round_to_pp_multiple=round_to_pp_multiple,
    )

    # Apply parallelization if provided
    for i, m in enumerate(model_parts):
        if parallelize_fn is not None:
            parallelize_fn(
                m,
                world_mesh=world_mesh,
                moe_mesh=moe_mesh,
                pp_enabled=True,
                dp_axis_names=dp_axis_names,
                cp_axis_name=cp_axis_name,
                tp_axis_name=tp_axis_name,
                ep_axis_name=ep_axis_name,
                ep_shard_axis_names=ep_shard_axis_names,
            )
            model_parts[i] = m
            stages[i].submod = m

    # Build pipeline schedule
    pp_schedule = build_pipeline_schedule(
        pipeline_parallel_schedule_csv,
        pipeline_parallel_schedule,
        microbatch_size,
        local_batch_size,
        stages,
        loss_fn,
        scale_grads=scale_grads,
    )

    # Determine if this rank has first/last stage
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage, stages
