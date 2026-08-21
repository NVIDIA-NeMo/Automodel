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
import math

from nemo_automodel.components.distributed.pipelining.hf_utils import MULTIMODAL_SUFFIXES

logger = logging.getLogger(__name__)


class PipelineStagePlanError(ValueError):
    """Raised when a pipeline stage plan does not match the model's module tree."""


def stage_ids_this_rank(pp_rank: int, pp_size: int, num_stages: int, style: str = "loop") -> tuple[int, ...]:
    """Return the global stage IDs assigned to one pipeline rank."""
    if num_stages % pp_size != 0:
        raise ValueError(f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}")
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + stage * pp_size for stage in range(stages_per_rank))
    if style == "v":
        if stages_per_rank != 2:
            raise ValueError(f"v schedules assume 2 stages per rank, got {stages_per_rank}")
        return pp_rank, num_stages - pp_rank - 1
    raise ValueError(f"Unknown pipeline stage assignment style: {style!r}")


def generate_hf_model_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    include_embeddings: bool = True,
    include_lm_head: bool = True,
    include_rotary_emb: bool = True,
    include_multimodal_encoders: bool = True,
    extra_module_fqns: list[str] | None = None,
    fqn_prefix: str = "model.",
    lm_head_fqn: str = "lm_head",
) -> list[list[str]]:
    """Generate the module names owned by each HuggingFace pipeline stage."""
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")
    if num_stages > num_layers:
        raise ValueError(f"Number of stages ({num_stages}) cannot exceed number of layers ({num_layers})")

    layers_per_stage, extra_layers = divmod(num_layers, num_stages)
    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules: list[str] = []
        stage_layer_count = layers_per_stage + (stage_idx < extra_layers)

        if stage_idx == 0:
            if include_embeddings:
                stage_modules.append(f"{fqn_prefix}embed_tokens")
            if include_multimodal_encoders:
                stage_modules.extend(f"{fqn_prefix}{suffix}" for suffix in MULTIMODAL_SUFFIXES)
            if extra_module_fqns:
                stage_modules.extend(extra_module_fqns)

        stage_modules.extend(
            f"{fqn_prefix}layers.{layer_idx}" for layer_idx in range(current_layer, current_layer + stage_layer_count)
        )
        current_layer += stage_layer_count

        if stage_idx == num_stages - 1:
            stage_modules.append(f"{fqn_prefix}norm")
            if include_lm_head:
                stage_modules.append(lm_head_fqn)
        if include_rotary_emb:
            stage_modules.append(f"{fqn_prefix}rotary_emb")

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def drop_absent_speculative_modules(
    module_names_per_stage: list[list[str]],
    present_module_fqns: set[str],
) -> list[list[str]]:
    """Remove speculative multimodal FQNs the model does not actually own.

    :func:`generate_hf_model_fqn_per_model_part` places every name in
    ``MULTIMODAL_SUFFIXES`` on stage 0 without knowing which encoders a given
    checkpoint has, so a text-only model legitimately receives FQNs that resolve
    to nothing. Those names are dropped here, which lets every remaining name be
    treated as a hard requirement.

    Args:
        module_names_per_stage: Module FQNs owned by each global pipeline stage.
        present_module_fqns: Every module FQN that exists in the model.

    Returns:
        The stage plan without the speculative names that do not resolve.
    """
    present_leaf_names = {fqn.rsplit(".", 1)[-1] for fqn in present_module_fqns}
    kept_per_stage: list[list[str]] = []
    dropped: list[str] = []
    for stage_modules in module_names_per_stage:
        kept: list[str] = []
        for name in stage_modules:
            leaf_name = name.rsplit(".", 1)[-1]
            if name in present_module_fqns or leaf_name not in MULTIMODAL_SUFFIXES:
                kept.append(name)
                continue
            dropped.append(name)
            if leaf_name in present_leaf_names:
                logger.warning(
                    "Pipeline stage plan names %r, which does not exist, while the model does own a module named "
                    "%r elsewhere; that module is assigned to no stage and will be dropped from every stage.",
                    name,
                    leaf_name,
                )
        kept_per_stage.append(kept)
    if dropped:
        logger.debug("Dropped %d speculative multimodal FQNs from the stage plan: %s", len(dropped), sorted(dropped))
    return kept_per_stage


def validate_stage_plan(
    module_names_per_stage: list[list[str]],
    *,
    present_module_fqns: set[str],
    layer_fqns: list[str],
) -> None:
    """Validate that a stage plan matches the model and partitions its layers.

    Args:
        module_names_per_stage: Module FQNs owned by each global pipeline stage.
        present_module_fqns: Every module FQN that exists in the model.
        layer_fqns: FQN of every transformer layer, in model order.

    Raises:
        PipelineStagePlanError: If a name does not resolve to a module, if a
            stage owns no transformer layer, or if the stages do not cover every
            transformer layer exactly once.
    """
    unresolved = {
        stage_index: sorted(name for name in stage_modules if name not in present_module_fqns)
        for stage_index, stage_modules in enumerate(module_names_per_stage)
    }
    unresolved = {stage_index: names for stage_index, names in unresolved.items() if names}
    if unresolved:
        details = "; ".join(f"stage {stage_index}: {names}" for stage_index, names in sorted(unresolved.items()))
        raise PipelineStagePlanError(
            f"Pipeline stage plan names modules that do not exist in the model ({details}). "
            "Every planned FQN must resolve to a submodule, otherwise the stage silently loses that module."
        )

    layer_fqn_set = set(layer_fqns)
    owner_per_layer: dict[str, list[int]] = {fqn: [] for fqn in layer_fqns}
    empty_stages = []
    for stage_index, stage_modules in enumerate(module_names_per_stage):
        owned = [name for name in stage_modules if name in layer_fqn_set]
        if not owned:
            empty_stages.append(stage_index)
        for name in owned:
            owner_per_layer[name].append(stage_index)
    if empty_stages:
        raise PipelineStagePlanError(
            f"Pipeline stages {empty_stages} own no transformer layer; every stage must own at least one of the "
            f"{len(layer_fqns)} layers."
        )

    missing = [fqn for fqn, owners in owner_per_layer.items() if not owners]
    duplicated = [f"{fqn} -> stages {owners}" for fqn, owners in owner_per_layer.items() if len(owners) > 1]
    if missing or duplicated:
        raise PipelineStagePlanError(
            "Pipeline stages must cover every transformer layer exactly once, but layers assigned to no stage: "
            f"{missing}; layers assigned to several stages: {duplicated}."
        )


def calculate_virtual_stages(
    num_layers: int,
    layers_per_stage: int | None,
    pp_size: int,
    is_single_stage_schedule: bool,
    round_to_pp_multiple: str | None = None,
) -> tuple[int, int]:
    """Calculate virtual pipeline stages and layers per rank."""
    if layers_per_stage is None:
        stages_per_rank = 1 if is_single_stage_schedule else 2
        return pp_size * stages_per_rank, stages_per_rank

    if layers_per_stage <= 0:
        raise ValueError(f"layers_per_stage must be positive, got {layers_per_stage}")
    num_virtual_stages = math.ceil(num_layers / layers_per_stage)
    remainder = num_virtual_stages % pp_size
    if remainder:
        if round_to_pp_multiple == "up":
            num_virtual_stages += pp_size - remainder
        elif round_to_pp_multiple == "down":
            num_virtual_stages -= remainder
        elif round_to_pp_multiple is None:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by pipeline parallel size "
                f"({pp_size}); adjust layers_per_stage={layers_per_stage} or request rounding."
            )
        else:
            raise ValueError(
                f"Invalid value for round_to_pp_multiple: {round_to_pp_multiple!r}; expected 'up' or 'down'"
            )

    stages_per_rank = num_virtual_stages // pp_size
    invalid_stage_count = stages_per_rank != 1 if is_single_stage_schedule else stages_per_rank < 2
    if invalid_stage_count:
        schedule_kind = "Single stage" if is_single_stage_schedule else "Multi-stage"
        required_stages = "exactly 1 stage" if is_single_stage_schedule else "at least 2 stages"
        raise ValueError(
            f"{schedule_kind} schedule requires {required_stages} per rank, "
            f"but layers_per_stage={layers_per_stage} produces {stages_per_rank}."
        )

    return num_virtual_stages, stages_per_rank
