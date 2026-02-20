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
import types
from typing import Callable, Optional, Protocol

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

from nemo_automodel.components.distributed.pipelining.hf_utils import (
    MULTIMODAL_SUFFIXES,
    TEXT_MODULE_ATTRS,
    get_text_module,
    patch_hf_model_for_pp,
)

logger = logging.getLogger(__name__)


class ParallelizeFnProtocol(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        world_mesh: DeviceMesh,
        moe_mesh: DeviceMesh,
        *,
        dp_axis_names: tuple[str, ...],
        cp_axis_name: str | None = None,
        tp_axis_name: str | None = None,
        ep_axis_name: str | None = None,
        ep_shard_axis_names: tuple[str, ...] | None = None,
    ) -> None: ...


@torch.no_grad()
def scale_grads_by_divisor(
    stages: list[PipelineStage],
    divisor: int,
) -> None:
    for stage in stages:
        if hasattr(stage, "scale_grads"):
            stage.scale_grads(divisor)


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
    raise ValueError(f"Unsupported pipeline schedule style: {style}. Expected one of ['loop', 'v'].")


def _is_truthy_env(var_name: str, default: str = "0") -> bool:
    return os.environ.get(var_name, default).strip().lower() in ("1", "true", "yes", "on")


def _enable_skip_output_merge_if_supported(schedule: _PipelineSchedule) -> bool:
    """Patch schedule.step to skip output merge when schedule internals are compatible.

    This avoids large transient output concatenations on the last PP stage, while
    isolating reliance on private schedule internals to a single guarded path.
    """
    required_attrs = ("step", "_has_backward", "_split_inputs", "_step_microbatches", "_n_microbatches")
    missing = [name for name in required_attrs if not hasattr(schedule, name)]
    has_stage_attr = hasattr(schedule, "_stages") or hasattr(schedule, "_stage")
    if missing or not has_stage_attr:
        logger.warning(
            "NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE requested, but schedule class %s is incompatible "
            "(missing attrs=%s, has_stage_attr=%s). Running without patch.",
            schedule.__class__.__name__,
            missing,
            has_stage_attr,
        )
        return False

    original_step = schedule.step

    def _step_without_output_merge(self, *args, target=None, losses=None, **kwargs):
        if self._has_backward and not torch.is_grad_enabled():
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        if hasattr(self, "_stages"):
            for stage in self._stages:
                stage.has_backward = self._has_backward
            for stage in self._stages:
                stage.clear_runtime_states()
        else:
            self._stage.has_backward = self._has_backward
            self._stage.clear_runtime_states()

        args_split, kwargs_split = self._split_inputs(args, kwargs)
        targets_split = list(torch.tensor_split(target, self._n_microbatches)) if target is not None else None
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)
        return None

    schedule.step = types.MethodType(_step_without_output_merge, schedule)
    # Keep a handle for debugging/reversion if needed.
    schedule._nemo_original_step = original_step
    logger.info(
        "Enabled NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE=1: replacing schedule.step output-merge path "
        "(class=%s).",
        schedule.__class__.__name__,
    )
    return True


def generate_hf_model_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    include_embeddings: bool = True,
    include_lm_head: bool = True,
    include_rotary_emb: bool = True,
    include_multimodal_encoders: bool = True,
    extra_module_fqns: Optional[list[str]] = None,
    fqn_prefix: str = "model.",
    lm_head_fqn: str = "lm_head",
    embed_module_name: str = "embed_tokens",
    layers_module_name: str = "layers",
    norm_module_name: str = "norm",
) -> list[list[str]]:
    """
    Generates module names for each pipeline stage for HuggingFace models.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        include_embeddings: Whether to include embedding layer in first stage
        include_lm_head: Whether to include lm_head in last stage (for CausalLM models)
        include_multimodal_encoders: Whether to include common vision/audio encoder modules in stage 0
        extra_module_fqns: Optional list of extra module FQNs to include in stage 0

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

        # First stage: add embeddings and multimodal encoders if requested
        if stage_idx == 0:
            if include_embeddings:
                stage_modules.append(f"{fqn_prefix}{embed_module_name}")
            if include_multimodal_encoders:
                stage_modules.extend([f"{fqn_prefix}{suffix}" for suffix in MULTIMODAL_SUFFIXES])
            if extra_module_fqns:
                stage_modules.extend(extra_module_fqns)

        # Add transformer layers for this stage
        for _ in range(stage_layer_count):
            stage_modules.append(f"{fqn_prefix}{layers_module_name}.{current_layer}")
            current_layer += 1

        # Last stage: add norm and lm_head if requested
        if stage_idx == num_stages - 1:
            stage_modules.append(f"{fqn_prefix}{norm_module_name}")
            if include_lm_head:
                stage_modules.append(lm_head_fqn)

        if include_rotary_emb:
            # Always include rotary_emb in all stages (it's needed for position embeddings)
            stage_modules.append(f"{fqn_prefix}rotary_emb")

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def _parse_pp_block_weights() -> dict[str, float]:
    """Parse optional PP block-weight overrides from env.

    Format:
      NEMOAUTOMODEL_PP_BLOCK_WEIGHTS="moe=3,attention=2,mamba=2,mlp=1"
    """
    defaults = {
        "moe": 3.0,
        "attention": 2.0,
        "mamba": 2.0,
        "mlp": 1.0,
    }
    raw = os.environ.get("NEMOAUTOMODEL_PP_BLOCK_WEIGHTS", "")
    if not raw:
        return defaults

    parsed = dict(defaults)
    for part in raw.split(","):
        token = part.strip()
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        try:
            parsed[key] = float(value.strip())
        except ValueError:
            logger.warning("Ignoring invalid PP block weight token: %s", token)
    return parsed


def generate_hf_model_fqn_per_model_part_weighted(
    num_stages: int,
    layer_weights: list[float],
    include_embeddings: bool = True,
    include_lm_head: bool = True,
    include_rotary_emb: bool = True,
    fqn_prefix: str = "model.",
    embed_module_name: str = "embed_tokens",
    layers_module_name: str = "layers",
    norm_module_name: str = "norm",
) -> list[list[str]]:
    """Generate contiguous stage splits using per-layer weights.

    This is useful for heterogeneous stacks (e.g., NemotronH mamba/attention/moe)
    where equal layer-count partitioning can still be compute-imbalanced.
    """
    num_layers = len(layer_weights)
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")
    if num_stages > num_layers:
        raise ValueError(f"Number of stages ({num_stages}) cannot exceed number of layers ({num_layers})")

    # Keep positive weights to avoid degenerate zero-target splits.
    weights = [max(float(w), 1.0e-6) for w in layer_weights]
    total_weight = sum(weights)

    module_names_per_stage = []
    start = 0
    remaining_weight = total_weight
    for stage_idx in range(num_stages):
        stage_modules = []

        if stage_idx == 0 and include_embeddings:
            stage_modules.append(f"{fqn_prefix}{embed_module_name}")

        remaining_stages = num_stages - stage_idx
        if stage_idx == num_stages - 1:
            end = num_layers
        else:
            min_end = start + 1
            max_end = num_layers - (remaining_stages - 1)
            stage_target = remaining_weight / remaining_stages

            best_end = min_end
            best_diff = float("inf")
            run = 0.0
            for end_candidate in range(min_end, max_end + 1):
                run += weights[end_candidate - 1]
                diff = abs(run - stage_target)
                if diff < best_diff:
                    best_diff = diff
                    best_end = end_candidate
            end = best_end

        for layer_idx in range(start, end):
            stage_modules.append(f"{fqn_prefix}{layers_module_name}.{layer_idx}")

        stage_weight = sum(weights[start:end])
        remaining_weight -= stage_weight
        start = end

        if stage_idx == num_stages - 1:
            stage_modules.append(f"{fqn_prefix}{norm_module_name}")
            if include_lm_head:
                stage_modules.append("lm_head")

        if include_rotary_emb:
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
        # Note: embeddings and lm_head are added to first/last stages, not counted separately
        num_virtual_stages = math.ceil(num_layers / layers_per_stage)

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


def _precompute_stage_shapes(
    stages: list[PipelineStage],
    model_config,
    microbatch_size: int,
    seq_len: int,
) -> None:
    """Precompute input/output meta tensors for each pipeline stage to bypass serial shape inference.

    By default, PipelineStage performs shape inference at runtime via a serial P2P chain:
    stage 0 → send → stage 1 → send → ... → stage N-1.  This is O(N) in the number of
    pipeline stages and becomes a bottleneck for large world sizes.

    This function sets ``inputs_meta`` and ``_outputs_meta`` on each stage *before* the
    first ``step()`` call, so that ``_shape_inference`` is never invoked and the serial
    chain is completely eliminated.

    Args:
        stages: The local pipeline stages (already parallelized).
        model_config: The HuggingFace model config (``model.config``).
        microbatch_size: Microbatch size used by the pipeline schedule.
        seq_len: Sequence length of the input data.
    """
    hidden_size = model_config.hidden_size
    vocab_size = model_config.vocab_size

    for stage in stages:
        # Infer the computation dtype from the stage's parameters
        try:
            model_dtype = next(stage.submod.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16

        # --- inputs_meta ---
        if stage.is_first:
            # First stage receives input_ids: [mb, seq_len] int64
            stage.inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            # Non-first stages receive hidden_states: [mb, seq_len, hidden_size]
            stage.inputs_meta = (torch.empty(microbatch_size, seq_len, hidden_size, device="meta", dtype=model_dtype),)

        # --- outputs_meta ---
        has_lm_head = hasattr(stage.submod, "lm_head") and stage.submod.lm_head is not None
        if has_lm_head:
            # Last stage with lm_head produces logits: [mb, seq_len, vocab_size]
            outputs_meta = (torch.empty(microbatch_size, seq_len, vocab_size, device="meta", dtype=model_dtype),)
        else:
            # Intermediate stages produce hidden_states: [mb, seq_len, hidden_size]
            outputs_meta = (torch.empty(microbatch_size, seq_len, hidden_size, device="meta", dtype=model_dtype),)
        stage._configure_outputs_meta(outputs_meta)

    logger.info(
        f"Precomputed pipeline stage shapes (seq_len={seq_len}, microbatch_size={microbatch_size}) — "
        f"serial shape inference bypassed"
    )


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
    has_model_attr = hasattr(model, "model") and getattr(model, "model", None) is not None
    has_backbone_attr = (not has_model_attr) and hasattr(model, "backbone") and getattr(model, "backbone", None) is not None
    if has_backbone_attr:
        text_model = model.backbone
        text_model_attr_name = ""
    elif has_model_attr:
        text_model = get_text_module(model.model)
        text_model_attr_name = ""
        for attr_name in TEXT_MODULE_ATTRS:
            if hasattr(model.model, attr_name):
                text_model_attr_name = attr_name
                break
    else:
        text_model = get_text_module(model)
        text_model_attr_name = ""
        for attr_name in TEXT_MODULE_ATTRS:
            if hasattr(model, attr_name):
                text_model_attr_name = attr_name
                break
    has_rotary_emb = hasattr(text_model, "rotary_emb")

    # Check for lm_head in multiple locations:
    has_lm_head = hasattr(text_model, "lm_head") or hasattr(model, "lm_head")
    lm_head_on_top_level = hasattr(model, "lm_head") and not hasattr(text_model, "lm_head")

    text_model_has_model_attr = hasattr(text_model, "model")

    if has_backbone_attr:
        # NemotronH-like models expose decoder stack as backbone.layers.
        num_layers = len(model.backbone.layers)
        embed_module_name = "embeddings"
        layers_module_name = "layers"
        norm_module_name = "norm_f"
    else:
        if text_model_has_model_attr:
            # Models like LlamaForCausalLM have model.layers
            num_layers = len(text_model.model.layers)
        else:
            # Direct model access
            num_layers = len(text_model.layers)
        embed_module_name = "embed_tokens"
        layers_module_name = "layers"
        norm_module_name = "norm"

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

    # Determine module prefix for text layers and where multimodal encoders live.
    if has_backbone_attr:
        base_prefix = ""
        layers_prefix = "backbone."
        include_multimodal_encoders = False
        extra_module_fqns = None
        lm_head_fqn = "lm_head"
    else:
        base_prefix = "model." if has_model_attr else ""
        layers_prefix = base_prefix
        include_multimodal_encoders = True
        extra_module_fqns = None

        text_model_attr_prefix = text_model_attr_name + "." if text_model_attr_name else ""
        layers_prefix = (
            f"{base_prefix}{text_model_attr_prefix}model."
            if text_model_has_model_attr
            else f"{base_prefix}{text_model_attr_prefix}"
        )

        # If layers live under a nested language_model, keep multimodal encoders at the base prefix
        if layers_prefix != base_prefix:
            include_multimodal_encoders = False
            extra_module_fqns = [f"{base_prefix}{suffix}" for suffix in MULTIMODAL_SUFFIXES]
            if lm_head_on_top_level:
                lm_head_fqn = "lm_head"
            else:
                lm_head_fqn = f"{base_prefix}{text_model_attr_name}.lm_head"
        else:
            lm_head_fqn = "lm_head"

    # Auto-generate module split if not provided
    if module_names_per_stage is None:
        use_weighted_split = os.environ.get("NEMOAUTOMODEL_PP_USE_WEIGHTED_SPLIT", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        weighted_layer_weights = None
        if use_weighted_split:
            if has_backbone_attr:
                layers_obj = model.backbone.layers
            elif has_model_attr:
                layers_obj = text_model.model.layers if hasattr(text_model, "model") else text_model.layers
            else:
                layers_obj = text_model.model.layers if hasattr(text_model, "model") else text_model.layers
            layers_iter = layers_obj.values() if hasattr(layers_obj, "values") else layers_obj
            block_types = [getattr(layer, "block_type", None) for layer in layers_iter]
            if all(bt is not None for bt in block_types):
                weight_map = _parse_pp_block_weights()
                weighted_layer_weights = [weight_map.get(str(bt).lower(), 1.0) for bt in block_types]
                logger.info(
                    "Using weighted PP split with block weights=%s for %s virtual stages.",
                    weight_map,
                    num_virtual_stages,
                )
            else:
                logger.info(
                    "Weighted PP split requested but block_type metadata is unavailable; falling back to equal-layer split."
                )

        if weighted_layer_weights is not None:
            module_names_per_stage = generate_hf_model_fqn_per_model_part_weighted(
                num_stages=num_virtual_stages,
                layer_weights=weighted_layer_weights,
                include_embeddings=True,
                include_lm_head=has_lm_head,
                include_rotary_emb=has_rotary_emb,
                fqn_prefix=layers_prefix,
                embed_module_name=embed_module_name,
                layers_module_name=layers_module_name,
                norm_module_name=norm_module_name,
            )
            if include_multimodal_encoders:
                module_names_per_stage[0].extend([f"{layers_prefix}{suffix}" for suffix in MULTIMODAL_SUFFIXES])
            if extra_module_fqns:
                module_names_per_stage[0].extend(extra_module_fqns)
            if has_lm_head and lm_head_fqn != "lm_head":
                module_names_per_stage[-1] = [
                    lm_head_fqn if module_fqn == "lm_head" else module_fqn for module_fqn in module_names_per_stage[-1]
                ]
        else:
            module_names_per_stage = generate_hf_model_fqn_per_model_part(
                num_stages=num_virtual_stages,
                num_layers=num_layers,
                include_embeddings=True,
                include_lm_head=has_lm_head,
                include_rotary_emb=has_rotary_emb,
                include_multimodal_encoders=include_multimodal_encoders,
                extra_module_fqns=extra_module_fqns,
                fqn_prefix=layers_prefix,
                lm_head_fqn=lm_head_fqn,
                embed_module_name=embed_module_name,
                layers_module_name=layers_module_name,
                norm_module_name=norm_module_name,
            )

    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"
    total_stages = len(module_names_per_stage)
    assert total_stages % pp_size == 0, f"Total stages {total_stages} must be divisible by PP size {pp_size}"
    local_stage_ids = list(stage_ids_this_rank(pp_rank, pp_size, total_stages, style=style))

    patch_template_once = os.environ.get("NEMOAUTOMODEL_PP_PATCH_TEMPLATE_ONCE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    desired_patch_cfg = (bool(patch_inner_model), bool(patch_causal_lm_model))
    if patch_template_once and any(desired_patch_cfg):
        current_patch_cfg = getattr(model, "_nemo_pp_forward_patch_cfg", None)
        if current_patch_cfg != desired_patch_cfg:
            patch_hf_model_for_pp(
                model, patch_inner_model=patch_inner_model, patch_causal_lm_model=patch_causal_lm_model
            )
            model._nemo_pp_forward_patch_cfg = desired_patch_cfg

    def _prune_model_to_modules(stage_model: nn.Module, modules_to_keep: set[str]) -> None:
        def _direct_child_names(parent_fqn: str) -> set[str]:
            """Return direct child names of parent_fqn that must be kept.

            Example:
              parent_fqn=backbone.layers
              keep entry backbone.layers.3.mixer -> direct child "3"
            """
            prefix = f"{parent_fqn}."
            children = set()
            for kept in modules_to_keep:
                if kept.startswith(prefix):
                    suffix = kept[len(prefix) :]
                    child = suffix.split(".", 1)[0]
                    if child:
                        children.add(child)
            return children

        # Helper function to handle nested module removal
        def _process_module(parent_module, parent_name=""):
            for name, module in list(parent_module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name

                if full_name in modules_to_keep:
                    continue

                # Special handling for layers (ModuleList/ModuleDict)
                if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
                    layers_to_keep = _direct_child_names(full_name)
                    if layers_to_keep:
                        if isinstance(module, nn.ModuleDict):
                            for layer_name in list(module.keys()):
                                if layer_name not in layers_to_keep:
                                    del module[layer_name]
                        elif isinstance(module, nn.ModuleList):
                            # Keep original layer ids in module names for checkpoint
                            # compatibility. Reindexing a ModuleList to [0..N-1] would
                            # change state_dict keys and break base-model load.
                            selected = {
                                str(i): module[i]
                                for i in sorted(int(idx) for idx in layers_to_keep if idx.isdigit())
                                if 0 <= i < len(module)
                            }
                            setattr(parent_module, name, nn.ModuleDict(selected))
                    else:
                        if isinstance(module, nn.ModuleDict):
                            setattr(parent_module, name, nn.ModuleDict())
                        elif isinstance(module, nn.ModuleList):
                            setattr(parent_module, name, nn.ModuleDict())

                elif (
                    full_name not in modules_to_keep
                    and not any(kept_name.startswith(full_name + ".") for kept_name in modules_to_keep)
                ):
                    setattr(parent_module, name, None)
                elif full_name not in modules_to_keep:
                    _process_module(module, full_name)

        _process_module(stage_model)

    def _annotate_stage_runtime_requirements(stage_model: nn.Module) -> None:
        # Conservative defaults: move all runtime tensors unless we can prove a stage doesn't need them.
        needs_attention_mask = True
        needs_cache_position = True
        needs_causal_mask_mapping = True

        target = None
        if hasattr(stage_model, "backbone") and getattr(stage_model, "backbone") is not None:
            target = stage_model.backbone
        elif hasattr(stage_model, "model") and getattr(stage_model, "model") is not None:
            target = stage_model.model
        elif hasattr(stage_model, "layers"):
            target = stage_model

        layers = getattr(target, "layers", None) if target is not None else None
        if layers is not None:
            try:
                layer_iter = layers.values() if hasattr(layers, "values") else layers
                block_types = [getattr(layer, "block_type", None) for layer in layer_iter if layer is not None]
            except TypeError:
                block_types = []
            if block_types:
                non_none_block_types = [bt for bt in block_types if bt is not None]
                is_nemotron_h_stage = bool(non_none_block_types) and all(
                    bt in {"mlp", "moe", "attention", "mamba"} for bt in non_none_block_types
                )
                # NemotronH-only signal: no attention/mamba blocks means these masks/positions are unused.
                if is_nemotron_h_stage:
                    if all(bt in ("mlp", "moe") for bt in non_none_block_types):
                        needs_attention_mask = False
                        needs_cache_position = False
                    # NemotronH PP path computes masks internally from attention_mask.
                    needs_causal_mask_mapping = False

        stage_model._nemo_pp_needs_attention_mask = needs_attention_mask
        stage_model._nemo_pp_needs_cache_position = needs_cache_position
        stage_model._nemo_pp_needs_causal_mask_mapping = needs_causal_mask_mapping
        if hasattr(stage_model, "backbone") and getattr(stage_model, "backbone") is not None:
            stage_model.backbone._nemo_pp_needs_attention_mask = needs_attention_mask
            stage_model.backbone._nemo_pp_needs_cache_position = needs_cache_position
            stage_model.backbone._nemo_pp_needs_causal_mask_mapping = needs_causal_mask_mapping
        if hasattr(stage_model, "model") and getattr(stage_model, "model") is not None:
            stage_model.model._nemo_pp_needs_attention_mask = needs_attention_mask
            stage_model.model._nemo_pp_needs_cache_position = needs_cache_position
            stage_model.model._nemo_pp_needs_causal_mask_mapping = needs_causal_mask_mapping

    build_from_local_union = os.environ.get("NEMOAUTOMODEL_PP_BUILD_FROM_LOCAL_UNION", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    rank_template_model = None
    if build_from_local_union and local_stage_ids:
        local_union_modules = set()
        for sid in local_stage_ids:
            local_union_modules.update(module_names_per_stage[sid])

        rank_template_model = copy.deepcopy(model)
        _prune_model_to_modules(rank_template_model, local_union_modules)
        logger.info(
            "PP Rank %s: built rank-local pruned template (local stages=%s, kept modules=%s)",
            pp_rank,
            len(local_stage_ids),
            len(local_union_modules),
        )

    reuse_rank_template_inplace = rank_template_model is not None and len(local_stage_ids) == 1

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        """Build a pipeline stage from specified module names."""
        if reuse_rank_template_inplace:
            stage_model = rank_template_model
        else:
            # Deep copy the model
            stage_model = copy.deepcopy(rank_template_model if rank_template_model is not None else model)
        if not patch_template_once and any(desired_patch_cfg):
            patch_hf_model_for_pp(
                stage_model, patch_inner_model=patch_inner_model, patch_causal_lm_model=patch_causal_lm_model
            )
        # Create a set of modules to keep
        modules_to_keep = set(module_names)
        logger.info(
            f"PP Rank {pp_rank}: Stage {stage_idx}: Keeping modules: {sorted(modules_to_keep, key=lambda x: x.split('.')[-1])}"
        )

        # Process the model
        _prune_model_to_modules(stage_model, modules_to_keep)
        _annotate_stage_runtime_requirements(stage_model)

        # Create pipeline stage
        stage = PipelineStage(
            stage_model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group(pp_axis_name),
        )

        return stage, stage_model

    stages = []
    models = []
    for stage_idx in local_stage_ids:
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

    num_local_stages = len(stages)
    num_total_stages = getattr(stages[0], "num_stages", num_local_stages) if num_local_stages > 0 else 0
    allow_underfill = _is_truthy_env("NEMOAUTOMODEL_PP_ALLOW_UNDERFILL", "0")
    strict_total_underfill = _is_truthy_env("NEMOAUTOMODEL_PP_STRICT_TOTAL_UNDERFILL", "0")

    # Hard check: if local microbatches are fewer than local stage chunks, this rank is
    # severely underfilled and usually not useful to run.
    if n_microbatches < num_local_stages and not allow_underfill:
        raise ValueError(
            f"Pipeline underfill detected: n_microbatches={n_microbatches}, local_stage_chunks={num_local_stages}, "
            f"total_stages={num_total_stages}, local_batch_size={local_batch_size}, microbatch_size={microbatch_size}. "
            "Set NEMOAUTOMODEL_PP_ALLOW_UNDERFILL=1 to override this check for exploratory runs."
        )

    # Soft check by default: compare against total virtual stages for bubble guidance.
    # Some interleaved high-batch configs can still be useful with n_microbatches < total_stages.
    if n_microbatches < num_total_stages:
        underfill_msg = (
            f"Pipeline underfill warning: n_microbatches={n_microbatches}, total_stages={num_total_stages}, "
            f"local_stage_chunks={num_local_stages}, local_batch_size={local_batch_size}, "
            f"microbatch_size={microbatch_size}. This can increase PP bubbles."
        )
        if strict_total_underfill and not allow_underfill:
            raise ValueError(
                underfill_msg
                + " Set NEMOAUTOMODEL_PP_ALLOW_UNDERFILL=1 to override, "
                + "or unset NEMOAUTOMODEL_PP_STRICT_TOTAL_UNDERFILL."
            )
        # Warnings may be globally filtered in some training setups; keep an INFO copy visible.
        logger.warning("%s", underfill_msg)
        logger.info("%s", underfill_msg)

    # Runtime schedule can execute multi-stage local chunks when provided a list.
    pass_stage_list = looped_schedule or (schedule_class == _PipelineScheduleRuntime and num_local_stages > 1)
    schedule_input = stages if pass_stage_list else stages[0]

    schedule = schedule_class(
        schedule_input,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        scale_grads=scale_grads,
    )
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches, {num_total_stages} total stages, "
        f"and {num_local_stages} local stage chunks."
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

    # Optional memory optimization for training/benchmark flows that never consume
    # the return value of schedule.step(): skip concatenating last-stage microbatch
    # outputs (e.g., full logits), which can create large transient allocations.
    if _is_truthy_env("NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE", "0"):
        _enable_skip_output_merge_if_supported(schedule)

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
    patch_stage_backward_maybe_with_nosync: bool = False,
    seq_len: int | None = None,
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
                dp_axis_names=dp_axis_names,
                cp_axis_name=cp_axis_name,
                tp_axis_name=tp_axis_name,
                ep_axis_name=ep_axis_name,
                ep_shard_axis_names=ep_shard_axis_names,
            )
            model_parts[i] = m
            stages[i].submod = m

    # Precompute stage shapes to bypass serial P2P shape inference.
    # This must happen *after* parallelization so that dtypes are final.
    if seq_len is not None:
        _precompute_stage_shapes(stages, model.config, microbatch_size, seq_len)

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

    # Patch FSDP backward for MoE models if requested
    if patch_stage_backward_maybe_with_nosync:
        from nemo_automodel.components.moe.fsdp_mixin import patched_backward_maybe_with_nosync

        for stage in stages:
            stage.backward_maybe_with_nosync = types.MethodType(patched_backward_maybe_with_nosync, stage)

        logger.info("Patched pipeline stages with MoE-aware FSDP backward logic")

    # Determine if this rank has first/last stage
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage, stages
