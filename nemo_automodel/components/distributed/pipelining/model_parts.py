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

import copy
import functools
import logging
from dataclasses import dataclass
from typing import Protocol

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import PipelineScheduleSingle

from nemo_automodel.components.distributed.pipelining.hf_utils import (
    MULTIMODAL_SUFFIXES,
    TEXT_MODULE_ATTRS,
    get_text_module,
    model_keeps_self_forward,
    patch_hf_model_for_pp,
)
from nemo_automodel.components.distributed.pipelining.module_plan import (
    PipelineStagePlanError,
    calculate_virtual_stages,
    drop_absent_speculative_modules,
    generate_hf_model_fqn_per_model_part,
    stage_ids_this_rank,
    validate_stage_plan,
)
from nemo_automodel.components.distributed.pipelining.schedules import resolve_pipeline_schedule
from nemo_automodel.shared.pipeline import PipelineModelMixin

logger = logging.getLogger(__name__)


class ParallelizeFnProtocol(Protocol):
    """Callable protocol for applying distributed parallelism to a model."""

    def __call__(
        self,
        model: nn.Module,
        world_mesh: DeviceMesh,
        moe_mesh: DeviceMesh | None,
        *,
        dp_axis_names: tuple[str, ...],
        cp_axis_name: str | None = None,
        tp_axis_name: str | None = None,
        ep_axis_name: str | None = None,
        ep_shard_axis_names: tuple[str, ...] | None = None,
    ) -> nn.Module | None: ...


@dataclass(frozen=True)
class PipelineModelPart:
    """A materialized model part before PyTorch pipeline runtime construction."""

    module: nn.Module
    stage_index: int
    num_stages: int

    @property
    def is_first(self) -> bool:
        """Whether this part is the first global pipeline stage."""
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this part is the last global pipeline stage."""
        return self.stage_index == self.num_stages - 1


def parallelize_model_parts(
    parts: list[PipelineModelPart],
    parallelize_fn: ParallelizeFnProtocol | None,
    *,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None,
    tp_axis_name: str | None,
    ep_axis_name: str | None,
    ep_shard_axis_names: tuple[str, ...] | None,
) -> list[PipelineModelPart]:
    """Apply distributed parallelism while preserving each part's stage identity."""
    if parallelize_fn is None:
        return parts

    parallelized_parts = []
    for part in parts:
        result = parallelize_fn(
            part.module,
            world_mesh=world_mesh,
            moe_mesh=moe_mesh,
            dp_axis_names=dp_axis_names,
            cp_axis_name=cp_axis_name,
            tp_axis_name=tp_axis_name,
            ep_axis_name=ep_axis_name,
            ep_shard_axis_names=ep_shard_axis_names,
        )
        module = part.module if result is None else result
        if not isinstance(module, nn.Module):
            raise TypeError("parallelize_fn must return a torch.nn.Module or None for in-place updates")
        parallelized_parts.append(PipelineModelPart(module, part.stage_index, part.num_stages))
    return parallelized_parts


def _wrap_stage_forward_to_emit_tensor(stage_model: nn.Module) -> None:
    """Unwrap ``ModelOutput.logits`` so a pipeline stage always emits tensor leaves."""
    from transformers.modeling_outputs import ModelOutput

    original_forward = stage_model.forward
    if getattr(original_forward, "_pp_unwraps_model_output", False):
        return

    @functools.wraps(original_forward)
    def _pp_tensor_forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        return output.logits if isinstance(output, ModelOutput) else output

    _pp_tensor_forward._pp_unwraps_model_output = True
    stage_model.forward = _pp_tensor_forward


def _prune_to_modules(stage_model: nn.Module, modules_to_keep: set[str], parent_name: str = "") -> None:
    """Remove modules not owned by one pipeline part."""
    for name, module in list(stage_model.named_children()):
        full_name = f"{parent_name}.{name}" if parent_name else name
        if full_name in modules_to_keep:
            continue

        if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
            children_to_keep = {fqn.rsplit(".", 1)[-1] for fqn in modules_to_keep if fqn.startswith(f"{full_name}.")}
            if isinstance(module, nn.ModuleDict):
                for child_name in list(module):
                    if child_name not in children_to_keep:
                        del module[child_name]
            else:
                indices_to_keep = {int(index) for index in children_to_keep if index.isdigit()}
                setattr(
                    stage_model,
                    name,
                    nn.ModuleDict(
                        {str(index): child for index, child in enumerate(module) if index in indices_to_keep}
                    ),
                )
        elif any(fqn.startswith(f"{full_name}.") for fqn in modules_to_keep):
            _prune_to_modules(module, modules_to_keep, full_name)
        else:
            setattr(stage_model, name, None)


def _materialize_model_part(
    model: nn.Module,
    module_names: list[str],
    *,
    stage_index: int,
    num_stages: int,
    pp_rank: int,
    patch_inner_model: bool,
    patch_causal_lm_model: bool,
) -> PipelineModelPart:
    """Deep-copy and prune a model to the modules owned by one stage."""
    stage_model = copy.deepcopy(model)
    if not model_keeps_self_forward(stage_model):
        patch_hf_model_for_pp(
            stage_model,
            patch_inner_model=patch_inner_model,
            patch_causal_lm_model=patch_causal_lm_model,
        )

    modules_to_keep = set(module_names)
    logger.info(
        "PP rank %d stage %d keeps modules: %s",
        pp_rank,
        stage_index,
        sorted(modules_to_keep, key=lambda name: name.rsplit(".", 1)[-1]),
    )
    _prune_to_modules(stage_model, modules_to_keep)
    _wrap_stage_forward_to_emit_tensor(stage_model)
    return PipelineModelPart(stage_model, stage_index, num_stages)


def _transformer_layer_fqns(
    model: nn.Module,
    layers: nn.Module,
    module_names_per_stage: list[list[str]],
) -> list[str]:
    """Return the FQN of every transformer layer, using the path the plan refers to.

    A model may expose its text stack under more than one attribute (for example a
    ``language_model`` property aliasing ``model``), so the container is located in
    the real module tree and the path that the stage plan actually addresses is
    preferred.

    Args:
        model: Un-split model that owns the transformer layers.
        layers: Container holding the transformer layers in model order.
        module_names_per_stage: Module FQNs owned by each global pipeline stage.

    Returns:
        One FQN per transformer layer, in model order.

    Raises:
        PipelineStagePlanError: If the layer container is not part of ``model``.
    """
    planned = {name for stage_modules in module_names_per_stage for name in stage_modules}
    candidates = [name for name, module in model.named_modules(remove_duplicate=False) if module is layers]
    if not candidates:
        raise PipelineStagePlanError(
            "Could not locate the transformer layer container inside the model's module tree; "
            "the pipeline stage plan cannot be validated against it."
        )
    layers_fqn = max(candidates, key=lambda prefix: sum(name.startswith(f"{prefix}.") for name in planned))
    return [f"{layers_fqn}.{index}" for index in range(len(layers))]


def split_model_into_parts(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    pp_schedule: str,
    module_names_per_stage: list[list[str]] | None = None,
    layers_per_stage: int | None = None,
    patch_inner_model: bool = True,
    patch_causal_lm_model: bool = True,
    round_to_pp_multiple: str | None = None,
) -> list[PipelineModelPart]:
    """Split a HuggingFace model into the model parts owned by this rank.

    The resulting stage plan is validated against the model's real module tree:
    speculative multimodal FQNs the model does not own are dropped, every
    remaining FQN must resolve to a submodule, and the stages must cover every
    transformer layer exactly once.

    Args:
        model: Un-split model to partition.
        pp_mesh: Pipeline-parallel device mesh.
        pp_schedule: Name of the pipeline schedule that consumes the stages.
        module_names_per_stage: Explicit module FQNs per global stage; generated
            from the model when omitted.
        layers_per_stage: Transformer layers per generated stage.
        patch_inner_model: Whether to patch the inner model's forward.
        patch_causal_lm_model: Whether to patch the causal-LM wrapper's forward.
        round_to_pp_multiple: ``"up"`` or ``"down"`` rounding of the virtual
            stage count to a multiple of the pipeline size.

    Returns:
        The model parts owned by this pipeline rank.

    Raises:
        ValueError: If the stage count is not divisible by the pipeline size.
        PipelineStagePlanError: If the stage plan does not match the model.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    has_model_attr = hasattr(model, "model")
    model_container = model.model if has_model_attr else model
    text_model = get_text_module(model_container)

    text_model_attr_name = next(
        (name for name in TEXT_MODULE_ATTRS if hasattr(model, name) or hasattr(model_container, name)),
        "",
    )
    text_model_has_model_attr = hasattr(text_model, "model")
    layers = text_model.model.layers if text_model_has_model_attr else text_model.layers
    schedule_class, stage_style = resolve_pipeline_schedule(pp_schedule)

    base_prefix = "model." if has_model_attr else ""
    text_model_attr_prefix = f"{text_model_attr_name}." if text_model_attr_name else ""
    text_model_prefix = f"{base_prefix}{text_model_attr_prefix}"
    layers_prefix = f"{text_model_prefix}model." if text_model_has_model_attr else text_model_prefix

    if module_names_per_stage is None:
        num_virtual_stages, _ = calculate_virtual_stages(
            num_layers=len(layers),
            layers_per_stage=layers_per_stage,
            pp_size=pp_size,
            is_single_stage_schedule=issubclass(schedule_class, PipelineScheduleSingle),
            round_to_pp_multiple=round_to_pp_multiple,
        )
        nested_text_model = layers_prefix != base_prefix
        # ``getattr(..., None) is not None`` rather than ``hasattr``: a module attribute
        # that is present but None owns no submodule, so planning it would name an FQN
        # that cannot resolve.
        text_model_has_lm_head = getattr(text_model, "lm_head", None) is not None
        lm_head_on_top_level = getattr(model, "lm_head", None) is not None and not text_model_has_lm_head
        lm_head_fqn = "lm_head" if not nested_text_model or lm_head_on_top_level else f"{text_model_prefix}lm_head"
        module_names_per_stage = generate_hf_model_fqn_per_model_part(
            num_stages=num_virtual_stages,
            num_layers=len(layers),
            include_lm_head=text_model_has_lm_head or lm_head_on_top_level,
            include_rotary_emb=getattr(text_model, "rotary_emb", None) is not None,
            include_multimodal_encoders=not nested_text_model,
            extra_module_fqns=(
                [f"{base_prefix}{suffix}" for suffix in MULTIMODAL_SUFFIXES] if nested_text_model else None
            ),
            fqn_prefix=layers_prefix,
            lm_head_fqn=lm_head_fqn,
        )
        if isinstance(model, PipelineModelMixin):
            module_names_per_stage = model.pipeline_stage_modules(
                module_names_per_stage,
                layers_prefix=layers_prefix,
                text_model=text_model,
            )

    present_module_fqns = {name for name, _ in model.named_modules(remove_duplicate=False) if name}
    module_names_per_stage = drop_absent_speculative_modules(module_names_per_stage, present_module_fqns)
    validate_stage_plan(
        module_names_per_stage,
        present_module_fqns=present_module_fqns,
        layer_fqns=_transformer_layer_fqns(model, layers, module_names_per_stage),
    )

    total_stages = len(module_names_per_stage)
    if total_stages % pp_size != 0:
        raise ValueError(f"Total stages {total_stages} must be divisible by PP size {pp_size}")

    return [
        _materialize_model_part(
            model,
            module_names_per_stage[stage_index],
            stage_index=stage_index,
            num_stages=total_stages,
            pp_rank=pp_rank,
            patch_inner_model=patch_inner_model,
            patch_causal_lm_model=patch_causal_lm_model,
        )
        for stage_index in stage_ids_this_rank(pp_rank, pp_size, total_stages, style=stage_style)
    ]
