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

"""Infrastructure for resolving model-owned tensor-parallel plans."""

from __future__ import annotations

import importlib
import logging
from types import FunctionType
from typing import Any

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

logger = logging.getLogger(__name__)


def translate_to_torch_parallel_style(style: str) -> ParallelStyle | None:
    """Translate a Hugging Face TP style into its PyTorch equivalent."""
    if style == "colwise":
        return ColwiseParallel()
    if style == "rowwise":
        return RowwiseParallel()
    if style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    if style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    if style == "sequence_parallel":
        return SequenceParallel()
    if style == "replicated_with_grad_allreduce":
        return None
    raise ValueError(f"Unknown parallel style: {style}")


def get_hf_tp_shard_plan(model: nn.Module) -> dict[str, ParallelStyle]:
    """Translate a model's native Hugging Face TP plan without architecture imports."""
    inner_model, model_prefix = _get_hf_tp_plan_target(model)
    model_cls = type(model)
    hf_tp_plan: dict[str, str | ParallelStyle] = {}

    if isinstance(getattr(model_cls, "_tp_plan", None), dict):
        hf_tp_plan.update(model_cls._tp_plan)
    if isinstance(getattr(model, "_tp_plan", None), dict):
        hf_tp_plan.update(model._tp_plan)
    if isinstance(getattr(inner_model, "_tp_plan", None), dict):
        prefix = f"{model_prefix}." if model_prefix else ""
        hf_tp_plan.update({f"{prefix}{key}": value for key, value in inner_model._tp_plan.items()})

    if not hf_tp_plan:
        raise AssertionError(f"Hugging Face tp plan is not supported for {model_cls.__name__}")

    embed_tokens = f"{model_prefix}.embed_tokens" if model_prefix else "embed_tokens"
    hf_tp_plan.setdefault(embed_tokens, "rowwise_rep")

    translated_plan: dict[str, ParallelStyle] = {}
    for key, value in hf_tp_plan.items():
        if isinstance(value, str) and (
            value.startswith("ep_") or value in {"local_colwise", "local_rowwise", "gather"}
        ):
            continue
        if key in {"lm_head", "language_model.lm_head"} and value == "colwise_rep":
            translated_plan[key] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
            continue
        if not isinstance(value, str):
            translated_plan[key] = value
            continue
        style = translate_to_torch_parallel_style(value)
        if style is not None:
            translated_plan[key] = style

    return translated_plan


def _get_hf_tp_plan_target(model: nn.Module) -> tuple[nn.Module, str]:
    """Find the inner transformer and its module prefix structurally."""
    nested_candidates = (
        ("model", "language_model"),
        ("language_model", "model"),
        ("language_model",),
        ("model",),
    )
    for path in nested_candidates:
        candidate: Any = model
        for name in path:
            candidate = getattr(candidate, name, None)
            if candidate is None:
                break
        if candidate is not None:
            return candidate, ".".join(path)
    return model, ""


def get_tp_plan(
    model: nn.Module,
    *,
    sequence_parallel: bool = False,
    tp_shard_plan: dict[str, ParallelStyle] | str | None = None,
    tp_size: int = 1,
    hf_plan_resolver=get_hf_tp_shard_plan,
) -> dict[str, ParallelStyle]:
    """Resolve a caller-provided, model-owned, or native Hugging Face TP plan."""
    if isinstance(tp_shard_plan, dict):
        return tp_shard_plan
    if isinstance(tp_shard_plan, str):
        return _get_requested_tp_plan(model, tp_shard_plan, sequence_parallel)

    plan_factory = getattr(model, "_nemo_tp_plan_factory", None)
    if callable(plan_factory):
        try:
            plan = plan_factory(model, sequence_parallel=sequence_parallel)
            if isinstance(plan, dict):
                return plan
            raise TypeError(f"_nemo_tp_plan_factory returned {type(plan)!r}, expected dict")
        except Exception as error:
            logger.info("Model-owned TP plan was unavailable for %s: %s", type(model).__name__, error)

    try:
        plan = hf_plan_resolver(model)
    except Exception as error:
        if tp_size <= 1:
            return {}
        raise ValueError(
            f"No usable tensor-parallel plan is available for {type(model).__name__}. "
            "Add a model-local `parallelizer.py`, define a Hugging Face `_tp_plan`, or pass `tp_shard_plan`."
        ) from error
    if not plan and tp_size > 1:
        raise ValueError(f"The Hugging Face TP plan for {type(model).__name__} translated to an empty plan")
    return plan


def _get_requested_tp_plan(
    model: nn.Module,
    tp_shard_plan: str,
    sequence_parallel: bool,
) -> dict[str, ParallelStyle]:
    """Resolve a caller-provided plan path or a model-local named plan."""
    if "." not in tp_shard_plan:
        plan_factory = getattr(model, "_nemo_tp_plan_factory", None)
        if callable(plan_factory):
            plan = plan_factory(model, sequence_parallel=sequence_parallel)
            if isinstance(plan, dict):
                return plan
        raise ValueError(
            f"Named tensor-parallel plan {tp_shard_plan!r} requires a model-local `_nemo_tp_plan_factory`."
        )

    try:
        module_name, attr_name = tp_shard_plan.rsplit(".", 1)
        plan_obj = getattr(importlib.import_module(module_name), attr_name)
        plan = plan_obj() if isinstance(plan_obj, FunctionType) else plan_obj
    except Exception as error:
        raise ValueError(f"Custom parallel plan {tp_shard_plan!r} is not valid: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError(f"Custom parallel plan must be a dictionary, got {type(plan)!r}")
    return plan
