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

"""Model-scoped metadata for task heads installed before FSDP."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Mapping

import torch
from torch import nn

from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

__all__ = ["PreFSDPHookResult"]

_TASK_HEAD_MODULE_NAME = "_nemo_task_head_module_name"


@dataclass(frozen=True)
class PreFSDPHookResult:
    """Declare a fresh task module installed by ``pre_fsdp_hook``.

    The declared module is excluded from tensor- and expert-specific sharding,
    replicated over tensor parallelism, and placed in its own FP32 FSDP unit
    over the data/context-parallel mesh. It stays trainable under PEFT and is
    included in resumable training checkpoints. The module must be newly
    created by the hook, attached exactly once below the model root, and expose
    ``reset_parameters()`` when it is created on the meta device. Under tensor
    or sequence parallelism, the module remains responsible for accepting the
    distributed activation layout supplied by its owning model. Consolidated
    Hugging Face export remains the owning model's responsibility.

    Args:
        task_module: Fresh parameter-owning module attached to the model by the
            hook. A compound task head can be represented by one container
            module. Every parameter must be trainable, float32, and owned only
            within that module's subtree.
    """

    task_module: nn.Module


def register_task_head_module(
    model: nn.Module,
    result: PreFSDPHookResult,
    *,
    pre_hook_module_ids: set[int],
    pre_hook_parameter_ids: set[int],
) -> str:
    """Validate and record the task module returned by a pre-FSDP hook."""
    module = result.task_module
    if not isinstance(module, nn.Module):
        raise TypeError("PreFSDPHookResult.task_module must be a torch.nn.Module")

    paths_by_id: dict[int, list[str]] = {}
    for name, candidate in model.named_modules(remove_duplicate=False):
        paths_by_id.setdefault(id(candidate), []).append(name)

    paths = [path for path in paths_by_id.get(id(module), ()) if path]
    if not paths:
        raise ValueError("The declared task module must be attached below the model root")
    if len(paths) != 1:
        raise ValueError(
            f"The declared task module must have exactly one model FQN; found aliases: {', '.join(sorted(paths))}"
        )
    name = paths[0]
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError(f"Declared task module {name!r} owns no parameters")
    if any(id(parameter) in pre_hook_parameter_ids for parameter in parameters):
        raise ValueError(f"Declared task module {name!r} reuses pre-hook parameters; the task module must be fresh")
    if id(module) in pre_hook_module_ids:
        raise ValueError(f"Declared task module {name!r} existed before the hook; the task module must be fresh")
    parameter_paths: dict[int, list[str]] = {}
    for parameter_name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_paths.setdefault(id(parameter), []).append(parameter_name)
    task_prefix = f"{name}."
    for parameter in parameters:
        outside_aliases = [
            parameter_name
            for parameter_name in parameter_paths.get(id(parameter), ())
            if not parameter_name.startswith(task_prefix)
        ]
        if outside_aliases:
            raise ValueError(
                f"Declared task module {name!r} shares parameters outside its subtree: "
                f"{', '.join(sorted(outside_aliases))}"
            )
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError(f"Declared task module {name!r} must own only float32 parameters")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError(f"Declared task module {name!r} must own only trainable parameters")

    setattr(model, _TASK_HEAD_MODULE_NAME, name)
    return name


def _task_head_owner_and_name(model: nn.Module) -> tuple[nn.Module, str | None]:
    """Resolve the module that owns task-head metadata and its local FQN."""
    pending = [model]
    seen: set[int] = set()
    while pending:
        owner = pending.pop()
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        name = vars(owner).get(_TASK_HEAD_MODULE_NAME)
        if name is not None:
            return owner, name
        children = getattr(owner, "_modules", {}) or {}
        for child_name in ("module", "_orig_mod"):
            child = children.get(child_name)
            if isinstance(child, nn.Module):
                pending.append(child)
    return model, None


def task_head_module_name(model: nn.Module) -> str | None:
    """Return the task-module FQN recorded on ``model`` or its transparent wrapper."""
    return _task_head_owner_and_name(model)[1]


def task_head_module(model: nn.Module) -> nn.Module | None:
    """Resolve the declared task module on ``model``."""
    owner, name = _task_head_owner_and_name(model)
    return owner.get_submodule(name) if name is not None else None


def is_task_head_parameter(model: nn.Module, name: str) -> bool:
    """Return whether ``name`` belongs to a declared task module."""
    normalized = canonical_parameter_fqn(name)
    while normalized.startswith("_orig_mod."):
        normalized = normalized.removeprefix("_orig_mod.")
    task_name = task_head_module_name(model)
    return task_name is not None and normalized.startswith(f"{task_name}.")


def exclude_task_heads_from_tp_plan(
    model: nn.Module,
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Remove TP rules that resolve exclusively inside the declared task module.

    A wildcard that also resolves outside a task module is rejected because
    removing it would silently disable tensor parallelism for unrelated model
    modules.
    """
    task_root = task_head_module_name(model)
    if task_root is None:
        return dict(plan)

    module_names = [name for name, _ in model.named_modules() if name]

    def _is_task_subtree(name: str) -> bool:
        return name == task_root or name.startswith(task_root + ".")

    filtered: dict[str, object] = {}
    for pattern, style in plan.items():
        matches = [name for name in module_names if fnmatchcase(name, pattern)]
        task_matches = [name for name in matches if _is_task_subtree(name)]
        if not task_matches:
            filtered[pattern] = style
            continue
        non_task_matches = [name for name in matches if not _is_task_subtree(name)]
        if non_task_matches:
            raise ValueError(
                f"Tensor-parallel rule {pattern!r} matches both the managed task module and backbone modules; "
                "use a narrower rule so the task module remains replicated"
            )
    return filtered
