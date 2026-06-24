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

"""Shared helpers for FSDP treatment of multimodal submodules."""

from typing import Literal

import torch.nn as nn

FrozenMultimodalSharding = Literal["shard", "replicate"]

MULTIMODAL_FSDP_MODULE_NAMES = (
    "audio_tower",
    "visual",
    "vision_tower",
    "vision_model",
    "image_encoder",
    "vision_encoder",
    "embed_vision",
    "audio_encoder",
    "audio_model",
    "mm_projector",
    "multi_modal_projector",
    "multimodal_projector",
    "vision_projector",
    "vit_large_projector",
    "audio_projector",
)

VALID_FROZEN_MULTIMODAL_SHARDING: tuple[FrozenMultimodalSharding, ...] = ("shard", "replicate")


def normalize_frozen_multimodal_sharding(value: str) -> FrozenMultimodalSharding:
    """Validate and normalize the frozen multimodal FSDP policy."""
    normalized = value.lower().replace("-", "_")
    if normalized not in VALID_FROZEN_MULTIMODAL_SHARDING:
        valid = ", ".join(VALID_FROZEN_MULTIMODAL_SHARDING)
        raise ValueError(f"frozen_multimodal_sharding must be one of: {valid}. Got {value!r}.")
    return normalized  # type: ignore[return-value]


def is_multimodal_module_name(name: str) -> bool:
    """Return True when ``name`` identifies a known multimodal tower/projector."""
    return name in MULTIMODAL_FSDP_MODULE_NAMES


def module_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return direct and recursive parameters for module-like test doubles too."""
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return []
    return list(parameters())


def _is_module_container(module: nn.Module) -> bool:
    container_types = tuple(cls for cls in (getattr(nn, "ModuleList", None), getattr(nn, "ModuleDict", None)) if cls)
    return bool(container_types) and isinstance(module, container_types)


def _container_items(module: nn.Module):
    module_dict_cls = getattr(nn, "ModuleDict", None)
    if module_dict_cls is not None and isinstance(module, module_dict_cls):
        return list(module.items())
    return list(enumerate(module))


def _named_children(module: nn.Module):
    named_children = getattr(module, "named_children", None)
    if not callable(named_children):
        return []
    return list(named_children())


def _shard_layer_containers_recursively(module: nn.Module, shard_module) -> bool:
    sharded_child = False
    for _, child in _named_children(module):
        if _is_module_container(child):
            for _, item in _container_items(child):
                if _is_module_container(item):
                    sharded_child |= _shard_layer_containers_recursively(item, shard_module)
                else:
                    shard_module(item)
                    sharded_child = True
        else:
            sharded_child |= _shard_layer_containers_recursively(child, shard_module)
    return sharded_child


def shard_trainable_multimodal_module(module: nn.Module, shard_module) -> None:
    """Shard a trainable multimodal module at layer-container granularity when possible."""
    if not _shard_layer_containers_recursively(module, shard_module):
        shard_module(module)


def iter_multimodal_modules(model: nn.Module):
    """Yield maximal multimodal submodules by qualified name."""
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        selected_names: list[str] = []
        for name, module in named_modules():
            if not name:
                continue
            if any(name == selected or name.startswith(selected + ".") for selected in selected_names):
                continue
            leaf_name = name.rsplit(".", 1)[-1]
            if is_multimodal_module_name(leaf_name):
                selected_names.append(name)
                yield name, module
        return

    seen_ids: set[int] = set()
    owners = [("", model)]
    inner_model = getattr(model, "model", None)
    if inner_model is not None and inner_model is not model:
        owners.append(("model", inner_model))

    for owner_name, owner in owners:
        for attr_name in MULTIMODAL_FSDP_MODULE_NAMES:
            module = getattr(owner, attr_name, None)
            if module is None or id(module) in seen_ids:
                continue
            seen_ids.add(id(module))
            module_name = f"{owner_name}.{attr_name}" if owner_name else attr_name
            yield module_name, module


def ignored_params_for_root(root: nn.Module, ignored_params: set[nn.Parameter]) -> set[nn.Parameter] | None:
    """Return the subset of ignored params owned by ``root`` for an FSDP root wrap."""
    if not ignored_params:
        return None
    root_params = set(module_parameters(root))
    ignored_in_root = root_params & ignored_params
    return ignored_in_root or None
