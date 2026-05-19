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


import importlib
import inspect
import logging
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import ModuleType

import torch.nn as nn

from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

logger = logging.getLogger(__name__)

_ModelArchMappingInput = Mapping[str, ModelPackageSpec] | Iterable[ModelPackageSpec] | None


def _normalize_model_arch_mapping(auto_map: _ModelArchMappingInput = None) -> OrderedDict[str, ModelPackageSpec]:
    """Normalize legacy dict mappings or tuple-based specs into an architecture lookup."""
    mapping: OrderedDict[str, ModelPackageSpec] = OrderedDict()
    if isinstance(auto_map, Mapping):
        spec_items = ((architecture, spec.with_architecture(architecture)) for architecture, spec in auto_map.items())
    else:
        spec_items = []
        for spec in auto_map or ():
            if spec.class_name is None and not spec.architectures:
                continue
            if not spec.architectures:
                raise ValueError(f"Model architecture spec for {spec.package!r} must declare at least one architecture")
            spec_items.extend((architecture, spec) for architecture in spec.architectures)

    for architecture, spec in spec_items:
        if spec.class_name is None:
            raise ValueError(f"Model architecture entry {architecture!r} must include a class name")
        if architecture in mapping:
            raise ValueError(f"Duplicated model architecture entry for {architecture!r}")
        mapping[architecture] = spec
    return mapping


@dataclass
class _BaseModelRegistry:
    model_specs: _ModelArchMappingInput = None
    model_arch_name_to_cls: "_BaseModelRegistry" = field(init=False, repr=False, compare=False)
    _model_specs: tuple[ModelPackageSpec, ...] = field(init=False)
    _loaded_model_classes: dict[str, type[nn.Module]] = field(default_factory=dict)
    _extra_model_classes: dict[str, type[nn.Module]] = field(default_factory=dict)
    _discarded_architectures: set[str] = field(default_factory=set)
    _manual_architecture_tags: dict[str, set[str]] = field(default_factory=dict)
    _architecture_to_specs: dict[str, tuple[ModelPackageSpec, ...]] = field(default_factory=dict)
    _model_type_to_specs: dict[str, tuple[ModelPackageSpec, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.model_specs, Mapping):
            raw_specs: Iterable[ModelPackageSpec] = _normalize_model_arch_mapping(self.model_specs).values()
        else:
            raw_specs = tuple(self.model_specs or ())
            _normalize_model_arch_mapping(raw_specs)
        self._model_specs = tuple(dict.fromkeys(raw_specs))
        self._rebuild_spec_indexes()
        self.model_arch_name_to_cls = self

    def _rebuild_spec_indexes(self) -> None:
        architecture_to_specs: dict[str, list[ModelPackageSpec]] = {}
        model_type_to_specs: dict[str, list[ModelPackageSpec]] = {}
        dynamic_specs = tuple(
            ModelPackageSpec.from_model_class(model_cls, architectures=(architecture,))
            for architecture, model_cls in self._extra_model_classes.items()
        )
        for spec in (*dynamic_specs, *self._model_specs):
            for architecture in spec.architectures:
                if architecture in self._discarded_architectures and architecture not in self._extra_model_classes:
                    continue
                architecture_to_specs.setdefault(architecture, []).append(spec)
            for model_type in spec.model_types:
                model_type_to_specs.setdefault(model_type, []).append(spec)
        self._architecture_to_specs = {
            architecture: tuple(specs) for architecture, specs in architecture_to_specs.items()
        }
        self._model_type_to_specs = {model_type: tuple(specs) for model_type, specs in model_type_to_specs.items()}

    def _discard_architecture(self, architecture: str) -> None:
        self._discarded_architectures.add(architecture)
        self._architecture_to_specs.pop(architecture, None)
        self._extra_model_classes.pop(architecture, None)
        self._loaded_model_classes.pop(architecture, None)
        self._manual_architecture_tags.pop(architecture, None)

    def _load_model_class(self, architecture: str) -> type[nn.Module]:
        if architecture in self._loaded_model_classes:
            return self._loaded_model_classes[architecture]
        spec = self.get_model_package_spec(architecture)
        if spec is None or spec.class_name is None:
            raise KeyError(architecture)
        model_cls = getattr(importlib.import_module(spec.module_path), spec.class_name)
        self._loaded_model_classes[architecture] = model_cls
        return model_cls

    def __contains__(self, architecture: str) -> bool:
        if architecture in self._extra_model_classes or architecture in self._loaded_model_classes:
            return True
        if self.get_model_package_spec(architecture) is None:
            return False
        try:
            self._load_model_class(architecture)
            return True
        except Exception:
            logger.debug("Model %s unavailable (import failed), removing from registry specs", architecture)
            self._discard_architecture(architecture)
            return False

    def __getitem__(self, architecture: str) -> type[nn.Module]:
        if architecture in self._extra_model_classes:
            return self._extra_model_classes[architecture]
        return self._load_model_class(architecture)

    def __setitem__(self, architecture: str, model_cls: type[nn.Module]) -> None:
        self._discarded_architectures.discard(architecture)
        self._extra_model_classes[architecture] = model_cls
        self._loaded_model_classes.pop(architecture, None)
        self._rebuild_spec_indexes()

    def keys(self):
        return set(self._architecture_to_specs) | set(self._extra_model_classes)

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return (
            f"_BaseModelRegistry(specs={len(self._architecture_to_specs)}, "
            f"extra={len(self._extra_model_classes)}, loaded={len(self._loaded_model_classes)})"
        )

    def _architecture_has_tag(self, architecture: str, tag: str) -> bool:
        if tag in self._manual_architecture_tags.get(architecture, set()):
            return True
        return any(tag in spec.tags for spec in self.get_model_package_specs_for_architecture(architecture))

    @property
    def supported_models(self):
        return self.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> type[nn.Module]:
        return self[model_arch]

    def has_custom_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* has a custom (non-HF) implementation."""
        return arch_name in self

    def has_retrieval_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* is a registered retrieval/encoder architecture."""
        warnings.warn(
            "has_retrieval_model() is deprecated; use RetrievalModelRegistry.get_model_package_spec() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._architecture_has_tag(arch_name, "retrieval"):
            return True
        from nemo_automodel._transformers.registry import RetrievalModelRegistry

        return RetrievalModelRegistry.get_model_package_spec(arch_name) is not None

    def register_retrieval(self, arch_name: str) -> None:
        """Mark *arch_name* as a retrieval/encoder architecture."""
        warnings.warn(
            "register_retrieval() is deprecated; register retrieval classes with RetrievalModelRegistry.register().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._manual_architecture_tags.setdefault(arch_name, set()).add("retrieval")

    def get_model_package_spec(self, architecture: str) -> ModelPackageSpec | None:
        """Return package metadata for an architecture without importing ``model.py``."""
        specs = self.get_model_package_specs_for_architecture(architecture)
        if specs:
            return specs[0]
        return None

    def get_model_package_specs_for_architecture(self, architecture: str) -> tuple[ModelPackageSpec, ...]:
        """Return all package metadata entries that declare *architecture*."""
        return self._architecture_to_specs.get(architecture, ())

    def get_model_package_specs_for_model_type(self, model_type: str) -> tuple[ModelPackageSpec, ...]:
        """Return package metadata entries that declare *model_type*."""
        return self._model_type_to_specs.get(model_type, ())

    @staticmethod
    def _iter_config_classes(module: ModuleType):
        """Yield local ``PretrainedConfig`` subclasses declared by *module*."""
        from transformers import PretrainedConfig

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is PretrainedConfig:
                continue
            if cls.__module__ != module.__name__:
                continue
            if not issubclass(cls, PretrainedConfig):
                continue
            if not isinstance(getattr(cls, "model_type", None), str):
                continue
            yield cls

    def _resolve_config_class(self, spec: ModelPackageSpec, model_type: str):
        module_path = spec.config_module_path
        if module_path is None:
            return None

        try:
            module = importlib.import_module(module_path)
        except ImportError:
            logger.debug("Config module is unavailable: %s", module_path)
            return None

        if spec.config_class_name is not None:
            return getattr(module, spec.config_class_name, None)

        for config_cls in self._iter_config_classes(module):
            if config_cls.model_type == model_type:
                return config_cls
        return None

    def ensure_config_registered(self, model_type: str) -> bool:
        """Register the local config class for *model_type* with HuggingFace, if declared."""
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if model_type in CONFIG_MAPPING:
            return True

        for spec in self.get_model_package_specs_for_model_type(model_type):
            config_cls = self._resolve_config_class(spec, model_type)
            if config_cls is None:
                continue
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError:
                if getattr(config_cls, "model_type", None) != model_type:
                    CONFIG_MAPPING.register(model_type, config_cls)
                else:
                    raise
            return True
        return False

    def resolve_custom_model_cls(self, architecture: str, config) -> type[nn.Module] | None:
        """Return the custom model class if it exists and supports *config*, else ``None``.

        Custom model classes may define a ``supports_config(config)`` classmethod
        to opt out for specific HF configs (e.g. a Mistral3 VLM with a dense
        Ministral3 text backbone instead of the expected Mistral4 MoE+MLA).
        """
        if architecture not in self:
            return None
        model_cls = self[architecture]
        if hasattr(model_cls, "supports_config") and not model_cls.supports_config(config):
            logger.info(
                "Custom model %s does not support config %s, falling back to HF",
                model_cls.__name__,
                type(config).__name__,
            )
            return None
        return model_cls

    def register(self, arch_name: str, model_cls: type[nn.Module], exist_ok: bool = False) -> None:
        """Register a custom model class for a given architecture name."""
        if not exist_ok and arch_name in self._extra_model_classes:
            raise ValueError(f"Duplicated model implementation for {arch_name}")
        self[arch_name] = model_cls
