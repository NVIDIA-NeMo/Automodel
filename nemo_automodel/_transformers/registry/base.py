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
from collections import OrderedDict
from dataclasses import dataclass, field
from types import ModuleType

import torch.nn as nn

from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

logger = logging.getLogger(__name__)


class _LazyArchMapping:
    """Lazy-loading mapping from architecture name to model class.

    Inspired by HuggingFace transformers' ``_LazyAutoMapping``.  Entries from the
    static ``ModelPackageSpec`` mapping are imported on first access and cached.  Additional entries
    can be added at runtime via ``register``.
    """

    def __init__(self, auto_map: OrderedDict[str, ModelPackageSpec] | dict[str, ModelPackageSpec] | None = None):
        self._specs: dict[str, ModelPackageSpec] = OrderedDict()
        self._tags: dict[str, set] = {}
        for key, spec in (auto_map or {}).items():
            if spec.class_name is None:
                raise ValueError(f"Model architecture entry {key!r} must include a class name")
            spec = spec.with_architecture(key)
            self._specs[key] = spec
            if spec.tags:
                self._tags[key] = set(spec.tags)
        self._loaded: dict[str, type[nn.Module]] = {}
        self._extra: dict[str, type[nn.Module]] = {}
        self._extra_specs: dict[str, ModelPackageSpec] = {}
        self._modules: dict[str, object] = {}

    def _load(self, key: str) -> type[nn.Module]:
        if key in self._loaded:
            return self._loaded[key]
        spec = self._specs[key]
        module_path = spec.module_path
        class_name = spec.class_name
        if module_path not in self._modules:
            self._modules[module_path] = importlib.import_module(module_path)
        cls = getattr(self._modules[module_path], class_name)
        self._loaded[key] = cls
        return cls

    def __contains__(self, key: str) -> bool:
        if key in self._extra or key in self._loaded:
            return True
        if key not in self._specs:
            return False
        try:
            self._load(key)
            return True
        except Exception:
            logger.debug("Model %s unavailable (import failed), removing from registry specs", key)
            self._specs.pop(key, None)
            self._tags.pop(key, None)
            return False

    def __getitem__(self, key: str) -> type[nn.Module]:
        if key in self._extra:
            return self._extra[key]
        if key in self._specs:
            return self._load(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: type[nn.Module]) -> None:
        self._extra[key] = value
        self._extra_specs[key] = ModelPackageSpec.from_module_path(value.__module__, value.__name__).with_architecture(
            key
        )

    def register(self, key: str, value: type[nn.Module], exist_ok: bool = False) -> None:
        """Register a model class under the given architecture name."""
        if not exist_ok and key in self._extra:
            raise ValueError(f"Duplicated model implementation for {key}")
        self._extra[key] = value
        self._extra_specs[key] = ModelPackageSpec.from_module_path(value.__module__, value.__name__).with_architecture(
            key
        )

    def has_tag(self, key: str, tag: str) -> bool:
        """Return ``True`` if *key* was registered with *tag*."""
        return tag in self._tags.get(key, set())

    def keys_with_tag(self, tag: str) -> set:
        """Return all architecture names that have *tag*."""
        return {k for k, tags in self._tags.items() if tag in tags}

    def keys(self):
        return set(self._specs.keys()) | set(self._extra.keys())

    def get_spec(self, key: str) -> ModelPackageSpec | None:
        """Return package metadata for *key* without importing the model class."""
        if key in self._extra_specs:
            return self._extra_specs[key]
        return self._specs.get(key)

    def specs(self) -> tuple[ModelPackageSpec, ...]:
        """Return metadata for all statically and dynamically registered model classes."""
        return (*self._specs.values(), *self._extra_specs.values())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"_LazyArchMapping(specs={len(self._specs)}, extra={len(self._extra)}, loaded={len(self._loaded)})"


@dataclass
class _ModelRegistry:
    model_arch_mapping: OrderedDict[str, ModelPackageSpec] | dict[str, ModelPackageSpec] | None = None
    model_arch_name_to_cls: _LazyArchMapping = field(default=None)
    package_specs: tuple[ModelPackageSpec, ...] = ()
    _retrieval_archs: set = field(default_factory=set)
    _architecture_to_specs: dict[str, tuple[ModelPackageSpec, ...]] = field(default_factory=dict)
    _model_type_to_specs: dict[str, tuple[ModelPackageSpec, ...]] = field(default_factory=dict)

    def __post_init__(self):
        if self.model_arch_name_to_cls is None:
            self.model_arch_name_to_cls = _LazyArchMapping(self.model_arch_mapping or {})
        self._retrieval_archs = self.model_arch_name_to_cls.keys_with_tag("retrieval")
        self._rebuild_spec_indexes()

    def _iter_specs(self) -> tuple[ModelPackageSpec, ...]:
        return (*self.model_arch_name_to_cls.specs(), *self.package_specs)

    def _rebuild_spec_indexes(self) -> None:
        architecture_to_specs: dict[str, list[ModelPackageSpec]] = {}
        model_type_to_specs: dict[str, list[ModelPackageSpec]] = {}
        for spec in self._iter_specs():
            for architecture in spec.architectures:
                architecture_to_specs.setdefault(architecture, []).append(spec)
            for model_type in spec.model_types:
                model_type_to_specs.setdefault(model_type, []).append(spec)
        self._architecture_to_specs = {
            architecture: tuple(specs) for architecture, specs in architecture_to_specs.items()
        }
        self._model_type_to_specs = {
            model_type: tuple(self._dedupe_specs(specs)) for model_type, specs in model_type_to_specs.items()
        }

    @staticmethod
    def _dedupe_specs(specs: list[ModelPackageSpec] | tuple[ModelPackageSpec, ...]) -> tuple[ModelPackageSpec, ...]:
        seen: set[str] = set()
        deduped: list[ModelPackageSpec] = []
        for spec in specs:
            if spec.package in seen:
                continue
            seen.add(spec.package)
            deduped.append(spec)
        return tuple(deduped)

    @staticmethod
    def _import_optional_module(spec: ModelPackageSpec, module_name: str) -> ModuleType | None:
        if module_name not in spec.optional_modules:
            return None
        module_path = spec.optional_module_path(module_name)
        try:
            return importlib.import_module(module_path)
        except ImportError:
            logger.debug("Optional model module is unavailable: %s", module_path)
            return None

    @property
    def supported_models(self):
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def has_custom_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* has a custom (non-HF) implementation."""
        return arch_name in self.model_arch_name_to_cls

    def has_retrieval_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* is a registered retrieval/encoder architecture."""
        return arch_name in self._retrieval_archs

    def register_retrieval(self, arch_name: str) -> None:
        """Mark *arch_name* as a retrieval/encoder architecture."""
        self._retrieval_archs.add(arch_name)

    def get_model_package_spec(self, architecture: str) -> ModelPackageSpec | None:
        """Return package metadata for an architecture without importing ``model.py``."""
        specs = self.get_model_package_specs_for_architecture(architecture)
        if specs:
            return specs[0]
        return self.model_arch_name_to_cls.get_spec(architecture)

    def get_model_package_specs_for_architecture(self, architecture: str) -> tuple[ModelPackageSpec, ...]:
        """Return all package metadata entries that declare *architecture*."""
        return self._architecture_to_specs.get(architecture, ())

    def get_model_package_specs_for_model_type(self, model_type: str) -> tuple[ModelPackageSpec, ...]:
        """Return package metadata entries that declare *model_type*."""
        return self._model_type_to_specs.get(model_type, ())

    def get_optional_module_for_architecture(self, architecture: str, module_name: str) -> ModuleType | None:
        """Import ``<model package>.<module_name>`` for an architecture if declared."""
        modules = self.iter_optional_modules_for_architectures((architecture,), module_name)
        return modules[0] if modules else None

    def iter_optional_modules_for_architectures(
        self, architectures: tuple[str, ...] | list[str], module_name: str
    ) -> tuple[ModuleType, ...]:
        """Import convention modules for the requested architectures only."""
        modules: list[ModuleType] = []
        seen_paths: set[str] = set()
        for architecture in architectures:
            for spec in self.get_model_package_specs_for_architecture(architecture):
                if module_name not in spec.optional_modules:
                    continue
                module_path = spec.optional_module_path(module_name)
                if module_path in seen_paths:
                    continue
                module = self._import_optional_module(spec, module_name)
                if module is not None:
                    seen_paths.add(module_path)
                    modules.append(module)
        return tuple(modules)

    def iter_optional_modules_for_model_type(self, model_type: str, module_name: str) -> tuple[ModuleType, ...]:
        """Import convention modules for packages that declare *model_type*."""
        modules: list[ModuleType] = []
        for spec in self.get_model_package_specs_for_model_type(model_type):
            module = self._import_optional_module(spec, module_name)
            if module is not None:
                modules.append(module)
        return tuple(modules)

    def iter_optional_modules(
        self,
        module_name: str,
        *,
        global_patches: bool | None = None,
        pre_config_patches: bool | None = None,
        post_shard_patches: bool | None = None,
        tokenizer_registrations: bool | None = None,
    ) -> tuple[ModuleType, ...]:
        """Import declared convention modules matching the provided metadata filters."""
        modules: list[ModuleType] = []
        seen_paths: set[str] = set()
        for spec in self._iter_specs():
            if module_name not in spec.optional_modules:
                continue
            if global_patches is not None and spec.global_patches is not global_patches:
                continue
            if pre_config_patches is not None and spec.pre_config_patches is not pre_config_patches:
                continue
            if post_shard_patches is not None and spec.post_shard_patches is not post_shard_patches:
                continue
            if tokenizer_registrations is not None and spec.tokenizer_registrations is not tokenizer_registrations:
                continue
            module_path = spec.optional_module_path(module_name)
            if module_path in seen_paths:
                continue
            module = self._import_optional_module(spec, module_name)
            if module is not None:
                seen_paths.add(module_path)
                modules.append(module)
        return tuple(modules)

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
        if architecture not in self.model_arch_name_to_cls:
            return None
        model_cls = self.model_arch_name_to_cls[architecture]
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
        self.model_arch_name_to_cls.register(arch_name, model_cls, exist_ok=exist_ok)
        self._rebuild_spec_indexes()
