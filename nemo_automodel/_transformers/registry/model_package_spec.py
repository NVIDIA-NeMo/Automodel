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

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelPackageSpec:
    """Registry entry for a model package.

    Args:
        package: Python package that owns the model implementation, usually
            ``nemo_automodel.components.models.<model_name>``.
        class_name: Model class to lazy-load from ``module_path`` for architectures
            declared by this spec. Config-only specs leave this unset.
        model_module: Module inside ``package`` that contains ``class_name``. Defaults
            to ``model``, producing ``<package>.model``.
        config_module: Module inside ``package`` that contains a custom HF config
            class. Used only for model types that need on-demand AutoConfig registration.
        config_class_name: Explicit config class name to register from
            ``config_module_path``. If unset, the registry can discover a local
            ``PretrainedConfig`` subclass by ``model_type``.
        architectures: HF ``config.architectures`` names that should resolve to this
            model class. These are expanded into the architecture-to-spec lookup.
        model_types: HF ``config.model_type`` names associated with this package,
            primarily for resolving and registering custom config classes.
    """

    package: str
    class_name: str | None = None
    model_module: str = "model"
    config_module: str | None = None
    config_class_name: str | None = None
    architectures: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "model_types", tuple(self.model_types))

    @classmethod
    def from_model_class(
        cls,
        model_cls: type,
        *,
        config_module: str | None = None,
        config_class_name: str | None = None,
        architectures: list[str] | tuple[str, ...] = (),
        model_types: tuple[str, ...] = (),
    ) -> "ModelPackageSpec":
        """Create a spec from a model class object."""
        package, sep, model_module = model_cls.__module__.rpartition(".")
        if not sep:
            package = ""
            model_module = model_cls.__module__
        return cls(
            package=package,
            class_name=model_cls.__name__,
            model_module=model_module,
            config_module=config_module,
            config_class_name=config_class_name,
            architectures=architectures,
            model_types=model_types,
        )

    @property
    def module_path(self) -> str:
        """Return the import path for the model implementation module."""
        if not self.package:
            return self.model_module
        return f"{self.package}.{self.model_module}"

    @property
    def config_module_path(self) -> str | None:
        """Return the import path for this model package's config module, if declared."""
        if self.config_module is None:
            return None
        if not self.package:
            return self.config_module
        return f"{self.package}.{self.config_module}"

    def with_architecture(self, architecture: str) -> "ModelPackageSpec":
        """Return a copy that records *architecture* as an alias for this package."""
        if architecture in self.architectures:
            return self
        return replace(self, architectures=(*self.architectures, architecture))
