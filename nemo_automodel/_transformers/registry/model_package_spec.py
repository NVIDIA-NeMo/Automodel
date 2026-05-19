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

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelPackageSpec:
    """Registry metadata for a model package and its optional convention modules."""

    package: str
    class_name: str | None = None
    model_module: str = "model"
    config_module: str | None = None
    config_class_name: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    architectures: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    optional_modules: frozenset[str] = field(default_factory=frozenset)
    global_patches: bool = False
    pre_config_patches: bool = False
    post_shard_patches: bool = False
    tokenizer_registrations: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "model_types", tuple(self.model_types))
        object.__setattr__(self, "optional_modules", frozenset(self.optional_modules))

    @classmethod
    def from_module_path(
        cls,
        module_path: str,
        class_name: str,
        *,
        config_module: str | None = None,
        config_class_name: str | None = None,
        tags: set[str] | frozenset[str] | tuple[str, ...] = (),
        architectures: tuple[str, ...] = (),
        model_types: tuple[str, ...] = (),
        optional_modules: set[str] | frozenset[str] | tuple[str, ...] = (),
        global_patches: bool = False,
        pre_config_patches: bool = False,
        post_shard_patches: bool = False,
        tokenizer_registrations: bool = False,
    ) -> "ModelPackageSpec":
        """Create a spec from a fully qualified model module path."""
        package, sep, model_module = module_path.rpartition(".")
        if not sep:
            package = ""
            model_module = module_path
        return cls(
            package=package,
            class_name=class_name,
            model_module=model_module,
            config_module=config_module,
            config_class_name=config_class_name,
            tags=frozenset(tags),
            architectures=architectures,
            model_types=model_types,
            optional_modules=frozenset(optional_modules),
            global_patches=global_patches,
            pre_config_patches=pre_config_patches,
            post_shard_patches=post_shard_patches,
            tokenizer_registrations=tokenizer_registrations,
        )

    @property
    def module_path(self) -> str:
        """Return the import path for the model implementation module."""
        if not self.package:
            return self.model_module
        return f"{self.package}.{self.model_module}"

    def optional_module_path(self, module_name: str) -> str:
        """Return the import path for a convention module such as ``patches``."""
        if not self.package:
            return f"{self.model_module}.{module_name}"
        return f"{self.package}.{module_name}"

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
