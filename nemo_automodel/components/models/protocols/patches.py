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

"""Structural protocols for model package ``patches.py`` convention modules."""

from typing import Any, Protocol, runtime_checkable

RuntimePatchTarget = tuple[str, str]
RuntimePatchSpecs = dict[str, RuntimePatchTarget]


@runtime_checkable
class GlobalPatchModule(Protocol):
    """Protocol for process-wide compatibility patches."""

    def apply_global_patches(self) -> None:
        """Apply idempotent process-wide compatibility patches."""


@runtime_checkable
class PreConfigPatchModule(Protocol):
    """Protocol for patches that must run before HuggingFace config construction."""

    def apply_pre_config_patches(self) -> None:
        """Apply idempotent config-construction compatibility patches."""


@runtime_checkable
class RuntimePatchSpecModule(Protocol):
    """Protocol for architecture-specific runtime patch declarations."""

    def get_runtime_patch_specs(self) -> RuntimePatchSpecs:
        """Return architecture names mapped to ``(module_path, function_name)`` patch hooks."""


@runtime_checkable
class PostShardPatchModule(Protocol):
    """Protocol for patches that run after sharding and checkpoint loading."""

    def apply_post_shard_patches(self, model_parts: list[object]) -> object:
        """Apply post-shard compatibility patches to model parts."""


@runtime_checkable
class PostShardPatchPredicate(Protocol):
    """Optional predicate for post-shard patches."""

    def should_apply_post_shard_patches(self, model_parts: list[object]) -> bool:
        """Return whether post-shard patches should run for model parts."""


@runtime_checkable
class LayerTypesConfigPatchModule(Protocol):
    """Protocol for the Step-3.5 layer-types config loading compatibility hook."""

    def load_config_with_layer_types_fix(
        self,
        pretrained_model_name_or_path: str,
        attn_implementation: str | None,
        trust_remote_code: bool,
        **kwargs: Any,
    ) -> object:
        """Load a config after applying layer-types compatibility fixes."""
