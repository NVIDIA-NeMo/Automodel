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

"""Compatibility wrappers for legacy v4-style remote-code model patches."""


def _get_nemotron_flash_patches():
    """Return the Nemotron Flash patch module via the model registry."""
    from nemo_automodel._transformers.registry import ModelRegistry

    module = ModelRegistry.get_optional_module_for_architecture("NemotronFlashForCausalLM", "patches")
    if module is None:
        raise ModuleNotFoundError("Nemotron Flash patch module is not registered")
    return module


def _is_nemotron_flash_config(cfg: object) -> bool:
    """Return True when *cfg* identifies a Nemotron Flash remote-code model."""
    return _get_nemotron_flash_patches()._is_nemotron_flash_config(cfg)


def should_fix_rotary_embeddings(model_parts: list[object]) -> bool:
    """Return True when the legacy rotary workaround should run."""
    return _get_nemotron_flash_patches().should_fix_rotary_embeddings(model_parts)


def fix_rotary_embeddings(model_parts: list[object]) -> int:
    """Install Nemotron-Flash-1B's native NTK ``inv_freq`` deterministically."""
    return _get_nemotron_flash_patches().fix_rotary_embeddings(model_parts)


__all__ = ["_is_nemotron_flash_config", "fix_rotary_embeddings", "should_fix_rotary_embeddings"]
