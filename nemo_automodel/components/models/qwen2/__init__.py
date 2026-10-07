# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Custom Qwen2 model implementation for NeMo Automodel."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["Qwen2ForCausalLM"]


if TYPE_CHECKING:
    from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM


def __getattr__(name: str) -> Any:
    """Load public model exports only when requested, keeping sidecars lightweight."""
    if name in ("Qwen2ForCausalLM",):
        return getattr(import_module("nemo_automodel.components.models.qwen2.model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
