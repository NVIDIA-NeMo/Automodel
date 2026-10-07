# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3VLConfig",
    "MiniMaxM3VLTextConfig",
    "MiniMaxM3VLVisionConfig",
    "ModelClass",
]


if TYPE_CHECKING:
    from nemo_automodel.components.models.minimax_m3_vl.config import (
        MiniMaxM3VLConfig,
        MiniMaxM3VLTextConfig,
        MiniMaxM3VLVisionConfig,
    )
    from nemo_automodel.components.models.minimax_m3_vl.model import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )


def __getattr__(name: str) -> Any:
    """Load public model exports only when requested, keeping sidecars lightweight."""
    if name in ("MiniMaxM3VLConfig", "MiniMaxM3VLTextConfig", "MiniMaxM3VLVisionConfig"):
        return getattr(import_module("nemo_automodel.components.models.minimax_m3_vl.config"), name)
    if name in ("MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration", "ModelClass"):
        return getattr(
            import_module("nemo_automodel.components.models.minimax_m3_vl.model"),
            {"ModelClass": "MiniMaxM3SparseForConditionalGeneration"}.get(name, name),
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
