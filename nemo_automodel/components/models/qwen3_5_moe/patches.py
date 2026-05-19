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

"""Runtime patch declarations for Qwen3.5 linear-attention models."""

_RUNTIME_PATCH_SPECS: dict[str, tuple[str, str]] = {
    "Qwen3_5ForCausalLM": (
        "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn",
        "apply_model_runtime_patches",
    ),
    "Qwen3_5ForConditionalGeneration": (
        "nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn",
        "apply_model_runtime_patches",
    ),
}


def get_runtime_patch_specs() -> dict[str, tuple[str, str]]:
    """Return architecture-name to runtime patch hook import specs."""
    return dict(_RUNTIME_PATCH_SPECS)
