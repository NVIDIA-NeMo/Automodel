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

"""Config-shape fingerprint that distinguishes Hy-MT2-30B-A3B from Hy3-preview.

Tencent ships both checkpoints with ``architectures: ["HYV3ForCausalLM"]`` and
``model_type: "hy_v3"`` even though the two models differ substantially
(48 vs 80 layers, 128 vs 192 experts, hidden 2048 vs 4096, etc.). The
auto-resolver in ``_transformers/model_init.py`` looks up the fingerprint here
so all Hy-MT2-specific knowledge stays inside this module.
"""

from typing import Any


def is_hy_mt2_config(config: Any) -> bool:
    """Return whether *config* describes Tencent's Hy-MT2-30B-A3B checkpoint."""
    return (
        getattr(config, "model_type", None) == "hy_v3"
        and getattr(config, "hidden_size", None) == 2048
        and getattr(config, "num_hidden_layers", None) == 48
        and getattr(config, "num_experts", None) == 128
        and getattr(config, "expert_hidden_dim", None) == 768
        and getattr(config, "moe_intermediate_size", None) == 768
        and hasattr(config, "enable_lm_head_fp32")
    )
