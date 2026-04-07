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

"""State dict adapter for Qwen3 dense model.

Converts between the HF checkpoint format (separate q_proj/k_proj/v_proj and
gate_proj/up_proj) and our native format (combined qkv_proj and gate_up_proj)
which is required for correct TP ColwiseParallel sharding.

The per-head q_norm and k_norm weights have the same key names in both formats
and pass through unchanged -- no special handling is required.
"""

from transformers import Qwen3Config

from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)


class Qwen3StateDictAdapter(CombinedProjectionStateDictAdapter):
    """State dict adapter for Qwen3 dense models.

    Handles the HF ↔ native format conversion:
    - HF: separate ``q_proj``, ``k_proj``, ``v_proj`` weights
    - Native: combined ``qkv_proj`` weight in KV-head-grouped interleaved layout
    - HF: separate ``gate_proj``, ``up_proj`` weights
    - Native: combined ``gate_up_proj`` weight in row-interleaved layout

    Keys not matched by the above patterns (``q_norm``, ``k_norm``, ``o_proj``,
    ``down_proj``, layer norms, embeddings, lm_head) pass through unchanged.

    Example::

        from transformers import Qwen3Config
        config = Qwen3Config.from_pretrained("Qwen/Qwen3-32B")
        adapter = Qwen3StateDictAdapter(config)

        custom_sd = adapter.from_hf(hf_state_dict)
        hf_sd = adapter.to_hf(custom_sd)
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
