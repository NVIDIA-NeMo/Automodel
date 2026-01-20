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

"""DeepSeek V3.2 Model.

Key differences from DeepSeek V3:
- Indexer mechanism for top-k sparse attention selection
- Updated MLA parameters (q_lora_rank, kv_lora_rank, head dims)
- Fewer routed experts (64 vs 256)
"""

from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import DeepseekV32Indexer, DeepseekV32MLA
from nemo_automodel.components.models.deepseek_v32.model import (
    DeepseekV32Block,
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
    ModelClass,
)
from nemo_automodel.components.models.deepseek_v32.state_dict_adapter import DeepSeekV32StateDictAdapter

__all__ = [
    "DeepseekV32Config",
    "DeepseekV32Indexer",
    "DeepseekV32MLA",
    "DeepseekV32Block",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
    "DeepSeekV32StateDictAdapter",
    "ModelClass",
]
