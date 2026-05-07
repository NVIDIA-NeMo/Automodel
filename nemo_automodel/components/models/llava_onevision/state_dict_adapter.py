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
"""State dict adapter for LLaVA-OneVision-1.5.

HF on-disk safetensors layout (from lmms-lab/LLaVA-OneVision-1.5-*):
  visual.{patch_embed,class_embedding,class_pos_emb,pre_layernorm,blocks.*,merger.*}
  model.{embed_tokens,layers.*,norm}
  lm_head.weight

NeMo in-memory module layout:
  model.visual.*
  model.language_model.*   (transformers.Qwen3Model; keys match the Qwen3 prefix)
  lm_head.*

Applies the same regex rename HF does via ``_checkpoint_conversion_mapping``:
  ^visual                         -> model.visual
  ^model(?!\\.(language_model|visual))  -> model.language_model
"""

import re
from typing import Any, Optional

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

_HF_TO_NEMO_RULES = [
    (re.compile(r"^visual\."), "model.visual."),
    (re.compile(r"^model\.(?!language_model\.|visual\.)"), "model.language_model."),
]

_NEMO_TO_HF_RULES = [
    (re.compile(r"^model\.visual\."), "visual."),
    (re.compile(r"^model\.language_model\."), "model."),
]


class LlavaOneVisionStateDictAdapter(StateDictAdapter):
    def __init__(self, config: Any = None, **kwargs):
        self.config = config

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        return {_rename(k, _HF_TO_NEMO_RULES): v for k, v in hf_state_dict.items()}

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        exclude = re.compile(exclude_key_regex) if exclude_key_regex else None
        out: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            if exclude and exclude.match(fqn):
                continue
            out[_rename(fqn, _NEMO_TO_HF_RULES)] = tensor
        return out

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        return [(_rename(fqn, _NEMO_TO_HF_RULES), tensor)]


def _rename(key: str, rules) -> str:
    for pattern, replacement in rules:
        new_key, n = pattern.subn(replacement, key, count=1)
        if n:
            return new_key
    return key
