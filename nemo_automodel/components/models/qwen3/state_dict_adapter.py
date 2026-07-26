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

"""Passthrough state-dict adapter for dense Qwen3 checkpoints."""

from __future__ import annotations

import re

import torch
from transformers import Qwen3Config


class Qwen3StateDictAdapter:
    """Convert dense Qwen3 checkpoints whose HuggingFace and NeMo keys match."""

    def __init__(self, config: Qwen3Config) -> None:
        self.config = config

    def from_hf(self, hf_state_dict: dict[str, torch.Tensor], **kwargs: object) -> dict[str, torch.Tensor]:
        """Copy a HuggingFace state dict and materialize a tied LM head if needed.

        Args:
            hf_state_dict: Mapping from parameter names to checkpoint tensors.
                Tensor shapes, dtypes, devices, and storage aliases are preserved.
            **kwargs: Compatibility arguments accepted by the checkpoint adapter
                interface; they do not alter dense Qwen3 tensors.

        Returns:
            A new mapping with the same tensor values and arbitrary parameter
            shapes. For tied models, ``lm_head.weight`` aliases the input
            embedding tensor when the checkpoint omits that key.
        """
        state_dict = dict(hf_state_dict)
        embed_key = "model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        if self.config.tie_word_embeddings and lm_head_key not in state_dict and embed_key in state_dict:
            state_dict[lm_head_key] = state_dict[embed_key]
        return state_dict

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        exclude_key_regex: str | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Copy a NeMo state dict, optionally excluding matching keys.

        Args:
            state_dict: Mapping from parameter names to checkpoint tensors with
                arbitrary parameter shapes; values are returned without mutation.
            exclude_key_regex: Optional regular expression selecting keys to omit.
            **kwargs: Compatibility arguments accepted by the checkpoint adapter
                interface; they do not alter dense Qwen3 tensors.

        Returns:
            A new HuggingFace-format mapping whose tensor shapes, dtypes,
            devices, and storage aliases match ``state_dict``.
        """
        if exclude_key_regex is None:
            return dict(state_dict)
        return {key: value for key, value in state_dict.items() if not re.search(exclude_key_regex, key)}
