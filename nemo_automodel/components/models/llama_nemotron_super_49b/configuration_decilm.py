# coding=utf-8
# Copyright 2024 Nvidia Corporation. All rights reserved.
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

import dataclasses
import warnings
from typing import Dict, Any

from transformers.utils import is_flash_attn_2_available

from .block_config import BlockConfig
from .transformers_4_44_2__configuration_llama import LlamaConfig
from .transformers_4_44_2__modeling_rope_utils import \
    rope_config_validation  # fake import to make AutoConfig infer the dependency

rope_config_validation  # this line is here to make sure that auto-formatting doesn't remove the import


class DeciLMConfig(LlamaConfig):
    model_type = "nemotron-nas"

    def __init__(
            self,
            block_configs: list[dict] | list[BlockConfig] = None,
            **kwargs,
    ):
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is None and is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"

        if block_configs is not None:
            if isinstance(block_configs[0], dict):
                block_configs = [BlockConfig(**conf) for conf in block_configs]

            using_unshifted_sink = any([block_config.attention.unshifted_sink for block_config in block_configs])
            if using_unshifted_sink and attn_implementation != "eager":
                warnings.warn("Forcing attn_implementation='eager' since some attention layers use unshifted sink")
                attn_implementation = "eager"

        super().__init__(attn_implementation=attn_implementation, **kwargs)

        self.intermediate_size = None
        self.num_key_value_heads = None

        if block_configs is not None:
            assert len(block_configs) == self.num_hidden_layers

        self.block_configs: list[BlockConfig] = block_configs

    def to_dict(self) -> Dict[str, Any]:
        self_dict = super().to_dict()
        if self.block_configs is not None:
            self_dict["block_configs"] = [dataclasses.asdict(conf) for conf in self.block_configs]
        return self_dict
