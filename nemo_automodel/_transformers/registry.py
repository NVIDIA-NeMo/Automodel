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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Set, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

# Maps model architecture class name -> module that contains the class.
# When the model is first requested, the module is imported and the class
# is resolved lazily (similar to transformers' MODEL_*_MAPPING_NAMES).
_MODEL_ARCH_TO_MODULE: Dict[str, str] = {
    "DeepseekV3ForCausalLM": "nemo_automodel.components.models.deepseek_v3.model",
    "DeepseekV32ForCausalLM": "nemo_automodel.components.models.deepseek_v32.model",
    "Glm4MoeForCausalLM": "nemo_automodel.components.models.glm4_moe.model",
    "Glm4MoeLiteForCausalLM": "nemo_automodel.components.models.glm4_moe_lite.model",
    "GptOssForCausalLM": "nemo_automodel.components.models.gpt_oss.model",
    "KimiK25VLForConditionalGeneration": "nemo_automodel.components.models.kimi_k25_vl.model",
    "KimiVLForConditionalGeneration": "nemo_automodel.components.models.kimivl.model",
    "LlamaForCausalLM": "nemo_automodel.components.models.llama.model",
    "MiniMaxM2ForCausalLM": "nemo_automodel.components.models.minimax_m2.model",
    "Ministral3ForCausalLM": "nemo_automodel.components.models.mistral3.model",
    "NemotronHForCausalLM": "nemo_automodel.components.models.nemotron_v3.model",
    "NemotronParseForConditionalGeneration": "nemo_automodel.components.models.nemotron_parse.model",
    "Qwen2ForCausalLM": "nemo_automodel.components.models.qwen2.model",
    "Qwen3MoeForCausalLM": "nemo_automodel.components.models.qwen3_moe.model",
    "Qwen3NextForCausalLM": "nemo_automodel.components.models.qwen3_next.model",
    "Qwen3OmniMoeThinkerForConditionalGeneration": "nemo_automodel.components.models.qwen3_omni_moe.model",
    "Qwen3VLMoeForConditionalGeneration": "nemo_automodel.components.models.qwen3_vl_moe.model",
    "Qwen3_5MoeForConditionalGeneration": "nemo_automodel.components.models.qwen3_5_moe.model",
    "Step3p5ForCausalLM": "nemo_automodel.components.models.step3p5.model",
}

# Maps an alternative architecture name to the canonical class name in
# _MODEL_ARCH_TO_MODULE.  Useful when the HF config ``architectures``
# field differs from the actual class name we ship.
_ALIASES: Dict[str, str] = {
    "Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniMoeThinkerForConditionalGeneration",
}


@dataclass
class _ModelRegistry:
    alias: Dict[str, str] = field(default_factory=dict)
    _cache: Dict[str, Type[nn.Module]] = field(default_factory=dict, repr=False)
    _dynamic: Dict[str, Type[nn.Module]] = field(default_factory=dict, repr=False)
    _walked_paths: Set[str] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self.alias.update(_ALIASES)

    @property
    def supported_models(self) -> Set[str]:
        return set(_MODEL_ARCH_TO_MODULE) | set(self.alias) | set(self._dynamic)

    def get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]:
        if model_arch in self._cache:
            return self._cache[model_arch]

        if model_arch in self._dynamic:
            return self._dynamic[model_arch]

        canonical = self.alias.get(model_arch, model_arch)

        if canonical in self._dynamic:
            cls = self._dynamic[canonical]
            self._cache[model_arch] = cls
            return cls

        module_path = _MODEL_ARCH_TO_MODULE.get(canonical)
        if module_path is None:
            raise KeyError(model_arch)

        module = importlib.import_module(module_path)
        cls = getattr(module, canonical)
        self._cache[model_arch] = cls
        return cls

    def register_modeling_path(self, path: str) -> None:
        """Walk a package tree and register any modules that export ModelClass."""
        if path not in self._walked_paths:
            self._walked_paths.add(path)
            self._walk_and_register(path)

    def _is_known(self, name: str) -> bool:
        if name in _MODEL_ARCH_TO_MODULE or name in self._dynamic:
            return True
        canonical = self.alias.get(name)
        return canonical is not None and (canonical in _MODEL_ARCH_TO_MODULE or canonical in self._dynamic)

    def _walk_and_register(self, modeling_path: str):
        reverse_alias: Dict[str, list] = {}
        for alias_name, canonical in self.alias.items():
            reverse_alias.setdefault(canonical, []).append(alias_name)

        package = importlib.import_module(modeling_path)
        for _, name, ispkg in pkgutil.walk_packages(package.__path__, modeling_path + "."):
            if not ispkg:
                try:
                    module = importlib.import_module(name)
                except Exception as e:
                    logger.warning(f"Ignore import error when loading {name}. {e}")
                    continue
                if hasattr(module, "ModelClass"):
                    entry = module.ModelClass
                    entries = entry if isinstance(entry, list) else [entry]
                    for cls in entries:
                        if not self._is_known(cls.__name__):
                            self._dynamic[cls.__name__] = cls
                        for a in reverse_alias.get(cls.__name__, []):
                            if not self._is_known(a):
                                self._dynamic[a] = cls


@lru_cache
def get_registry():
    return _ModelRegistry()

ModelRegistry = get_registry()
