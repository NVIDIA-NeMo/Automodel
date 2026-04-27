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


import importlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple, Type, Union

import torch.nn as nn

logger = logging.getLogger(__name__)

# Static mapping: architecture name → (module_path, class_name[, tags]).
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
# Optional third element is a set of tags (e.g. {"retrieval"}) used by
# downstream code to classify model archs without importing them.
MODEL_ARCH_MAPPING = OrderedDict(
    [
        (
            "BaichuanForCausalLM",
            ("nemo_automodel.components.models.baichuan.model", "BaichuanForCausalLM"),
        ),
        (
            "DeepseekV3ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v3.model", "DeepseekV3ForCausalLM"),
        ),
        (
            "DeepseekV32ForCausalLM",
            ("nemo_automodel.components.models.deepseek_v32.model", "DeepseekV32ForCausalLM"),
        ),
        (
            "Glm4MoeForCausalLM",
            ("nemo_automodel.components.models.glm4_moe.model", "Glm4MoeForCausalLM"),
        ),
        (
            "Glm4MoeLiteForCausalLM",
            ("nemo_automodel.components.models.glm4_moe_lite.model", "Glm4MoeLiteForCausalLM"),
        ),
        (
            "GlmMoeDsaForCausalLM",
            ("nemo_automodel.components.models.glm_moe_dsa.model", "GlmMoeDsaForCausalLM"),
        ),
        (
            "Gemma4ForConditionalGeneration",
            ("nemo_automodel.components.models.gemma4_moe.model", "Gemma4ForConditionalGeneration"),
        ),
        (
            "GptOssForCausalLM",
            ("nemo_automodel.components.models.gpt_oss.model", "GptOssForCausalLM"),
        ),
        (
            "KimiK25ForConditionalGeneration",
            ("nemo_automodel.components.models.kimi_k25_vl.model", "KimiK25VLForConditionalGeneration"),
        ),
        (
            "KimiK25VLForConditionalGeneration",
            ("nemo_automodel.components.models.kimi_k25_vl.model", "KimiK25VLForConditionalGeneration"),
        ),
        (
            "KimiVLForConditionalGeneration",
            ("nemo_automodel.components.models.kimivl.model", "KimiVLForConditionalGeneration"),
        ),
        (
            "LlamaBidirectionalForSequenceClassification",
            (
                "nemo_automodel.components.models.llama_bidirectional.model",
                "LlamaBidirectionalForSequenceClassification",
                {"retrieval"},
            ),
        ),
        (
            "LlamaBidirectionalModel",
            ("nemo_automodel.components.models.llama_bidirectional.model", "LlamaBidirectionalModel", {"retrieval"}),
        ),
        (
            "LlamaForCausalLM",
            ("nemo_automodel.components.models.llama.model", "LlamaForCausalLM"),
        ),
        (
            "MiniMaxM2ForCausalLM",
            ("nemo_automodel.components.models.minimax_m2.model", "MiniMaxM2ForCausalLM"),
        ),
        (
            "Ministral3ForCausalLM",
            ("nemo_automodel.components.models.mistral3.model", "Ministral3ForCausalLM"),
        ),
        (
            "Mistral4ForCausalLM",
            ("nemo_automodel.components.models.mistral4.model", "Mistral4ForCausalLM"),
        ),
        (
            "Mistral3ForConditionalGeneration",
            ("nemo_automodel.components.models.mistral4.model", "Mistral3ForConditionalGeneration"),
        ),
        (
            "NemotronHForCausalLM",
            ("nemo_automodel.components.models.nemotron_v3.model", "NemotronHForCausalLM"),
        ),
        (
            "NemotronH_Nano_Omni_Reasoning_V3",
            (
                "nemo_automodel.components.models.nemotron_omni.model",
                "NemotronOmniForConditionalGeneration",
            ),
        ),
        (
            "NemotronParseForConditionalGeneration",
            ("nemo_automodel.components.models.nemotron_parse.model", "NemotronParseForConditionalGeneration"),
        ),
        (
            "LlavaOneVisionForConditionalGeneration",
            (
                "nemo_automodel.components.models.llava_onevision.model",
                "LlavaOneVisionForConditionalGeneration",
            ),
        ),
        (
            "Qwen2ForCausalLM",
            ("nemo_automodel.components.models.qwen2.model", "Qwen2ForCausalLM"),
        ),
        (
            "Qwen3MoeForCausalLM",
            ("nemo_automodel.components.models.qwen3_moe.model", "Qwen3MoeForCausalLM"),
        ),
        (
            "Qwen3NextForCausalLM",
            ("nemo_automodel.components.models.qwen3_next.model", "Qwen3NextForCausalLM"),
        ),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            (
                "nemo_automodel.components.models.qwen3_omni_moe.model",
                "Qwen3OmniMoeThinkerForConditionalGeneration",
            ),
        ),
        (
            "Qwen3VLMoeForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_vl_moe.model", "Qwen3VLMoeForConditionalGeneration"),
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration"),
        ),
        (
            "Step3p5ForCausalLM",
            ("nemo_automodel.components.models.step3p5.model", "Step3p5ForCausalLM"),
        ),
    ]
)


# Custom model_type → config class for models that have auto_map in their
# checkpoint config.json.  Registered eagerly with AutoConfig so that
# AutoConfig.from_pretrained can resolve them without trust_remote_code.
_CUSTOM_CONFIG_REGISTRATIONS: Dict[str, Tuple[str, str]] = {
    "baichuan": ("nemo_automodel.components.models.baichuan.configuration", "BaichuanConfig"),
    "kimi_k25": ("nemo_automodel.components.models.kimi_k25_vl.model", "KimiK25VLConfig"),
    "kimi_vl": ("nemo_automodel.components.models.kimivl.model", "KimiVLConfig"),
    "llava_onevision": ("nemo_automodel.components.models.llava_onevision.model", "LlavaOneVisionConfig"),
    "mistral4": ("nemo_automodel.components.models.mistral4.configuration", "Mistral4Config"),
}


def _register_custom_configs() -> None:
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for model_type, (module_path, cls_name) in _CUSTOM_CONFIG_REGISTRATIONS.items():
        if model_type not in CONFIG_MAPPING:
            try:
                mod = importlib.import_module(module_path)
                cfg_cls = getattr(mod, cls_name)
                AutoConfig.register(model_type, cfg_cls)
            except Exception:
                logger.debug("Failed to register config for model_type=%s", model_type, exc_info=True)


_register_custom_configs()


class _LazyArchMapping:
    """Lazy-loading mapping from architecture name to model class.

    Inspired by HuggingFace transformers' ``_LazyAutoMapping``.  Entries from the
    static ``auto_map`` are imported on first access and cached.  Additional entries
    can be added at runtime via ``register``.
    """

    def __init__(self, auto_map: Union[OrderedDict, Dict[str, tuple], None] = None):
        # Entries may be (module_path, class_name) or (module_path, class_name, tags).
        # Strip the optional tags and store them separately.
        self._auto_map: Dict[str, Tuple[str, str]] = OrderedDict()
        self._tags: Dict[str, set] = {}
        for key, value in (auto_map or {}).items():
            self._auto_map[key] = (value[0], value[1])
            if len(value) > 2:
                self._tags[key] = value[2]
        self._loaded: Dict[str, Type[nn.Module]] = {}
        self._extra: Dict[str, Type[nn.Module]] = {}
        self._modules: Dict[str, object] = {}

    def _load(self, key: str) -> Type[nn.Module]:
        if key in self._loaded:
            return self._loaded[key]
        module_path, class_name = self._auto_map[key]
        if module_path not in self._modules:
            self._modules[module_path] = importlib.import_module(module_path)
        cls = getattr(self._modules[module_path], class_name)
        self._loaded[key] = cls
        return cls

    def __contains__(self, key: str) -> bool:
        if key in self._extra or key in self._loaded:
            return True
        if key not in self._auto_map:
            return False
        try:
            self._load(key)
            return True
        except Exception:
            logger.debug("Model %s unavailable (import failed), removing from auto_map", key)
            self._auto_map.pop(key, None)
            return False

    def __getitem__(self, key: str) -> Type[nn.Module]:
        if key in self._extra:
            return self._extra[key]
        if key in self._auto_map:
            return self._load(key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Type[nn.Module]) -> None:
        self._extra[key] = value

    def register(self, key: str, value: Type[nn.Module], exist_ok: bool = False) -> None:
        """Register a model class under the given architecture name."""
        if not exist_ok and key in self._extra:
            raise ValueError(f"Duplicated model implementation for {key}")
        self._extra[key] = value

    def has_tag(self, key: str, tag: str) -> bool:
        """Return ``True`` if *key* was registered with *tag*."""
        return tag in self._tags.get(key, set())

    def keys_with_tag(self, tag: str) -> set:
        """Return all architecture names that have *tag*."""
        return {k for k, tags in self._tags.items() if tag in tags}

    def keys(self):
        return set(self._auto_map.keys()) | set(self._extra.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"_LazyArchMapping(auto_map={len(self._auto_map)}, extra={len(self._extra)}, loaded={len(self._loaded)})"


@dataclass
class _ModelRegistry:
    model_arch_name_to_cls: _LazyArchMapping = field(default=None)
    _retrieval_archs: set = field(default_factory=set)

    def __post_init__(self):
        if self.model_arch_name_to_cls is None:
            self.model_arch_name_to_cls = _LazyArchMapping(MODEL_ARCH_MAPPING)
        self._retrieval_archs = self.model_arch_name_to_cls.keys_with_tag("retrieval")

    @property
    def supported_models(self):
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def has_custom_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* has a custom (non-HF) implementation."""
        return arch_name in self.model_arch_name_to_cls

    def has_retrieval_model(self, arch_name: str) -> bool:
        """Return ``True`` if *arch_name* is a registered retrieval/encoder architecture."""
        return arch_name in self._retrieval_archs

    def register_retrieval(self, arch_name: str) -> None:
        """Mark *arch_name* as a retrieval/encoder architecture."""
        self._retrieval_archs.add(arch_name)

    def resolve_custom_model_cls(self, architecture: str, config) -> Union[Type[nn.Module], None]:
        """Return the custom model class if it exists and supports *config*, else ``None``.

        Custom model classes may define a ``supports_config(config)`` classmethod
        to opt out for specific HF configs (e.g. a Mistral3 VLM with a dense
        Ministral3 text backbone instead of the expected Mistral4 MoE+MLA).
        """
        if architecture not in self.model_arch_name_to_cls:
            return None
        model_cls = self.model_arch_name_to_cls[architecture]
        if hasattr(model_cls, "supports_config") and not model_cls.supports_config(config):
            logger.info(
                "Custom model %s does not support config %s, falling back to HF",
                model_cls.__name__,
                type(config).__name__,
            )
            return None
        return model_cls

    def register(self, arch_name: str, model_cls: Type[nn.Module], exist_ok: bool = False) -> None:
        """Register a custom model class for a given architecture name."""
        self.model_arch_name_to_cls.register(arch_name, model_cls, exist_ok=exist_ok)


@lru_cache
def get_registry():
    return _ModelRegistry()


ModelRegistry = get_registry()
