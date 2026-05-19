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

from collections import OrderedDict
from functools import lru_cache

from nemo_automodel._transformers.registry.base import _ModelRegistry as _BaseModelRegistry
from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

# Static mapping: architecture name → ModelPackageSpec.
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
MODEL_ARCH_MAPPING: OrderedDict[str, ModelPackageSpec] = OrderedDict(
    [
        (
            "BaichuanForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.baichuan",
                class_name="BaichuanForCausalLM",
                architectures=("BaichuanForCausalLM",),
            ),
        ),
        (
            "DeepseekV3ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.deepseek_v3",
                class_name="DeepseekV3ForCausalLM",
                architectures=("DeepseekV3ForCausalLM",),
            ),
        ),
        (
            "DeepseekV32ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.deepseek_v32",
                class_name="DeepseekV32ForCausalLM",
                architectures=("DeepseekV32ForCausalLM",),
            ),
        ),
        (
            "DeepseekV4ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.deepseek_v4",
                class_name="DeepseekV4ForCausalLM",
                architectures=("DeepseekV4ForCausalLM",),
            ),
        ),
        (
            "Ernie4_5_MoeForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.ernie4_5",
                class_name="Ernie4_5_MoeForCausalLM",
                architectures=("Ernie4_5_MoeForCausalLM",),
            ),
        ),
        (
            "Glm4MoeForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.glm4_moe",
                class_name="Glm4MoeForCausalLM",
                architectures=("Glm4MoeForCausalLM",),
            ),
        ),
        (
            "Glm4MoeLiteForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.glm4_moe_lite",
                class_name="Glm4MoeLiteForCausalLM",
                architectures=("Glm4MoeLiteForCausalLM",),
            ),
        ),
        (
            "GlmMoeDsaForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.glm_moe_dsa",
                class_name="GlmMoeDsaForCausalLM",
                architectures=("GlmMoeDsaForCausalLM",),
            ),
        ),
        (
            "Gemma4ForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.gemma4_moe",
                class_name="Gemma4ForConditionalGeneration",
                architectures=("Gemma4ForConditionalGeneration",),
            ),
        ),
        (
            "GptOssForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.gpt_oss",
                class_name="GptOssForCausalLM",
                architectures=("GptOssForCausalLM",),
            ),
        ),
        (
            "KimiK25ForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.kimi_k25_vl",
                class_name="KimiK25VLForConditionalGeneration",
                architectures=("KimiK25ForConditionalGeneration",),
            ),
        ),
        (
            "KimiK25VLForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.kimi_k25_vl",
                class_name="KimiK25VLForConditionalGeneration",
                architectures=("KimiK25VLForConditionalGeneration",),
            ),
        ),
        (
            "KimiVLForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.kimivl",
                class_name="KimiVLForConditionalGeneration",
                architectures=("KimiVLForConditionalGeneration",),
            ),
        ),
        (
            "LlamaBidirectionalForSequenceClassification",
            ModelPackageSpec(
                package="nemo_automodel.components.models.llama_bidirectional",
                class_name="LlamaBidirectionalForSequenceClassification",
                tags=frozenset({"retrieval"}),
                architectures=("LlamaBidirectionalForSequenceClassification",),
            ),
        ),
        (
            "LlamaBidirectionalModel",
            ModelPackageSpec(
                package="nemo_automodel.components.models.llama_bidirectional",
                class_name="LlamaBidirectionalModel",
                tags=frozenset({"retrieval"}),
                architectures=("LlamaBidirectionalModel",),
            ),
        ),
        (
            "LlamaForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.llama",
                class_name="LlamaForCausalLM",
                architectures=("LlamaForCausalLM",),
            ),
        ),
        (
            "MiniMaxM2ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.minimax_m2",
                class_name="MiniMaxM2ForCausalLM",
                architectures=("MiniMaxM2ForCausalLM",),
            ),
        ),
        (
            "MiMoV2FlashForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.mimo_v2_flash",
                class_name="MiMoV2FlashForCausalLM",
                architectures=("MiMoV2FlashForCausalLM",),
            ),
        ),
        (
            "Ministral3ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.mistral3",
                class_name="Ministral3ForCausalLM",
                architectures=("Ministral3ForCausalLM",),
            ),
        ),
        (
            "Ministral3BidirectionalModel",
            ModelPackageSpec(
                package="nemo_automodel.components.models.ministral_bidirectional",
                class_name="Ministral3BidirectionalModel",
                tags=frozenset({"retrieval"}),
                architectures=("Ministral3BidirectionalModel",),
            ),
        ),
        (
            "Mistral4ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.mistral4",
                class_name="Mistral4ForCausalLM",
                architectures=("Mistral4ForCausalLM",),
            ),
        ),
        (
            "Mistral3ForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.mistral4",
                class_name="Mistral3ForConditionalGeneration",
                architectures=("Mistral3ForConditionalGeneration",),
            ),
        ),
        (
            "Mistral3FP8VLMForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.mistral3_vlm",
                class_name="Mistral3FP8VLMForConditionalGeneration",
                architectures=("Mistral3FP8VLMForConditionalGeneration",),
            ),
        ),
        (
            "NemotronHForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.nemotron_v3",
                class_name="NemotronHForCausalLM",
                architectures=("NemotronHForCausalLM",),
            ),
        ),
        (
            "NemotronH_Nano_Omni_Reasoning_V3",
            ModelPackageSpec(
                package="nemo_automodel.components.models.nemotron_omni",
                class_name="NemotronOmniForConditionalGeneration",
                architectures=("NemotronH_Nano_Omni_Reasoning_V3",),
            ),
        ),
        (
            "NemotronParseForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.nemotron_parse",
                class_name="NemotronParseForConditionalGeneration",
                architectures=("NemotronParseForConditionalGeneration",),
            ),
        ),
        (
            "LLaVAOneVision1_5_ForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.llava_onevision",
                class_name="LLaVAOneVision1_5_ForConditionalGeneration",
                architectures=("LLaVAOneVision1_5_ForConditionalGeneration",),
            ),
        ),
        (
            "HYV3ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.hy_v3",
                class_name="HYV3ForCausalLM",
                architectures=("HYV3ForCausalLM",),
            ),
        ),
        (
            "Qwen2ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.qwen2",
                class_name="Qwen2ForCausalLM",
                architectures=("Qwen2ForCausalLM",),
            ),
        ),
        (
            "Qwen3MoeForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.qwen3_moe",
                class_name="Qwen3MoeForCausalLM",
                architectures=("Qwen3MoeForCausalLM",),
            ),
        ),
        (
            "Qwen3NextForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.qwen3_next",
                class_name="Qwen3NextForCausalLM",
                architectures=("Qwen3NextForCausalLM",),
            ),
        ),
        (
            "Qwen3OmniMoeForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.qwen3_omni_moe",
                class_name="Qwen3OmniMoeThinkerForConditionalGeneration",
                architectures=("Qwen3OmniMoeForConditionalGeneration",),
            ),
        ),
        (
            "Qwen3VLMoeForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.qwen3_vl_moe",
                class_name="Qwen3VLMoeForConditionalGeneration",
                architectures=("Qwen3VLMoeForConditionalGeneration",),
            ),
        ),
        (
            "Qwen3_5MoeForConditionalGeneration",
            ModelPackageSpec(
                package="nemo_automodel.components.models.qwen3_5_moe",
                class_name="Qwen3_5MoeForConditionalGeneration",
                architectures=("Qwen3_5MoeForConditionalGeneration",),
            ),
        ),
        (
            "Step3p5ForCausalLM",
            ModelPackageSpec(
                package="nemo_automodel.components.models.step3p5",
                class_name="Step3p5ForCausalLM",
                architectures=("Step3p5ForCausalLM",),
            ),
        ),
    ]
)


_CUSTOM_CONFIG_SPECS: tuple[ModelPackageSpec, ...] = (
    ModelPackageSpec(
        package="nemo_automodel.components.models.baichuan",
        model_types=("baichuan",),
        config_module="configuration",
        config_class_name="BaichuanConfig",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.deepseek_v4",
        model_types=("deepseek_v4",),
        config_module="config",
        config_class_name="DeepseekV4Config",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.hy_v3",
        model_types=("hy_v3",),
        config_module="config",
        config_class_name="HYV3Config",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.kimi_k25_vl",
        model_types=("kimi_k25", "kimi_k25_vl"),
        config_module="model",
        config_class_name="KimiK25VLConfig",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.kimivl",
        model_types=("kimi_vl",),
        config_module="model",
        config_class_name="KimiVLConfig",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.llava_onevision",
        model_types=("llavaonevision1_5",),
        config_module="model",
        config_class_name="Llavaonevision1_5Config",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mimo_v2_flash",
        model_types=("mimo_v2_flash",),
        config_module="config",
        config_class_name="MiMoV2FlashConfig",
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mistral4",
        model_types=("mistral4",),
        config_module="configuration",
        config_class_name="Mistral4Config",
    ),
)


MODEL_PACKAGE_SPECS: tuple[ModelPackageSpec, ...] = (*_CUSTOM_CONFIG_SPECS,)


class _ModelRegistry(_BaseModelRegistry):
    """Model-specific registry initialized from AutoModel's static mappings."""

    def __init__(
        self,
        model_arch_name_to_cls=None,
        package_specs: tuple[ModelPackageSpec, ...] = MODEL_PACKAGE_SPECS,
        model_arch_mapping: OrderedDict[str, ModelPackageSpec] | dict[str, ModelPackageSpec] = MODEL_ARCH_MAPPING,
    ) -> None:
        super().__init__(
            model_arch_mapping=model_arch_mapping,
            model_arch_name_to_cls=model_arch_name_to_cls,
            package_specs=package_specs,
        )


@lru_cache
def get_registry() -> _ModelRegistry:
    """Return the process-wide model registry singleton."""
    return _ModelRegistry()


ModelRegistry = get_registry()
