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

from functools import lru_cache

from nemo_automodel._transformers.registry.base import _BaseModelRegistry
from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

# Static model package specs. The registry derives architecture lookups from each spec's architectures.
# Analogous to HuggingFace transformers' MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.
# Models are loaded lazily on first access rather than imported at startup.
MODEL_PACKAGE_SPECS: tuple[ModelPackageSpec, ...] = (
    ModelPackageSpec(
        package="nemo_automodel.components.models.baichuan",
        class_name="BaichuanForCausalLM",
        config_module="configuration",
        config_class_name="BaichuanConfig",
        architectures=("BaichuanForCausalLM",),
        model_types=("baichuan",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.deepseek_v3",
        class_name="DeepseekV3ForCausalLM",
        architectures=("DeepseekV3ForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.deepseek_v32",
        class_name="DeepseekV32ForCausalLM",
        architectures=("DeepseekV32ForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.deepseek_v4",
        class_name="DeepseekV4ForCausalLM",
        config_module="config",
        config_class_name="DeepseekV4Config",
        architectures=("DeepseekV4ForCausalLM",),
        model_types=("deepseek_v4",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.ernie4_5",
        class_name="Ernie4_5_MoeForCausalLM",
        architectures=("Ernie4_5_MoeForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.glm4_moe",
        class_name="Glm4MoeForCausalLM",
        architectures=("Glm4MoeForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.glm4_moe_lite",
        class_name="Glm4MoeLiteForCausalLM",
        architectures=("Glm4MoeLiteForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.glm_moe_dsa",
        class_name="GlmMoeDsaForCausalLM",
        architectures=("GlmMoeDsaForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.gemma4_moe",
        class_name="Gemma4ForConditionalGeneration",
        architectures=("Gemma4ForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.gpt_oss",
        class_name="GptOssForCausalLM",
        architectures=("GptOssForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.kimi_k25_vl",
        class_name="KimiK25VLForConditionalGeneration",
        config_module="model",
        config_class_name="KimiK25VLConfig",
        architectures=("KimiK25ForConditionalGeneration", "KimiK25VLForConditionalGeneration"),
        model_types=("kimi_k25", "kimi_k25_vl"),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.kimivl",
        class_name="KimiVLForConditionalGeneration",
        config_module="model",
        config_class_name="KimiVLConfig",
        architectures=("KimiVLForConditionalGeneration",),
        model_types=("kimi_vl",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.llama",
        class_name="LlamaForCausalLM",
        architectures=("LlamaForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.minimax_m2",
        class_name="MiniMaxM2ForCausalLM",
        architectures=("MiniMaxM2ForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mimo_v2_flash",
        class_name="MiMoV2FlashForCausalLM",
        config_module="config",
        config_class_name="MiMoV2FlashConfig",
        architectures=("MiMoV2FlashForCausalLM",),
        model_types=("mimo_v2_flash",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mistral3",
        class_name="Ministral3ForCausalLM",
        architectures=("Ministral3ForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mistral4",
        class_name="Mistral4ForCausalLM",
        config_module="configuration",
        config_class_name="Mistral4Config",
        architectures=("Mistral4ForCausalLM",),
        model_types=("mistral4",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mistral4",
        class_name="Mistral3ForConditionalGeneration",
        architectures=("Mistral3ForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.mistral3_vlm",
        class_name="Mistral3FP8VLMForConditionalGeneration",
        architectures=("Mistral3FP8VLMForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.nemotron_v3",
        class_name="NemotronHForCausalLM",
        architectures=("NemotronHForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.nemotron_omni",
        class_name="NemotronOmniForConditionalGeneration",
        architectures=("NemotronH_Nano_Omni_Reasoning_V3",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.nemotron_parse",
        class_name="NemotronParseForConditionalGeneration",
        architectures=("NemotronParseForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.llava_onevision",
        class_name="LLaVAOneVision1_5_ForConditionalGeneration",
        config_module="model",
        config_class_name="Llavaonevision1_5Config",
        architectures=("LLaVAOneVision1_5_ForConditionalGeneration",),
        model_types=("llavaonevision1_5",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.hy_v3",
        class_name="HYV3ForCausalLM",
        config_module="config",
        config_class_name="HYV3Config",
        architectures=("HYV3ForCausalLM",),
        model_types=("hy_v3",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.qwen2",
        class_name="Qwen2ForCausalLM",
        architectures=("Qwen2ForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.qwen3_moe",
        class_name="Qwen3MoeForCausalLM",
        architectures=("Qwen3MoeForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.qwen3_next",
        class_name="Qwen3NextForCausalLM",
        architectures=("Qwen3NextForCausalLM",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.qwen3_omni_moe",
        class_name="Qwen3OmniMoeThinkerForConditionalGeneration",
        architectures=("Qwen3OmniMoeForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.qwen3_vl_moe",
        class_name="Qwen3VLMoeForConditionalGeneration",
        architectures=("Qwen3VLMoeForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.qwen3_5_moe",
        class_name="Qwen3_5MoeForConditionalGeneration",
        architectures=("Qwen3_5MoeForConditionalGeneration",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.step3p5",
        class_name="Step3p5ForCausalLM",
        architectures=("Step3p5ForCausalLM",),
    ),
)

RETRIEVAL_MODEL_PACKAGE_SPECS: tuple[ModelPackageSpec, ...] = (
    ModelPackageSpec(
        package="nemo_automodel.components.models.llama_bidirectional",
        class_name="LlamaBidirectionalForSequenceClassification",
        architectures=("LlamaBidirectionalForSequenceClassification",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.llama_bidirectional",
        class_name="LlamaBidirectionalModel",
        architectures=("LlamaBidirectionalModel",),
    ),
    ModelPackageSpec(
        package="nemo_automodel.components.models.ministral_bidirectional",
        class_name="Ministral3BidirectionalModel",
        architectures=("Ministral3BidirectionalModel",),
    ),
)


@lru_cache
def make_registry(model_specs: tuple[ModelPackageSpec, ...]) -> _BaseModelRegistry:
    """Return a process-wide model registry singleton for package specs."""
    return _BaseModelRegistry(model_specs=model_specs)


ModelRegistry = make_registry(MODEL_PACKAGE_SPECS)
RetrievalModelRegistry = make_registry(RETRIEVAL_MODEL_PACKAGE_SPECS)
