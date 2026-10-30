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

"""Shared documentation catalog metadata and generation markers."""

# Architectures documented under a different literal name in
# docs/model-coverage/. Each value must appear verbatim in at least one .mdx
# file under docs/model-coverage/.
#
# Add an entry here ONLY when the documentation legitimately uses a different
# name than the registry / HF class name (e.g., HF class name differs from
# registry alias, case differences, or a variant is grouped on a shared family
# page). New entries should include a short inline comment explaining the
# mismatch.
#
# Shared by registry validation, the generated architecture table, and
# optional architecture resolution from example YAMLs.
_DOC_ARCH_ALIASES = {
    # HF ships the class as ``BaiChuanForCausalLM`` (CamelCase) — registry
    # uses ``BaichuanForCausalLM``. Documented on the Baichuan page.
    "BaichuanForCausalLM": "BaiChuanForCausalLM",
    # HF upstream renamed ``Gemma3nForConditionalGeneration`` between releases;
    # the "Gemma 3n" variant is covered on the Gemma 3 VL page.
    "Gemma3nForConditionalGeneration": "Gemma 3n",
    # Checkpoint-facing alias of ``KimiK25VLForConditionalGeneration``, covered
    # by the Kimi-VL page.
    "KimiK25ForConditionalGeneration": "Kimi-K25-VL",
    "KimiK25VLForConditionalGeneration": "Kimi-K25-VL",
    # Retrieval/bi-encoder variants of Llama, covered on the GritLM page.
    "LlamaBidirectionalForSequenceClassification": "GritLM",
    "LlamaBidirectionalModel": "GritLM",
    # Multimodal retrieval variant covered by the existing NVIDIA Llama Nemotron embedding page.
    "LlamaNemotronVLModel": "llama-nemotron-embed-1b-v2",
    # HF ships ``LlavaOnevisionForConditionalGeneration`` (lowercase "n");
    # registry uses ``LlavaOneVisionForConditionalGeneration`` (the NVIDIA
    # re-impl for LLaVA-OneVision-1.5 with RICE ViT).
    "LlavaOneVisionForConditionalGeneration": "LlavaOnevisionForConditionalGeneration",
    # Registry also exposes the NVIDIA LLaVA-OneVision-1.5 re-impl under the
    # class name ``LLaVAOneVision1_5_ForConditionalGeneration`` (all-caps
    # "LLaVA" + explicit "1_5_" infix). The same model is documented on the
    # lmms-lab/llava-onevision page under ``LlavaOneVisionForConditionalGeneration``.
    "LLaVAOneVision1_5_ForConditionalGeneration": "LlavaOneVisionForConditionalGeneration",
    # Ministral3 text model; covered on the Ministral3 / Ministral3-VL pages
    # that list the VLM arch ``Mistral3ForConditionalGeneration``.
    "Ministral3ForCausalLM": "Mistral3ForConditionalGeneration",
    # Bi-encoder variant of Ministral3, covered on the same Ministral3 / Ministral3-VL pages.
    "Ministral3BidirectionalModel": "Mistral3ForConditionalGeneration",
    # Mistral4 text model is the backbone of Mistral-Small-4 VLM; documented
    # on the Mistral-Small-4 page via the recipe path ``mistral4``.
    "Mistral4ForCausalLM": "mistral4",
    # OLMo2 page uses the vendor-branded spelling ``OLMo2`` (all caps "OLM");
    # HF normalized the class name to ``Olmo2``.
    "Olmo2ForCausalLM": "OLMo2ForCausalLM",
    # HF upstream added an extra underscore between "5" and "VL"
    # (``Qwen2_5_VLForConditionalGeneration``); the Qwen2.5-VL page still uses
    # the pre-rename spelling.
    "Qwen2_5_VLForConditionalGeneration": "Qwen2_5VLForConditionalGeneration",
    # Qwen3-Omni and Qwen3-VL are documented with the VL-facing arch name; the
    # registry wires their MoE backbones under these keys.
    "Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration": "Qwen3VLForConditionalGeneration",
    # Qwen3.5-MoE text and VL entry points share the unified Qwen3.5 page.
    "Qwen3_5MoeForCausalLM": "Qwen3.5-MoE",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3.5-MoE",
    # Dense Qwen3.5 text/VL backbone; grouped with the VL variants on the
    # Qwen3.5-VL page.
    "Qwen3_5ForCausalLM": "Qwen3.5",
    "Qwen3_5ForConditionalGeneration": "Qwen3.5",
    # Legacy checkpoint architecture name for the renamed Qwen3.8-Flash-Next
    # implementation, documented under its current architecture name.
    "Qwen4ExpForConditionalGeneration": "Qwen3_8_FlashNextForConditionalGeneration",
    # HF split Seed-OSS into its own arch; the Seed page (``seed.mdx``) covers
    # both Seed-Coder and Seed-OSS under the "Seed-OSS" name.
    "SeedOssForCausalLM": "Seed-OSS",
}


TABLE_ROW_COUNT = 10


HOMEPAGE_START_MARKER = "{/* BEGIN GENERATED LATEST MODEL SUPPORT */}"


HOMEPAGE_END_MARKER = "{/* END GENERATED LATEST MODEL SUPPORT */}"


SUPPORT_LOG_START_MARKER = "{/* BEGIN GENERATED MODEL SUPPORT LOG */}"


SUPPORT_LOG_END_MARKER = "{/* END GENERATED MODEL SUPPORT LOG */}"


REGISTRY_START_MARKER = "{/* BEGIN GENERATED MODEL ARCHITECTURES */}"


REGISTRY_END_MARKER = "{/* END GENERATED MODEL ARCHITECTURES */}"


DATED_SUPPORT_TABLE_HEADER = "| Date | Type | Model | Recipe |"


DATED_MODEL_TABLE_HEADER = "| Date | Model | Architectures | Recipe |"


MODEL_TYPE_OVERVIEW_PATHS = (
    ("LLM", "llm/index.mdx"),
    ("VLM", "vlm/index.mdx"),
    ("Multimodal", "multimodal/index.mdx"),
    ("Omni", "omni/index.mdx"),
    ("dLLM", "dllm/index.mdx"),
    ("Diffusion", "diffusion/index.mdx"),
    ("Embedding", "embedding/index.mdx"),
    ("Reranking", "reranker/index.mdx"),
)


MODEL_TYPE_ROUTE_CATEGORIES = {
    "LLM": "large-language-models",
    "Encoder-Decoder": "large-language-models",
    "VLM": "vision-language-models",
    "Multimodal": "multimodal",
    "Omni": "omni",
    "dLLM": "dllm",
    "Diffusion": "diffusion",
    "Embedding": "embedding-models",
    "Reranking": "reranking-models",
}


GENERATED_MARKER_PAIRS = (
    (HOMEPAGE_START_MARKER, HOMEPAGE_END_MARKER),
    (SUPPORT_LOG_START_MARKER, SUPPORT_LOG_END_MARKER),
    (REGISTRY_START_MARKER, REGISTRY_END_MARKER),
)


PROVIDER_ORGS = {
    "allenai": "allenai",
    "baai": "BAAI",
    "baichuan-inc": "baichuan-inc",
    "baidu": "baidu",
    "bigcode": "bigcode",
    "black-forest-labs": "black-forest-labs",
    "bytedance-seed": "ByteDance-Seed",
    "cohere": "CohereForAI",
    "coherelabs": "CohereLabs",
    "deepseek-ai": "deepseek-ai",
    "diffusers": "diffusers",
    "eleutherai": "EleutherAI",
    "google": "google",
    "google-t5": "google-t5",
    "gsai-ml": "GSAI-ML",
    "huggingfacetb": "HuggingFaceTB",
    "hunyuanvideo-community": "hunyuanvideo-community",
    "ibm": "ibm-granite",
    "ibm-ai-platform": "ibm-ai-platform",
    "inceptionai": "inceptionai",
    "inclusionai": "inclusionAI",
    "internlm": "internlm",
    "lgai-exaone": "LGAI-EXAONE",
    "lightricks": "Lightricks",
    "liquidai": "LiquidAI",
    "llava-hf": "llava-hf",
    "lmms-lab": "lmms-lab",
    "meta": "meta-llama",
    "microsoft": "microsoft",
    "MiniMaxAI": "MiniMaxAI",
    "mistralai": "mistralai",
    "moonshotai": "moonshotai",
    "nanbeige": "Nanbeige",
    "meta-models": "meta-models",
    "nvidia": "nvidia",
    "openai": "openai",
    "openai-community": "openai-community",
    "openbmb": "openbmb",
    "opengvlab": "OpenGVLab",
    "orionstar": "OrionStarAI",
    "parasail-ai": "parasail-ai",
    "poolside": "poolside",
    "qwen": "Qwen",
    "recursivesi": "RecursiveSI",
    "reflection": "reflection",
    "stabilityai": "stabilityai",
    "stepfun-ai": "stepfun-ai",
    "tencent": "tencent",
    "thinkingmachines": "thinkingmachines",
    "thudm": "zai-org",
    "tiiuae": "tiiuae",
    "upstage": "upstage",
    "wan-ai": "Wan-AI",
    "xiaomimimo": "XiaomiMiMo",
    "z-lab": "z-lab",
}


MAX_TOTAL_ENCODED_BYTES = 256 * 1024
