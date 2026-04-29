# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared data for doc-coverage checks.

The ``Doc coverage`` GitHub Actions workflow (via
``.github/scripts/post_doc_coverage_comment.py``) loads this module by
file path to read ``_DOC_ARCH_ALIASES``. The recipe-coverage pytest
(``test_recipe_doc_coverage.py``) imports it as a regular package
member. Keep this file pure data + stdlib so both consumers stay
import-cheap.
"""

# Architectures documented under a different literal name in
# docs/model-coverage/. Each value must appear verbatim in at least one .md
# file under docs/model-coverage/.
#
# Add an entry here ONLY when the documentation legitimately uses a different
# name than the registry / HF class name (e.g., HF class name differs from
# registry alias, case differences, or a variant is grouped on a shared family
# page). New entries should include a short inline comment explaining the
# mismatch.
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
    # Qwen3-Omni, Qwen3-VL and Qwen3.5-VL are documented with the VL-facing
    # arch name; the registry wires their MoE backbones under these keys.
    "Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration": "Qwen3VLForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeVLForConditionalGeneration",
    # Dense Qwen3.5 text/VL backbone; grouped with the VL variants on the
    # Qwen3.5-VL page.
    "Qwen3_5ForConditionalGeneration": "Qwen3.5",
    # HF split Seed-OSS into its own arch; the Seed page (``seed.md``) covers
    # both Seed-Coder and Seed-OSS under the "Seed-OSS" name.
    "SeedOssForCausalLM": "Seed-OSS",
}
