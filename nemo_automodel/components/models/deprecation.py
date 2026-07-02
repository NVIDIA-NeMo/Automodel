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

"""Deprecation warnings for custom model classes scheduled for removal."""

from __future__ import annotations

import warnings

_DEPRECATED_MODEL_YAMLS: dict[str, tuple[str, ...]] = {
    "BaichuanForCausalLM": (
        "examples/llm_finetune/baichuan/baichuan_2_7b_mock_fp8.yaml",
        "examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml",
        "examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml",
    ),
    "Qwen2ForCausalLM": (
        "examples/llm_benchmark/qwen/custom_qwen2_5_32b_peft_benchmark.yaml",
        "examples/llm_benchmark/qwen/custom_qwen2_5_32b_peft_benchmark_2nodes.yaml",
        "examples/llm_benchmark/qwen/qwen2_5_7b_peft_benchmark.yaml",
        "examples/llm_finetune/agent/qwen2_5_3b_function_calling.yaml",
        "examples/llm_finetune/agent/qwen2_5_3b_function_calling_lora.yaml",
        "examples/llm_finetune/qwen/qwen2_5_0p5b_instruct_fineproofs_chat.yaml",
        "examples/llm_finetune/qwen/qwen2_5_32b_peft_benchmark.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_hellaswag_fp8.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_instruct_chat.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_squad_muon.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml",
        "examples/llm_finetune/qwen/qwen25_magi_prefix_tree_rollouts.yaml",
        "examples/llm_finetune/seed/seed_coder_8b_instruct_hellaswag_fp8.yaml",
        "examples/llm_finetune/seed/seed_coder_8b_instruct_squad.yaml",
        "examples/llm_finetune/seed/seed_coder_8b_instruct_squad_peft.yaml",
        "examples/llm_finetune/seed/seed_oss_36B_hellaswag.yaml",
        "examples/llm_finetune/seed/seed_oss_36B_hellaswag_peft.yaml",
    ),
    "LlamaForCausalLM": (
        "examples/llm_benchmark/llama3_1/llama3_1_8b_peft_benchmark.yaml",
        "examples/llm_benchmark/llama3_3/custom_llama3_1_70b_pretrain_benchmark_8nodes.yaml",
        "examples/llm_benchmark/llama3_3/custom_llama3_3_70b_instruct_peft_benchmark.yaml",
        "examples/llm_benchmark/llama3_3/custom_llama3_3_70b_instruct_peft_benchmark_2nodes.yaml",
        "examples/llm_benchmark/llama3_3/llama3_3_70b_instruct_peft_benchmark.yaml",
        "examples/llm_benchmark/llama3_3/llama_3_3_70b_instruct_squad_peft_pp.yaml",
        "examples/llm_finetune/llama3_1/customizer_llama_3_1_8b_full_sft_tp.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_columnmapped_lora.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_fp8.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp_dynamic_seq_len.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_instruct_squad_qlora_2node.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml",
        "examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad.yaml",
        "examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad_peft.yaml",
        "examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad_peft_qlora_spark.yaml",
        "examples/llm_pretrain/llama3_70b_pretrain.yaml",
        "examples/speculative/eagle1/llama_eagle1_perfectblend.yaml",
        "examples/speculative/eagle2/llama_eagle2_perfectblend.yaml",
        "examples/speculative/eagle3/llama_eagle3_mvp.yaml",
        "examples/speculative/eagle3/llama_eagle3_mvp_flash_attn.yaml",
        "examples/speculative/eagle3/llama_eagle3_perfectblend.yaml",
        "examples/speculative/eagle3/llama_eagle3_remote.yaml",
        "examples/speculative/eagle3_1/llama_eagle3_1_mvp.yaml",
        "examples/speculative/eagle3_1/llama_eagle3_1_perfectblend.yaml",
        "examples/speculative/p-eagle/llama_peagle_mvp.yaml",
    ),
    "Glm4MoeForCausalLM": (
        "examples/llm_benchmark/glm/glm_4.5_air_te_deepep.yaml",
        "examples/llm_benchmark/glm/glm_4.7_te_deepep.yaml",
        "examples/llm_benchmark/glm/glm47_lora.yaml",
        "examples/llm_finetune/glm/glm_4.5_air_te_deepep.yaml",
        "examples/llm_finetune/glm/glm_4.7_te_deepep.yaml",
    ),
    "Glm4MoeLiteForCausalLM": (
        "examples/llm_benchmark/glm/glm_4.7_flash_lora.yaml",
        "examples/llm_benchmark/glm/glm_4.7_flash_te_deepep.yaml",
        "examples/llm_benchmark/glm/glm_4.7_flash_te_deepep_lora.yaml",
        "examples/llm_finetune/glm/glm_4.7_flash_te_deepep.yaml",
        "examples/llm_finetune/glm/glm_4.7_flash_te_packed_sequence.yaml",
    ),
    "KimiVLForConditionalGeneration": ("examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml",),
}


def _llama_config_name_or_path(config: object) -> str:
    """Return the HF checkpoint id stored on a Llama config, if any."""
    return str(getattr(config, "_name_or_path", "") or getattr(config, "name_or_path", "") or "")


def is_llama_3_2_1b_config(config: object) -> bool:
    """Return True when *config* refers to a supported Llama 3.2 1B checkpoint."""
    return "Llama-3.2-1B" in _llama_config_name_or_path(config)


def warn_deprecated_llama_for_causal_lm(config: object) -> None:
    """Emit a deprecation warning for Llama 3.x except supported 3.2-1B checkpoints."""
    if is_llama_3_2_1b_config(config):
        return
    warn_deprecated_model_class("LlamaForCausalLM")


def warn_deprecated_model_class(model_cls_name: str) -> None:
    """Emit a deprecation warning for custom model classes removed in 26.10.

    Args:
        model_cls_name: Name of the model class being instantiated.
    """
    yaml_paths = _DEPRECATED_MODEL_YAMLS.get(model_cls_name)
    if yaml_paths is None:
        return

    yaml_list = ", ".join(yaml_paths)
    warnings.warn(
        f"{model_cls_name} is deprecated and will be removed in NeMo AutoModel 26.10 container release and NeMo-Automodel v0.7.0. "
        f"Associated example configs: {yaml_list}",
        category=FutureWarning,
        stacklevel=3,
    )
