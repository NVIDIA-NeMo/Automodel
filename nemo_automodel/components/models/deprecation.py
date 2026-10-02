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

"""Deprecation warnings for legacy model classes and checkpoint examples."""

from __future__ import annotations

import os
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
    "KimiVLForConditionalGeneration": ("examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml",),
}


# Checkpoint deprecations approved from the 2026-10-01 age audit.
# Selected release or initial HF commit predates 2024-10-01 (24 months);
# the 9 entries before 2023-10-01 are included. This is a fixed list, not a
# rolling age policy. Each entry records the audit date and primary source.
# Paths include teacher/tokenizer/processor references; deprecation applies
# to the named checkpoint, not every model using one of these recipe files.
_DEPRECATED_CHECKPOINT_YAMLS: dict[str, tuple[str, ...]] = {
    # 2019-02-18: HF initial commit.
    # https://huggingface.co/openai-community/gpt2/commits/main
    "openai-community/gpt2": (
        "examples/llm_pretrain/megatron_pretrain_gpt2.yaml",
        "examples/llm_pretrain/megatron_pretrain_gpt2_domain_mixture.yaml",
    ),
    # 2019-12-11: HF initial commit.
    # https://huggingface.co/google-t5/t5-small/commits/main
    "google-t5/t5-small": ("examples/llm_finetune/t5/t5_small_squad.yaml",),
    # 2021-08-05: HF initial commit.
    # https://huggingface.co/EleutherAI/gpt-j-6b/commits/main
    "EleutherAI/gpt-j-6b": ("examples/llm_finetune/eleutherai/gpt_j_6b_squad_peft.yaml",),
    # 2022-04-07: HF initial commit.
    # https://huggingface.co/EleutherAI/gpt-neox-20b/commits/main
    "EleutherAI/gpt-neox-20b": ("examples/llm_finetune/eleutherai/gpt_neox_20b_squad_peft.yaml",),
    # 2023-04-24: HF initial commit.
    # https://huggingface.co/tiiuae/falcon-7b/commits/main
    "tiiuae/falcon-7b": ("examples/llm_finetune/tiiuae/falcon_7b_squad_peft.yaml",),
    # 2023-06-08: HF initial commit.
    # https://huggingface.co/BAAI/Aquila-7B/commits/8b3a0e2b06b67dbcc8fdc44168a19065d6a95a2e
    "BAAI/Aquila-7B": ("examples/llm_finetune/baai/aquila_7b_squad_peft.yaml",),
    # 2023-08-30: publisher release.
    # https://mbzuai.ac.ae/news/meet-jais-the-worlds-most-advanced-arabic-large-language-model-open-sourced-by-g42s-inception/
    "inceptionai/jais-13b": ("examples/llm_finetune/inceptionai/jais_13b_squad_peft.yaml",),
    # 2023-09-06: HF initial commit.
    # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/commits/main
    "baichuan-inc/Baichuan2-7B-Chat": (
        "examples/llm_finetune/baichuan/baichuan_2_7b_mock_fp8.yaml",
        "examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml",
        "examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml",
    ),
    # 2023-09-29: HF initial commit.
    # https://huggingface.co/stabilityai/stablelm-3b-4e1t/commits/main
    "stabilityai/stablelm-3b-4e1t": ("examples/llm_finetune/stabilityai/stablelm_3b_4e1t_squad_peft.yaml",),
    # 2023-10-25: HF initial commit.
    # https://huggingface.co/zai-org/chatglm3-6b/commits/main
    "zai-org/chatglm3-6b": ("examples/llm_finetune/zai_org/chatglm3_6b_squad_peft.yaml",),
    # 2023-11-29: HF initial commit.
    # https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat/commits/main
    "deepseek-ai/deepseek-llm-7b-chat": ("examples/llm_finetune/deepseek_ai/deepseek_llm_7b_chat_squad_peft.yaml",),
    # 2023-12-05: HF initial commit.
    # https://huggingface.co/llava-hf/llava-1.5-7b-hf/commits/main
    "llava-hf/llava-1.5-7b-hf": ("examples/vlm_finetune/llava_hf/llava_1_5_7b_hf_medpix_peft.yaml",),
    # 2024-01-21: publisher release.
    # https://us.orionstar.com/xwzx/225.html
    "OrionStarAI/Orion-14B-Base": ("examples/llm_finetune/orionstarai/orion_14b_base_squad_peft.yaml",),
    # 2024-02-11: HF initial commit.
    # https://huggingface.co/parasail-ai/GritLM-7B-vllm/commits/main
    # GritLM inherits upstream history; the conversion repo was created 2024-11-27.
    "parasail-ai/GritLM-7B-vllm": ("examples/llm_finetune/parasail_ai/gritlm_7b_vllm_squad_peft.yaml",),
    # 2024-02-29: HF initial commit.
    # https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/commits/main
    "Qwen/Qwen1.5-MoE-A2.7B": ("examples/llm_finetune/qwen/qwen1_5_moe_a2_7b_qlora.yaml",),
    # 2024-07-19: HF initial commit.
    # https://huggingface.co/nvidia/Minitron-8B-Base/commits/main/README.md
    "nvidia/Minitron-8B-Base": ("examples/llm_finetune/nvidia/minitron_8b_base_squad_peft.yaml",),
    # 2024-07-23: publisher release.
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
    "meta-llama/Llama-3.1-70B": (
        "examples/llm_benchmark/llama3_3/custom_llama3_1_70b_pretrain_benchmark_8nodes.yaml",
        "examples/llm_pretrain/llama3_70b_pretrain.yaml",
    ),
    # 2024-07-23: publisher release.
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
    "meta-llama/Llama-3.1-8B": (
        "examples/llm_benchmark/llama3_1/llama3_1_8b_peft_benchmark.yaml",
        "examples/llm_benchmark/llama3_1/llama3_1_8b_quack_rope.yaml",
        "examples/llm_benchmark/llama3_1/llama3_1_8b_quack_rope_baseline.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_columnmapped_lora.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_fp8.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp_dynamic_seq_len.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_squad_peft_rtx_spark.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml",
        "examples/retrieval/bi_encoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml",
    ),
    # 2024-07-23: publisher release.
    # https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
    "meta-llama/Llama-3.1-8B-Instruct": (
        "examples/llm_finetune/llama3_1/customizer_llama_3_1_8b_full_sft_tp.yaml",
        "examples/llm_finetune/llama3_1/llama3_1_8b_instruct_squad_qlora_2node.yaml",
    ),
    # 2024-08-01: publisher release.
    # https://bfl.ai/blog/24-08-01-bfl
    "black-forest-labs/FLUX.1-dev": (
        "examples/diffusion/finetune/flux_t2i_flow.yaml",
        "examples/diffusion/finetune/flux_t2i_flow_lora.yaml",
        "examples/diffusion/generate/configs/generate_flux.yaml",
        "examples/diffusion/pretrain/flux_t2i_flow.yaml",
    ),
    # 2024-08-07: publisher release.
    # https://www.lgresearch.ai/blog/view?seq=460
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct": (
        "examples/llm_finetune/lgai_exaone/exaone_3_0_7_8b_instruct_squad_peft.yaml",
    ),
    # 2024-09-08: publisher release.
    # https://huggingface.co/upstage/solar-pro-preview-instruct
    "upstage/solar-pro-preview-instruct": ("examples/llm_finetune/upstage/solar_pro_preview_instruct_squad_peft.yaml",),
    # 2024-09-15: HF initial commit.
    # https://huggingface.co/Qwen/Qwen2.5-0.5B/commits/main
    "Qwen/Qwen2.5-0.5B": ("examples/llm_finetune/qwen/qwen25_magi_prefix_tree_rollouts.yaml",),
    # 2024-09-15: HF initial commit.
    # https://huggingface.co/Qwen/Qwen2.5-3B/commits/main
    "Qwen/Qwen2.5-3B": (
        "examples/llm_finetune/agent/qwen2_5_3b_function_calling.yaml",
        "examples/llm_finetune/agent/qwen2_5_3b_function_calling_lora.yaml",
    ),
    # 2024-09-15: HF initial commit.
    # https://huggingface.co/Qwen/Qwen2.5-7B/commits/main
    "Qwen/Qwen2.5-7B": (
        "examples/llm_benchmark/qwen/qwen2_5_7b_peft_benchmark.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_hellaswag_fp8.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_squad_muon.yaml",
        "examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml",
    ),
    # 2024-09-16: HF initial commit.
    # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commits/main
    "Qwen/Qwen2.5-0.5B-Instruct": ("examples/llm_finetune/qwen/qwen2_5_0p5b_instruct_fineproofs_chat.yaml",),
    # 2024-09-16: HF initial commit.
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/commits/main
    "Qwen/Qwen2.5-7B-Instruct": (
        "examples/llm_finetune/qwen/qwen2_5_7b_instruct_chat.yaml",
        "examples/multimodal_pretrain/bagel/bagel_pretrain.yaml",
    ),
    # 2024-09-17: HF initial commit.
    # https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/commits/main
    "Qwen/Qwen2.5-32B-Instruct": (
        "examples/llm_benchmark/qwen/custom_qwen2_5_32b_peft_benchmark.yaml",
        "examples/llm_benchmark/qwen/custom_qwen2_5_32b_peft_benchmark_2nodes.yaml",
        "examples/llm_finetune/qwen/qwen2_5_32b_peft_benchmark.yaml",
    ),
    # 2024-09-25: publisher release.
    # https://huggingface.co/meta-llama/Llama-3.2-1B
    "meta-llama/Llama-3.2-1B": (
        "examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_hsdp.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_megatron_fsdp.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_packed_cp2.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_flashoptim.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_megatron_fsdp.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_neftune.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_qat.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_skypilot.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_skypilot_kubernetes.yaml",
        "examples/llm_finetune/llama3_2/llama3_2_1b_squad_skypilot_kubernetes_2nodes.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_cp2.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_pp2.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_tp2.yaml",
        "examples/retrieval/bi_encoder/llama3_2_1b.yaml",
        "examples/retrieval/cross_encoder/llama3_2_1b.yaml",
    ),
    # 2024-09-25: publisher release.
    # https://huggingface.co/meta-llama/Llama-3.2-1B
    "meta-llama/Llama-3.2-1B-Instruct": (
        "examples/llm_finetune/llama3_2/customizer_llama_3_2_1b_full_sft.yaml",
        "examples/llm_finetune/llama3_2/customizer_llama_3_2_1b_full_sft_chat.yaml",
        "examples/llm_finetune/llama3_2/customizer_llama_3_2_1b_peft.yaml",
        "examples/llm_finetune/llama3_2/customizer_llama_3_2_1b_peft_packing.yaml",
    ),
    # 2024-09-25: publisher release.
    # https://huggingface.co/meta-llama/Llama-3.2-1B
    "meta-llama/Llama-3.2-3B": (
        "examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_cp2.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_pp2.yaml",
        "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_tp2.yaml",
    ),
    # 2024-09-25: publisher release.
    # https://huggingface.co/meta-llama/Llama-3.2-1B
    "meta-llama/Llama-3.2-3B-Instruct": (
        "examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_squad.yaml",
        "examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_squad_peft.yaml",
    ),
}


def warn_deprecated_checkpoint(model_id: str | os.PathLike[str]) -> None:
    """Warn when loading a checkpoint included in the approved age-based list.

    Args:
        model_id: Exact Hugging Face checkpoint ID or local model path.
    """
    checkpoint = os.fspath(model_id)
    yaml_paths = _DEPRECATED_CHECKPOINT_YAMLS.get(checkpoint)
    if yaml_paths is None:
        return

    warnings.warn(
        f"{checkpoint} is deprecated following the 2026-10-01 age review: "
        "its release or initial Hugging Face commit predates 2024-10-01. "
        f"Associated example configs: {', '.join(yaml_paths)}",
        category=FutureWarning,
        stacklevel=3,
    )


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
