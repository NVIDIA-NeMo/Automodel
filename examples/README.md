# Examples

> **New?** Start with one of these:
> - **LLM**: `llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml` -- fine-tune Qwen3-0.6B on 1 GPU (8 GB VRAM)
> - **VLM**: `vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml` -- fine-tune Gemma-3 VL with LoRA

Run any example:

```bash
automodel finetune llm -c examples/llm_finetune/<config>
automodel finetune vlm -c examples/vlm_finetune/<config>
```

## LLM Fine-Tuning

### Beginner (1 GPU, small model)

| Example | Model | Task | Method | GPUs |
|---------|-------|------|--------|------|
| `qwen/qwen3_0p6b_hellaswag.yaml` | Qwen3-0.6B | SFT | Full | 1 |
| `qwen/qwen3_0p6b_hellaswag_peft.yaml` | Qwen3-0.6B | PEFT | LoRA | 1 |
| `gemma/gemma_3_270m_squad.yaml` | Gemma-3-270M | SFT | Full | 1 |
| `gemma/gemma_3_270m_squad_peft.yaml` | Gemma-3-270M | PEFT | LoRA | 1 |
| `phi/phi_2_squad.yaml` | Phi-2 | SFT | Full | 1 |
| `olmo/olmo_2_0425_1b_instruct_squad.yaml` | OLMo-2-1B | SFT | Full | 1 |
| `nemotron_flash/nemotron_flash_1b_squad.yaml` | Nemotron-Flash-1B | SFT | Full | 1 |

### Intermediate (multi-GPU, 7-9B models)

| Example | Model | Task | Method | GPUs |
|---------|-------|------|--------|------|
| `qwen/qwen2_5_7b_squad.yaml` | Qwen2.5-7B | SFT | Full | 8 |
| `qwen/qwen2_5_7b_squad_peft.yaml` | Qwen2.5-7B | PEFT | LoRA | 1-8 |
| `mistral/mistral_7b_squad.yaml` | Mistral-7B | SFT | Full | 8 |
| `gemma/gemma_2_9b_it_squad.yaml` | Gemma-2-9B-IT | SFT | Full | 8 |
| `llama3_1/llama3_1_8b_squad_qlora.yaml` | Llama-3.1-8B | PEFT | QLoRA | 1 |
| `falcon/falcon3_7b_instruct_squad.yaml` | Falcon3-7B | SFT | Full | 8 |
| `seed/seed_coder_8b_instruct_squad.yaml` | Seed-Coder-8B | SFT | Full | 8 |

### Advanced (large models, multi-node, MoE)

| Example | Model | Task | Method | GPUs |
|---------|-------|------|--------|------|
| `llama3_3/llama_3_3_70b_instruct_squad_peft.yaml` | Llama-3.3-70B | PEFT | LoRA | 8+ |
| `qwen/qwq_32b_squad.yaml` | QwQ-32B | SFT | Full | 8+ |
| `qwen/qwen3_moe_30b_te_deepep.yaml` | Qwen3-MoE-30B | SFT | Full (TE+DeepEP) | 8+ |
| `deepseek_v32/deepseek_v32_hellaswag_pp.yaml` | DeepSeek-V3.2 | SFT | Pipeline Parallel | multi-node |
| `gpt_oss/gpt_oss_120b.yaml` | GPT-OSS-120B | SFT | Full (TE+DeepEP) | multi-node |
| `minimax_m2/minimax_m2.5_hellaswag_pp.yaml` | MiniMax-M2.5 | SFT | Pipeline Parallel | multi-node |
| `stepfun/step_3.5_flash_hellaswag_pp.yaml` | Step-3.5-Flash | SFT | Pipeline Parallel | multi-node |

### FP8 Training

| Example | Model | GPUs |
|---------|-------|------|
| `qwen/qwen2_5_7b_hellaswag_fp8.yaml` | Qwen2.5-7B | 8 |
| `llama3_1/llama3_1_8b_hellaswag_fp8.yaml` | Llama-3.1-8B | 8 |
| `mistral/mistral_7b_hellaswag_fp8.yaml` | Mistral-7B | 8 |
| `phi/phi_4_hellaswag_fp8.yaml` | Phi-4 | 8 |

## VLM Fine-Tuning

| Example | Model | Task | Method | GPUs |
|---------|-------|------|--------|------|
| `gemma3/gemma3_vl_4b_cord_v2_peft.yaml` | Gemma-3-4B VL | PEFT | LoRA | 1 |
| `gemma3/gemma3_vl_4b_cord_v2.yaml` | Gemma-3-4B VL | SFT | Full | 8 |
| `gemma3n/gemma3n_vl_4b_medpix_peft.yaml` | Gemma-3n-4B VL | PEFT | LoRA | 1 |
| `qwen2_5/qwen2_5_vl_3b_rdr.yaml` | Qwen2.5-VL-3B | SFT | Full | 8 |
| `qwen3/qwen3_vl_4b_instruct_rdr.yaml` | Qwen3-VL-4B | SFT | Full | 8 |
| `qwen3/qwen3_vl_moe_235b.yaml` | Qwen3-VL-235B MoE | SFT | Full | multi-node |
| `kimi/kimi2vl_cordv2.yaml` | Kimi-2-VL | SFT | Full | 8 |
| `phi4/phi4_mm_cv17.yaml` | Phi-4-MM | SFT | Full | 8 |

## Other Tasks

| Example | Task | Model |
|---------|------|-------|
| `llm_pretrain/nanogpt_pretrain.yaml` | Pretraining (nano-GPT) | Custom GPT-2 |
| `llm_pretrain/deepseekv3_pretrain.yaml` | Pretraining | DeepSeek-V3 |
| `llm_kd/llama3_2/llama3_2_1b_kd.yaml` | Knowledge Distillation | Llama-3.2-1B |
| `llm_seq_cls/glue/mrpc_roberta_lora.yaml` | Sequence Classification | RoBERTa |
| `gemma/functiongemma_xlam.yaml` | Tool Calling | FunctionGemma |
| `biencoder/llama3_2_1b_biencoder.yaml` | Bi-Encoder / Retrieval | Llama-3.2-1B |
