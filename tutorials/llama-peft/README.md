# Llama LoRA Fine-Tuning

This directory contains a **parameter-efficient fine-tuning (PEFT)** tutorial for training **LoRA adapters** on **Llama-3.2-1B** with NeMo AutoModel.

The tutorial uses the **SQuAD** dataset and is designed as a **single-GPU friendly** workflow for task adaptation when full-parameter supervised fine-tuning is not necessary.

## Files

| File | Description |
| --- | --- |
| [`finetune.ipynb`](./finetune.ipynb) | Step-by-step notebook covering PEFT concepts, dataset inspection, launch, checkpoint loading, and SFT vs PEFT tradeoffs. |
| [`llama_peft_config.yaml`](./llama_peft_config.yaml) | NeMo AutoModel config for LoRA fine-tuning `meta-llama/Llama-3.2-1B` on SQuAD. |

## Prerequisites

- **NeMo AutoModel** installed from the repository root.
- **Hugging Face access** to `meta-llama/Llama-3.2-1B`.
- **Hugging Face authentication** configured with `huggingface-cli login` or `HF_TOKEN`.
- **NVIDIA GPU** with enough memory for Llama-3.2-1B LoRA fine-tuning.

## Run

From this directory:

```bash
automodel llama_peft_config.yaml
```

The config trains LoRA adapters with `dim: 8`, `alpha: 32`, and `match_all_linear: true`, then saves consolidated safetensors checkpoints.

## When To Use PEFT

Use this tutorial when you want **lower memory usage**, **faster iteration**, or **task-specific adapters** without updating every parameter in the base model. For maximum task quality with sufficient GPU resources, full SFT may still be a better fit.
