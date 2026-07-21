# Fine-Tuning DiffusionGemma with NeMo AutoModel

## Introduction

**DiffusionGemma** is a block-diffusion language model. Unlike an autoregressive (AR)
model that generates one token at a time left-to-right, a block-diffusion model fills in
a block of response tokens (a "canvas") by iteratively denoising the canvas. The canvas
starts as noise and is refined over several passes, conditioned on the prompt.

This guide covers **Supervised Fine-Tuning (SFT)** of the DiffusionGemma **26B-A4B** model,
which is a Mixture-of-Experts (MoE) model with 26B total and approximately 4B active parameters,
using NeMo AutoModel with both full fine-tuning and Low-Rank Adaptation (LoRA).

The released checkpoint is available on the Hugging Face Hub:
[`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it).

### Workflow Overview

| Step | What You Do |
| :--- | :--- |
| 1. Install | Install NeMo AutoModel using pip or a container |
| 2. Configure | Select an example YAML configuration file for full SFT or LoRA, and specify your dataset |
| 3. Train | Launch training with `torchrun` on eight GPUs |
| 4. Inspect | Read the training and diffusion loss curves |

## Model Overview

DiffusionGemma combines a causal encoder and a bidirectional decoder:

- **Encoder**: Reads the clean prompt and response sequence with causal attention.
- **Decoder**: Denoises the canvas (the response region) with bidirectional,
  block-causal attention, predicting the clean token at every canvas position.

The `DiffusionGemmaSFTRecipe` handles the following key training mechanics:

- **Uniform-random corruption**: For each example, a corruption level
  $t \sim U(\text{eps}, 1)$ is sampled, and supervised canvas positions are independently
  replaced with uniform random vocabulary tokens (no `[MASK]` token is used). The model
  learns to recover the clean token at every supervised canvas position.
- **Self-conditioning**: The decoder optionally conditions on its own previous prediction,
  which is mixed in per example during training.
- **Frozen router**: The MoE router is kept frozen during SFT, and experts and dense layers
  are trained (full SFT) or adapted using LoRA.
- **Single-turn SFT**: The loss supervises the final response turn, and multi-turn histories
  are masked.

The recipe runs with **Fully Sharded Data Parallel 2 (FSDP2) and expert parallelism (EP=8)**
and mixed precision (FP32 master weights and BF16 compute) with a canvas length of 256.

## Launch Training

DiffusionGemma SFT runs on a single eight-GPU node (EP=8). Two example configuration files
are provided in the `examples/dllm_sft/` directory:

| Configuration File | Description |
| :--- | :--- |
| [`diffusion_gemma_sft.yaml`](../../../examples/dllm_sft/diffusion_gemma_sft.yaml) | Full fine-tuning on the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset |
| [`diffusion_gemma_lora.yaml`](../../../examples/dllm_sft/diffusion_gemma_lora.yaml) | LoRA fine-tuning |

Both configurations automatically pull the checkpoint from the Hugging Face Hub
(`google/diffusiongemma-26B-A4B-it`). Because the GSM8K dataset is consumed in the OpenAI
chat-messages format, you must generate the JSONL file (`./gsm8k_chat_train.jsonl`) before
launching training:

```bash
python examples/dllm_sft/prep_gsm8k.py
```

**Full SFT:**

```bash
torchrun --standalone --nproc-per-node=8 \
    examples/dllm_sft/finetune.py \
    -c examples/dllm_sft/diffusion_gemma_sft.yaml
```

**LoRA:**

```bash
torchrun --standalone --nproc-per-node=8 \
    examples/dllm_sft/finetune.py \
    -c examples/dllm_sft/diffusion_gemma_lora.yaml
```


## Training Results

The following sections show the SFT and LoRA training curves on the GSM8K dataset for the
first 200 steps.

**SFT**

![DiffusionGemma SFT training curves](./diffusiongemma_sft.png)

**LoRA**

![DiffusionGemma LoRA training curves](./diffusiongemma_lora.png)
