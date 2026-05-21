
# Diffusion Language Model (dLLM) Fine-Tuning and Generation with NeMo AutoModel

## Introduction

Diffusion language models (dLLMs) generate text by iteratively denoising masked tokens, rather than generating one token at a time left-to-right like autoregressive (AR) models. Starting from a sequence of `[MASK]` tokens, the model progressively unmasks the most confident positions over multiple denoising steps until the full response is revealed.

This approach enables **parallel token generation** and **bidirectional attention**, which gives the model more context for each prediction compared to AR models.

NeMo AutoModel currently supports the following dLLM model family:

- **LLaDA (MDLM)** — Bidirectional masked diffusion. The model receives corrupted tokens and predicts the clean token at each masked position.

### Workflow Overview

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Install  │--->│ 2. Configure │--->│   3. Train   │--->│ 4. Generate  │
│              │    │    YAML      │    │              │    │              │
│ pip install  │    │  Recipe +    │    │  torchrun    │    │  Run dLLM    │
│ or Docker    │    │  dLLM config │    │              │    │  inference   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

| Step | Section | What You Do |
|------|---------|-------------|
| **1. Install** | [Install NeMo AutoModel](#install-nemo-automodel) | Install the package via pip or Docker |
| **2. Configure** | [Configure Your Training Recipe](#configure-your-training-recipe) | Write a YAML config specifying model, data, dLLM mode, and training settings |
| **3. Train** | [Fine-Tune the Model](#fine-tune-the-model) | Launch training with `torchrun` |
| **4. Generate** | [Generation / Inference](#generation--inference) | Generate text from a fine-tuned checkpoint |

### Supported Models

| Model Family | dLLM Mode | Loss | Inference | Example Config |
|---|---|---|---|---|
| LLaDA | `mdlm` | MDLM cross-entropy | Block-by-block, full-forward (no KV cache) | [llada_sft.yaml](../../../examples/dllm_sft/llada_sft.yaml) |

## Install NeMo AutoModel

```bash
pip3 install nemo-automodel
```

Alternatively, use the pre-built Docker container:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02.00
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:26.02.00
```

For the full set of installation methods, see the [installation guide](../installation.md).

## Configure Your Training Recipe

dLLM fine-tuning is driven by:

1. A **recipe script** ([`train_ft.py`](../../../nemo_automodel/recipes/dllm/train_ft.py)) — orchestrates the training loop with dLLM-specific corruption, loss, and batch handling.
2. A **YAML configuration file** — specifies the model, data, optimizer, dLLM-specific settings, and distributed training strategy.

The recipe uses a **strategy pattern** to handle differences between model families. The `dllm.mode` field in the YAML selects the strategy:

| Mode | Strategy | Description |
|------|----------|-------------|
| `mdlm` | `MDLMStrategy` | LLaDA-style: model receives corrupted tokens, MDLM cross-entropy loss |

### LLaDA Configuration

See [llada_sft.yaml](../../../examples/dllm_sft/llada_sft.yaml) for the full working config. The key dLLM-specific sections are:

```yaml
model:
  pretrained_model_name_or_path: GSAI-ML/LLaDA-8B-Base
  torch_dtype: float32
  trust_remote_code: true

dllm:
  mode: mdlm
  mask_token_id: 126336       # LLaDA mask token
  eps: 0.001                  # Minimum corruption ratio

dataset:
  unshifted: true             # Required for dLLM training
```

### Key dLLM Config Fields

| Field | Description |
|-------|-------------|
| `dllm.mode` | Training strategy (`mdlm`) |
| `dllm.mask_token_id` | Token ID used for masking (`126336` for LLaDA) |
| `dllm.eps` | Minimum corruption ratio to avoid zero-corruption samples |
| `dataset.unshifted` | Must be `true` for dLLM — disables the autoregressive input/target shift |

## Fine-Tune the Model

```bash
torchrun --nproc-per-node=8 \
    nemo_automodel/recipes/dllm/train_ft.py \
    -c examples/dllm_sft/llada_sft.yaml
```

## Generation / Inference

The generation script ([`generate.py`](../../../examples/dllm_generate/generate.py)) supports chat, raw, and infilling modes for LLaDA checkpoints.

### LLaDA Generation

```bash
python examples/dllm_generate/generate.py \
    --checkpoint <path> \
    --prompt "Explain what a neural network is."
```

### Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--steps` | Number of denoising steps | 128 |
| `--max_new_tokens` | Maximum tokens to generate | 128 |
| `--block_size` | Tokens per denoising block | 32 |
| `--temperature` | Gumbel noise temperature (0 = greedy) | 0.0 |
| `--remasking` | Confidence scoring strategy for selecting which positions to unmask | `low_confidence` |
