
# Diffusion Language Model (dLLM) Fine-Tuning and Generation with NeMo AutoModel

## Introduction

Diffusion language models (dLLMs) generate text by iteratively denoising masked tokens, rather than generating one token at a time left-to-right like autoregressive (AR) models. Starting from a sequence of `[MASK]` tokens, the model progressively unmasks the most confident positions over multiple denoising steps until the full response is revealed.

This approach enables **parallel token generation** and **bidirectional attention**, which gives the model more context for each prediction compared to AR models.

NeMo AutoModel currently supports the following dLLM model families:

- **LLaDA / LLaDA2 (MDLM)** — Bidirectional masked diffusion. The model receives corrupted tokens and predicts the clean token at each masked position.
- **Nemotron-Labs-Diffusion (Hybrid)** — Combines diffusion with an autoregressive loss. During training, the model processes clean tokens plus a `masked_indices` sidecar and learns both a diffusion objective and an AR objective simultaneously.
- **DFlash** — Speculative block diffusion. A small draft model proposes tokens for a block conditioned on frozen target LM hidden states; a decay-weighted loss trains it to predict the target's distribution (see [DFlash paper](https://arxiv.org/abs/2602.17270)).

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
| **1. Install** | [Install NeMo AutoModel](#install-nemo-automodel) | Install the package using pip or Docker |
| **2. Configure** | [Configure Your Training Recipe](#configure-your-training-recipe) | Write a YAML config specifying model, data, dLLM mode, and training settings |
| **3. Train** | [Fine-Tune the Model](#fine-tune-the-model) | Launch training with `torchrun` |
| **4. Generate** | [Run Inference](#run-inference) | Generate text from a fine-tuned checkpoint |

### Supported Models

| Model Family | dLLM Mode | Loss | Inference | Example Config |
|---|---|---|---|---|
| LLaDA / LLaDA2 | `mdlm` | MDLM cross-entropy | Block-by-block, full-forward (no KV cache) | [llada2_sft.yaml](../../../examples/dllm_sft/llada2_sft.yaml) |
| Nemotron-Labs-Diffusion | `hybrid` | Diffusion + AR (alpha-weighted) | Block diffusion with KV cache | [nemotron_labs_diffusion_sft.yaml](../../../examples/dllm_sft/nemotron_labs_diffusion_sft.yaml) |
| DFlash | `dflash` | Decay-weighted cross-entropy (Eq. 4) | Speculative block decoding (draft + target) | [dflash_sft.yaml](../../../examples/dllm_sft/dflash_sft.yaml) |

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
| `hybrid` | `HybridStrategy` | Nemotron-Labs-Diffusion-style: model receives clean tokens + `masked_indices`, combined diffusion + AR loss |
| `dflash` | `DFlashStrategy` | DFlash: frozen target LM provides hidden states; draft model trained with decay-weighted loss |

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

### Nemotron-Labs-Diffusion Configuration

See [nemotron_labs_diffusion_sft.yaml](../../../examples/dllm_sft/nemotron_labs_diffusion_sft.yaml) for the full working config. The key dLLM-specific sections are:

```yaml
model:
  pretrained_model_name_or_path: nvidia/Nemotron-Labs-Diffusion-8B-Base
  torch_dtype: float32          # Master-weight dtype. Use `float32` for an fp32 master copy or `bfloat16` for bf16.
  trust_remote_code: true
  dlm_paradigm: block_diff       # required for SFT: HF default "bidirectional" is the inference mode
  block_size: 32

dllm:
  mode: hybrid
  mask_token_id: 100              # Nemotron-Labs-Diffusion mask token
  eps: 0.001
  ar_loss_alpha: 0.3              # weight on the diffusion branch (AR branch is unweighted)
  pad_seq_len_divisible: 1024

dataset:
  unshifted: true
```

### Key dLLM Config Fields

| Field | Description |
|-------|-------------|
| `dllm.mode` | Training strategy (`mdlm`, `hybrid`, or `dflash`) |
| `dllm.mask_token_id` | Token ID used for masking (`126336` for LLaDA, `156895` for LLaDA2.1, `100` for Nemotron-Labs-Diffusion) |
| `dllm.eps` | Minimum corruption ratio to avoid zero-corruption samples |
| `dllm.block_size` | When set, use block-wise corruption (otherwise uniform). Hybrid mode only. |
| `dllm.half_life_ratio` | Half-life ratio for block-wise corruption (defaults to 0.25 when unset). Hybrid mode only. |
| `dllm.ar_loss_alpha` | Weight applied to the diffusion branch in the hybrid loss. Hybrid mode only. |
| `dataset.unshifted` | Must be `true` for dLLM — disables the autoregressive input/target shift |

### DFlash Configuration

DFlash trains a small draft model to predict tokens conditioned on a frozen causal target LM.
Only the draft model's weights are updated; the target LM is loaded once and kept frozen.

See [dflash_sft.yaml](../../../examples/dllm_sft/dflash_sft.yaml) for the full working config.
The key DFlash-specific sections are:

```yaml
model:                                          # Draft model
  _target_: transformers.AutoModel.from_pretrained
  pretrained_model_name_or_path: z-lab/Qwen3-4B-DFlash-b16
  trust_remote_code: true
  torch_dtype: bfloat16

dllm:
  mode: dflash
  mask_token_id: null                           # Resolved automatically from target tokenizer
  eps: 0.001

dflash:
  target_model_id: Qwen/Qwen3-4B               # Frozen causal LM
  target_torch_dtype: bfloat16
  block_size: 0                                 # 0 reads from draft model config
  loss_decay_gamma: 0.0                         # 0 uses paper defaults (γ=7 for block_size=16)
  num_blocks_per_sample: 512                    # Paper default (Appendix A.1)
  attention_backend: flex_attention             # required for N > ~64; sdpa OOMs
  overlap_anchors: true                         # paper samples anchors independently
```

| Field | Description |
|-------|-------------|
| `dflash.target_model_id` | Hub ID of the frozen causal LM that conditions the draft |
| `dflash.block_size` | Tokens per draft block; `0` reads from draft model config |
| `dflash.loss_decay_gamma` | Decay γ for Eq. 4; `0` uses paper defaults (7/5/4 for block sizes 16/10/8) |
| `dflash.num_blocks_per_sample` | Number of anchor blocks processed per sequence per step (paper default: 512, Appendix A.1) |
| `dflash.attention_backend` | `flex_attention` (sparse, scales to 512 anchors) or `sdpa` (dense, OOMs above ~64). Default `sdpa` for backward compat — set to `flex_attention` for production runs |
| `dflash.overlap_anchors` | `true` (paper, independent sampling) or `false` (non-overlapping stars-and-bars, caps at `seq_len // block_size`) |

#### Prepare DFlash Training Data

The paper trains on responses **regenerated by the target model** (§5.1):
*"Instead of directly using the original dataset, we construct our training set
with the responses generated by the target model for better target alignment."*
Skipping this step trains the draft on a different output distribution than the
target produces at inference, which directly reduces acceptance length.

The existing `nemo_automodel.components.speculative.regenerate` script handles
this. Stand up an SGLang server hosting the target, then re-roll the assistant
turns:

```bash
# 1. Serve the target model on the local node (default port 30000)
python -m sglang.launch_server \
    --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --served-model-name nemotron-30b \
    --trust-remote-code

# 2. Regenerate the dataset's assistant turns through the target (separate shell)
python -m nemo_automodel.components.speculative.regenerate \
    --input-data nvidia/Nemotron-Post-Training-Dataset-v2 \
    --output-dir /data/dflash-train-regen \
    --model nemotron-30b \
    --temperature 0.0 \
    --shard-size 1000 \
    --concurrency 32 \
    --resume
```

Then point the recipe's `dataset.path_or_dataset_id` at the regenerated
parquet shards (`/data/dflash-train-regen`) instead of the raw HF dataset.

## Fine-Tune the Model

### Fine-Tune LLaDA2

```bash
torchrun --nproc-per-node=8 \
    examples/dllm_sft/finetune.py \
    -c examples/dllm_sft/llada2_sft.yaml
```

### Fine-Tune with DFlash

```bash
torchrun --nproc-per-node=8 \
    examples/dllm_sft/finetune.py \
    -c examples/dllm_sft/dflash_sft.yaml
```

### Fine-Tune Nemotron-Labs-Diffusion

```bash
torchrun --nproc-per-node=8 \
    nemo_automodel/recipes/dllm/train_ft.py \
    -c examples/dllm_sft/nemotron_labs_diffusion_sft.yaml
```

## Run Inference

The generation script ([`generate.py`](../../../examples/dllm_generate/generate.py)) supports chat, raw, and infilling modes. Pick the sampler that matches the trained family with `--sampler {llada,nemotron}`.

`--checkpoint` accepts any of: a path to a `consolidated/` directory, a step directory (`.../epoch_0_step_499`), or the top-level checkpoint dir (the script will follow `LATEST/model/consolidated/`).

### Generate with LLaDA

```bash
python examples/dllm_generate/generate.py \
    --checkpoint <path> \
    --prompt "Explain what a neural network is." \
    --sampler llada
```

### Nemotron-Labs-Diffusion Generation

```bash
python examples/dllm_generate/generate.py \
    --checkpoint <path> \
    --prompt "What is 2+2?" \
    --sampler nemotron
```

The `nemotron` path internally invokes the model's built-in block-diffusion `model.generate(...)` (with the AR-seed mechanism), while the `llada` path uses the standalone `DLLMSampler.sample(...)`.
