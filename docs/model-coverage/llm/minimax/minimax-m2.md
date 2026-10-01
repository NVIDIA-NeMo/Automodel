---
title: "MiniMaxAI/MiniMax-M2.1"
description: "Use MiniMax-M2.1 with NeMo AutoModel for language model fine-tuning, with documented checkpoints, runnable recipes, setup guidance, and model reference details."
slug: model-coverage/large-language-models/minimax/MiniMax-M2.1
---

[MiniMax-M2](https://huggingface.co/MiniMaxAI) is MiniMax's large Mixture-of-Experts language model with linear attention for efficient long-context inference.

## Quick Start

Follow the [installation instructions](/get-started/installation). Allocate 8 nodes with 8 GPUs each; run on every node with `NODE_RANK` and `MASTER_ADDR` set. Complete the [recipe setup](#recipe-setup) before launching.

```bash
uv run torchrun --nnodes 8 --nproc-per-node 8 \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29500 \
  -m nemo_automodel.cli.app examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```

## Choose a Workflow

| Goal | Start Here |
| --- | --- |
| Fine-tune MiniMaxAI/MiniMax-M2.1 | [minimax_m2.1_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml). 8 nodes, 8 GPUs per node; see [Recipe Setup](#recipe-setup). |
| SFT — MiniMax-M2.5 on HellaSwag with pipeline parallelism | [minimax_m2.5_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml) |
| SFT — MiniMax-M2.7 on HellaSwag with pipeline parallelism | [minimax_m2.7_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_pp.yaml) |

## Model Context

| Property | Value |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `MiniMaxM2ForCausalLM` |
| **Parameters** | 229B total / 10B active |
| **HF Org** | [MiniMaxAI](https://huggingface.co/MiniMaxAI) |
| Decoder Layers | 62 |
| Hidden Size | 3,072 |
| Attention | 48 attention heads / 8 key-value heads |
| Context Length | 196,608 tokens (checkpoint configuration) |
| Vocabulary Size | 200,064 |
| Experts | 256 routed / 8 selected per token |

### Architecture

- `MiniMaxM2ForCausalLM`

### Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.mdx).

### Setup and Additional Examples

**1. Clone and install from source** ([full instructions](/get-started/installation)):

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
uv sync --locked --all-groups --all-extras
```

<Note>
This recipe was validated on **8 nodes × 8 GPUs (64 H100s)**. See the [Launcher Guide](../../../launcher/slurm.mdx) for multi-node setup.

</Note>

**2. Run the recipe** from inside the repo:

```bash
uv run automodel --nproc-per-node=8 examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```

<Accordion title="Run with Docker">
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.06.00
```

**2.** Navigate to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```
</Accordion>

See the [Installation Guide](../../../guides/installation.mdx) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.mdx).

### Recipe Setup

Read [minimax_m2.1_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml) for checkpoint access, data paths, and backend dependencies. Use the [Slurm launcher guide](/job-launchers/slurm-cluster) to prepare the distributed allocation.

## Available Models

- **MiniMax-M2.1**
- **MiniMax-M2.5**
- **MiniMax-M2.7**

| Model | HF ID |
|---|---|
| MiniMax M2.1 | [`MiniMaxAI/MiniMax-M2.1`](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) |
| MiniMax M2.5 | [`MiniMaxAI/MiniMax-M2.5`](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) |
| MiniMax M2.7 | [`MiniMaxAI/MiniMax-M2.7`](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) |

## Related Resources

- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)

- [Checkpoint Configuration](https://huggingface.co/MiniMaxAI/MiniMax-M2.1/blob/cd97f59135f37b2a6bf09356e485d5e4aeb7dc9c/config.json)
