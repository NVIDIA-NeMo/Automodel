---
title: "MiniMaxAI/MiniMax-M2.1"
description: "Fine-tune MiniMaxAI/MiniMax-M2.1 on HellaSwag with NeMo AutoModel using pipeline and expert parallelism."
slug: model-coverage/large-language-models/minimax/MiniMax-M2.1
---

[MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) is a mixture-of-experts (MoE) language model for text generation. NeMo AutoModel provides full fine-tuning recipes on HellaSwag using pipeline parallelism (PP) and expert parallelism (EP).

## Quick Start

After [installation](/get-started/installation) and [recipe setup](#recipe-setup), run this HellaSwag full fine-tuning example from the repository root on each of 8 nodes with 8 GPUs. Set `NODE_RANK` and `MASTER_ADDR` as described below.

```bash
uv run torchrun --nnodes 8 --nproc-per-node 8 \
  --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29500 \
  -m nemo_automodel.cli.app examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```

## Choose a Workflow

| Workflow | Example Setup | Recipe |
| --- | --- | --- |
| Full Fine-Tuning | HellaSwag; 8 nodes; PP2; EP32 | [View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml) |
| Full Fine-Tuning (MiniMax-M2.5) | HellaSwag; 8 nodes; PP2; EP32 | [View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml) |
| Full Fine-Tuning (MiniMax-M2.7) | HellaSwag; 8 nodes; PP2; EP16 | [View YAML](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_pp.yaml) |

## Model Context

MiniMax-M2.1 uses 256 routed experts, selecting 8 per token. The dimensions and context limit below describe the checkpoint configuration.

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

### Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.mdx).

### Set Up the Environment

1. Clone the repository and install from source ([full instructions](/get-started/installation)):

   ```bash
   git clone https://github.com/NVIDIA-NeMo/Automodel.git
   cd Automodel
   uv sync --locked --all-groups --all-extras
   ```

2. Complete [Recipe Setup](#recipe-setup), then run the [Quick Start](#quick-start) command on every node.

For container setup, see the [Installation Guide](/get-started/installation). Use the [Slurm Launcher Guide](/job-launchers/slurm-cluster) to prepare a multi-node environment.

### Recipe Setup

The [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml) configures PP2 and EP32, with 8 nodes in its CI settings. The Quick Start uses 8 GPUs per node (64 processes); this is the example topology, not a measured hardware minimum.

The Quick Start recipe sets `checkpoint.enabled: false`, so it does not save trained weights.

Use the same checkout and environment on every node. Set `NODE_RANK` to a unique value from `0` through `7`, and set `MASTER_ADDR` on all nodes to the address of node `0`. Ensure that address is reachable on port `29500`.

The recipe loads `MiniMaxAI/MiniMax-M2.1` and `rowan/hellaswag` from Hugging Face and selects Transformer Engine and HybridEP backends. Ensure these backends are available in the environment on every node before launching.

## Available Models

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
