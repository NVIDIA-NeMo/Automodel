---
description: "Key features and core concepts of NeMo AutoModel for scalable LLM and VLM training with Hugging Face integration"
categories: ["concepts-architecture"]
tags: ["features", "benchmarks", "parallelism", "peft", "distributed", "recipes", "components"]
personas: ["machine-learning-engineer", "researcher", "devops"]
difficulty: "beginner"
content_type: "concept"
---

(about-key-features)=

# Key Features and Concepts

NeMo AutoModel provides GPU-accelerated, `transformers`-compatible training for LLMs and VLMs. It combines Hugging Face's model ecosystem with NVIDIA's optimized training stack, delivering high throughput without sacrificing ease of use.

## Why NeMo AutoModel?

- **Hugging Face native**: Train any model from the Hub with no checkpoint conversion -- day-0 support for new releases.
- **High performance**: Custom CUDA kernels (TransformerEngine, DeepEP, FlexAttn) deliver up to 279 TFLOPs/sec/GPU.
- **Any scale**: The same recipe runs on 1 GPU or across hundreds of nodes -- parallelism is configuration, not code.
- **Hackable**: Linear training scripts with YAML config. No hidden trainer abstractions.
- **Open source**: Apache 2.0 licensed, NVIDIA-supported, and actively maintained.

### Performance Highlights

| Model | GPUs | TFLOPs/sec/GPU | Tokens/sec/GPU | Optimizations |
|-------|-----:|---------------:|---------------:|---------------|
| DeepSeek V3 671B | 256 | 250 | 1,002 | TE + DeepEP |
| GPT-OSS 20B | 8 | 279 | 13,058 | TE + DeepEP + FlexAttn |
| Qwen3 MoE 30B | 8 | 212 | 11,842 | TE + DeepEP |

See the [full benchmark results](../performance-summary.md) for configuration details and more models.

---

## Training Workflows

NeMo AutoModel supports a range of training tasks across LLM and VLM modalities.

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Supervised Fine-Tuning (SFT)
:link: ../guides/llm/finetune
:link-type: doc
Full-parameter fine-tuning for task-specific adaptation.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` PEFT (LoRA / QLoRA)
:link: ../guides/llm/finetune
:link-type: doc
Memory-efficient fine-tuning by updating only low-rank adapter weights.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Pre-Training
:link: ../guides/llm/pretraining
:link-type: doc
Train models from scratch on large-scale datasets.
:::

:::{grid-item-card} {octicon}`dependabot;1.5em;sd-mr-1` Knowledge Distillation
:link: ../guides/llm/knowledge-distillation
:link-type: doc
Transfer knowledge from a large teacher to a smaller student model.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Tool Calling
:link: ../guides/llm/toolcalling
:link-type: doc
Fine-tune models for structured function calling with tool schemas.
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Quantization-Aware Training
:link: ../guides/quantization-aware-training
:link-type: doc
Train with quantization for deployment-ready models.
:::

::::

---

## Parallelism and Scaling

NeMo AutoModel leverages PyTorch-native parallelism strategies to scale training from a single GPU to multi-node clusters.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` FSDP2
Fully Sharded Data Parallelism with DTensor for memory-efficient distributed training. Supports Hybrid Sharding (HSDP) for multi-node.
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Pipeline Parallelism
Torch-native pipelining composable with FSDP2 and DTensor for 3D parallelism.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` FP8 Mixed Precision
FP8 training via torchao for reduced memory and higher throughput on supported models.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Multi-Node with SLURM
Add a `slurm:` section to any YAML config and launch with the `automodel` CLI. See the [Cluster guide](../launcher/cluster.md).
:::

::::

---

## Core Concepts

### Recipes

Recipes are executable Python scripts paired with YAML configuration files. Each recipe defines a complete training workflow:

1. **Load** a model and tokenizer from Hugging Face (via `_target_` in YAML)
2. **Prepare** a dataset with the appropriate collator and chat template
3. **Train** with a configurable loop (gradient accumulation, validation, logging)
4. **Checkpoint** using Distributed Checkpoint (DCP) with SafeTensors output

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train
```

Override any field from the CLI:

```bash
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
  --step_scheduler.local_batch_size 16
```

### Components

Components are modular, self-contained building blocks that recipes assemble:

| Component | Purpose |
|-----------|---------|
| `datasets/` | LLM and VLM datasets with collators, tokenization, and chat templates |
| `distributed/` | FSDP2, MegatronFSDP, tensor/sequence/pipeline parallelism |
| `_peft/` | LoRA and QLoRA implementations |
| `attention/` | Fused attention, rotary embeddings, FlexAttn |
| `checkpoint/` | DCP save/load with SafeTensors output |
| `moe/` | Mixture of Experts routing and DeepEP integration |
| `optim/` | Optimizers and LR schedulers |
| `loss/` | Cross-entropy, linear cross-entropy, KD loss |
| `launcher/` | SLURM and interactive job launch |

Each component can be used independently and has no cross-module imports.

### The `automodel` CLI

The CLI simplifies job launch across environments:

```bash
# Single-node interactive
automodel config.yaml

# Multi-node SLURM batch (add a slurm: section to the YAML)
automodel config.yaml
```

See the [Local Workstation](../launcher/local-workstation.md) and [Cluster](../launcher/cluster.md) guides.

---

## Checkpointing

NeMo AutoModel writes Distributed Checkpoints (DCP) with SafeTensors shards. Checkpoints carry partition metadata to:

- **Merge** into a single Hugging Face-compatible checkpoint for inference or sharing.
- **Reshard** when loading onto a different mesh or topology.
- **Resume** training from any checkpoint without manual intervention.

See the [Checkpointing guide](../guides/checkpointing.md) for details.

## Experiment Tracking

NeMo AutoModel integrates with MLflow and Weights & Biases for experiment tracking, metric logging, and artifact management. See the [Experiment Tracking guide](../guides/mlflow-logging.md).
