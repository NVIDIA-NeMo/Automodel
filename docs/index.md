---
description: "NeMo AutoModel is a PyTorch DTensor-native SPMD open-source training library for scalable LLM and VLM training and fine-tuning with day-0 Hugging Face model support"
categories:
  - documentation
  - home
tags:
  - training
  - fine-tuning
  - distributed
  - gpu-accelerated
  - spmd
  - dtensor
personas:
  - Machine Learning Engineers
  - Data Scientists
  - Researchers
  - DevOps Professionals
difficulty: beginner
content_type: index
---

(automodel-home)=

# NeMo AutoModel Documentation

GPU-accelerated PyTorch training for LLMs and VLMs with day-0 Hugging Face model support.

## Training Workflows

Pick your modality and task to find the right guide.

|            | SFT | PEFT (LoRA) | Pretrain | Knowledge Distillation |
|------------|-----|-------------|----------|------------------------|
| **LLM**    | [Guide](guides/llm/finetune.md) | [Guide](guides/llm/finetune.md) | [Guide](guides/llm/pretraining.md) | [Guide](guides/llm/knowledge-distillation.md) |
| **VLM**    | [Guide](guides/overview.md) | [Guide](guides/omni/gemma3-3n.md) | -- | -- |

### How Do I Scale?

Every workflow above supports all three scales -- just change how you launch.

| Scale | Launch Method | Guide |
|-------|---------------|-------|
| **Single GPU** | `python examples/llm_finetune/finetune.py -c config.yaml` | [Local Workstation](launcher/local-workstation.md) |
| **Multi-GPU** | `torchrun --nproc-per-node=N examples/llm_finetune/finetune.py -c config.yaml` | [Local Workstation](launcher/local-workstation.md) |
| **Multi-Node** | Add a `slurm:` section to your YAML config, then `automodel finetune llm -c config.yaml` | [Cluster Deployment](launcher/cluster.md) |

## Performance

Training throughput on NVIDIA GPUs with optimized kernels for Hugging Face models.

| Model | GPUs | TFLOPs/sec/GPU | Tokens/sec/GPU | Optimizations |
|-------|-----:|---------------:|---------------:|---------------|
| DeepSeek V3 671B | 256 | 250 | 1,002 | TE + DeepEP |
| GPT-OSS 20B | 8 | 279 | 13,058 | TE + DeepEP + FlexAttn |
| Qwen3 MoE 30B | 8 | 212 | 11,842 | TE + DeepEP |

See the [full benchmark results](performance-summary.md) for configuration details and more models.

## Get Started

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: guides/installation
:link-type: doc
Install via PyPI, Docker, or from source.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: guides/configuration
:link-type: doc
YAML-driven recipes with CLI overrides.
:::

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` Hugging Face Compatibility
:link: guides/huggingface-api-compatibility
:link-type: doc
Day-0 support for any model on the Hub.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Model Coverage
:link: model-coverage/overview
:link-type: doc
Supported LLM and VLM families.
:::

::::

## Advanced Topics

Optimize training with advanced parallelism, precision, and checkpointing strategies.

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Pipeline Parallelism
:link: guides/pipelining
:link-type: doc
Torch-native pipelining composable with FSDP2 and DTensor.
+++
{bdg-secondary}`3d-parallelism`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` FP8 Training
:link: guides/fp8-training
:link-type: doc
Mixed-precision FP8 training with torchao for supported models.
+++
{bdg-secondary}`fp8` {bdg-secondary}`mixed-precision`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Checkpointing
:link: guides/checkpointing
:link-type: doc
Distributed checkpoints with SafeTensors output.
+++
{bdg-secondary}`dcp` {bdg-secondary}`safetensors`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Gradient Checkpointing
:link: guides/gradient-checkpointing
:link-type: doc
Trade compute for memory with activation checkpointing.
+++
{bdg-secondary}`memory-efficiency`
:::

:::{grid-item-card} {octicon}`meter;1.5em;sd-mr-1` Quantization-Aware Training
:link: guides/quantization-aware-training
:link-type: doc
Train with quantization for deployment-ready models.
+++
{bdg-secondary}`qat`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` MLflow Logging
:link: guides/mlflow-logging
:link-type: doc
Track experiments and metrics with MLflow integration.
+++
{bdg-secondary}`experiment-tracking`
:::

::::

## Launch & Infrastructure

Run training on local workstations or multi-node clusters.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`device-desktop;1.5em;sd-mr-1` Local Workstation
:link: launcher/local-workstation
:link-type: doc
Interactive single-node and multi-GPU training.
+++
{bdg-secondary}`torchrun` {bdg-secondary}`interactive`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Cluster (SLURM)
:link: launcher/cluster
:link-type: doc
Multi-node training with SLURM and the `automodel` CLI.
+++
{bdg-secondary}`slurm` {bdg-secondary}`multi-node`
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: Get Started
repository-structure.md
guides/installation.md
guides/configuration.md
guides/huggingface-api-compatibility.md
launcher/local-workstation.md
launcher/cluster.md
::::

::::{toctree}
:hidden:
:caption: Announcements
Accelerating Large-Scale Mixture-of-Experts Training in PyTorch with NeMo Automodel <https://github.com/NVIDIA-NeMo/Automodel/discussions/777>
Challenges in Enabling PyTorch Native Pipeline Parallelism for Hugging Face Transformer Models <https://github.com/NVIDIA-NeMo/Automodel/discussions/589>
Google Gemma 3n: Efficient Multimodal Fine-tuning Made Simple <https://github.com/NVIDIA-NeMo/Automodel/discussions/494>
Fine-tune Hugging Face Models Instantly with Day-0 Support with NVIDIA NeMo AutoModel <https://github.com/NVIDIA-NeMo/Automodel/discussions/477>
::::

::::{toctree}
:hidden:
:caption: Performance

performance-summary.md
::::

::::{toctree}
:hidden:
:caption: Recipes & E2E Examples
guides/overview.md
guides/llm/finetune.md
guides/llm/toolcalling.md
guides/llm/pretraining.md
guides/llm/nanogpt-pretraining.md
guides/llm/sequence-classification.md
guides/omni/gemma3-3n.md
guides/vlm/qwen3_5.md
guides/quantization-aware-training.md
guides/llm/databricks.md
::::

::::{toctree}
:hidden:
:caption: Model Coverage
model-coverage/overview.md
model-coverage/llm.md
model-coverage/vlm.md
::::

::::{toctree}
:hidden:
:caption: Datasets

guides/dataset-overview.md
guides/llm/dataset.md
guides/llm/retrieval-dataset.md
guides/llm/column-mapped-text-instruction-dataset.md
guides/llm/column-mapped-text-instruction-iterable-dataset.md
guides/vlm/dataset.md
::::

::::{toctree}
:hidden:
:caption: Development
guides/checkpointing.md
guides/gradient-checkpointing.md
guides/pipelining.md
guides/llm/knowledge-distillation.md
guides/fp8-training.md
guides/mlflow-logging.md

apidocs/index.rst
::::
