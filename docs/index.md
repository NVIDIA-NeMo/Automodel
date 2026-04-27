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

PyTorch-native training that scales from 1 GPU to thousands with a single config change. Load any Hugging Face model, point at your data, and start training -- no checkpoint conversion, no boilerplate.
**Quick links:** [🤗 HF Compatible](guides/huggingface-api-compatibility.md) | [🚀 Performance](performance-summary.md) | [📐 Scalability](about/key-features.md) | [🎯 SFT & PEFT](guides/llm/finetune.md) | [🎨 Diffusion](guides/diffusion/finetune.md) | [👁️ VLM](guides/vlm/gemma4.md)

::::{grid} 2 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About
:link: about/index
:link-type: doc
Overview of NeMo AutoModel and its capabilities.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Key Features
:link: about/key-features
:link-type: doc
Supported workflows, parallelism, recipes, and benchmarks.
:::

:::{grid-item-card} {octicon}`hubot;1.5em;sd-mr-1` 🤗 HF Integration
:link: guides/huggingface-api-compatibility
:link-type: doc
A `transformers`-compatible library with accelerated model implementations.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Model Coverage
:link: model-coverage/overview
:link-type: doc
Built on `transformers` for day-0 model support and OOTB compatibility.
:::

::::

## Get Started

```bash
uv pip install nemo-automodel

automodel --nproc-per-node=2 llama3_2_1b_squad.yaml
```

See the [installation guide](guides/installation.md) for Docker, source builds, and multi-node setup.
See the [configuration guide](guides/configuration.md) for YAML recipes and CLI overrides.
Launch on a [local workstation](launcher/local-workstation.md) or [SLURM cluster](launcher/slurm.md).

## Latest Model Support

New models are added regularly. Pick a model below to start fine-tuning, or see the [full release log](model-coverage/latest-models.md).

| Date | Modality | Model |
|------|----------|-------|
| 2026-04-07 | LLM | [GLM-5.1](https://github.com/NVIDIA-NeMo/Automodel/discussions/1719) ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/glm/glm_5.1_hellaswag_pp.yaml)) |
| 2026-04-02 | VLM | Gemma 4 ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma4/gemma4_4b.yaml)) |
| 2026-03-16 | VLM | [Mistral Small 4](https://github.com/NVIDIA-NeMo/Automodel/discussions/1558) ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/mistral4/mistral4_medpix.yaml)) |
| 2026-03-11 | LLM | [Nemotron Super v3](https://github.com/NVIDIA-NeMo/Automodel/discussions/976) ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag.yaml)) |
| 2026-03-03 | Diffusion | FLUX.1-dev ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/diffusion/finetune/flux_t2i_flow.yaml)) |

## Recipes & Guides

Find the right guide for your task -- fine-tuning, pretraining, distillation, diffusion, and more.

| I want to...                | Choose this when...                                                                 | Input Data                                        | Model     | Guide                                                     |
| --------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------- | --------- | --------------------------------------------------------- |
| **SFT (full fine-tune)**    | You need maximum accuracy and have the GPU budget to update all weights             | Instruction / chat dataset                        | LLM       | [Start fine-tuning](guides/llm/finetune.md)               |
| **PEFT (LoRA)**             | You want to fine-tune on limited GPU memory; updates <1 % of parameters             | Instruction / chat dataset                        | LLM       | [Start LoRA](guides/llm/finetune.md)        |
| **Tool / function calling** | Your model needs to call APIs or tools with structured arguments                    | Function-calling dataset (queries + tool schemas) | LLM       | [Add tool calling](guides/llm/toolcalling.md)             |
| **Fine-tune VLM**           | Your task involves both images and text (e.g., visual QA, captioning)               | Image + text dataset                              | VLM       | [Fine-tune VLM](guides/omni/gemma3-3n.md)                 |
| **Fine-tune Gemma 4**       | You want to fine-tune Gemma 4 for structured extraction from images (e.g., receipts) | Image + text dataset                              | VLM       | [Fine-tune Gemma 4](guides/vlm/gemma4.md)                 |
| **Fine-tune dLLM**          | You want to fine-tune a diffusion language model (e.g., LLaDA) using masked denoising | Instruction / chat dataset                        | dLLM      | [Fine-tune dLLM](guides/dllm/finetune.md)                 |
| **Fine-tune Diffusion**     | You want to fine-tune a diffusion model for image or video generation               | Video / Image dataset                             | Diffusion | [Fine-tune Diffusion](guides/diffusion/finetune.md)       |
| **Fine-tune VLM-MoE**       | You need large-scale vision-language training with sparse MoE efficiency            | Image + text dataset                              | VLM (MoE) | [Fine-tune VLM-MoE](guides/vlm/qwen3-5.md)                |
| **Embedding fine-tune**     | You want to improve text similarity for search, retrieval, or RAG         | Text pairs / retrieval corpus                     | LLM       | {bdg-info}`Coming Soon`                                   |
| **Fine-tune a large MoE**   | You are adapting a large sparse MoE model (DeepSeek-V3, GLM-5, etc.) to your domain | Text dataset (e.g., HellaSwag)                    | LLM (MoE) | [Fine-tune MoE](guides/llm/large-moe-finetune.md)         |
| **Fine-tune DeepSeek V4 Flash** | You want to fine-tune the DeepSeek V4 Flash hybrid-attention MoE (SWA / CSA / HCA + hash-routing) | Text dataset (e.g., HellaSwag)                    | LLM (MoE) | [Fine-tune DeepSeek V4 Flash](guides/llm/dsv4-flash.md)   |
| **Sequence classification** | You need to classify text into categories (sentiment, topic, NLI)                   | Text + labels (e.g., GLUE MRPC)                   | LLM       | [Train classifier](guides/llm/sequence-classification.md) |
| **QAT fine-tune**           | You want a quantized model that keeps accuracy for efficient deployment             | Text dataset                                      | LLM       | [Enable QAT](guides/quantization-aware-training.md)       |
| **Knowledge distillation**  | You want a smaller, faster model that retains most of the teacher's quality         | Instruction dataset + teacher model               | LLM       | [Distill a model](guides/llm/knowledge-distillation.md)   |
| **Pretrain an LLM**         | You are building a base model from scratch on your own corpus                       | Large unlabeled text corpus (e.g., FineWeb-Edu)   | LLM       | [Start pretraining](guides/llm/pretraining.md)            |
| **Pretrain (NanoGPT)**      | You want quick pretraining experiments on a single node                             | FineWeb / text corpus                             | LLM       | [Try NanoGPT](guides/llm/nanogpt-pretraining.md)          |

## Performance

Training throughput on NVIDIA GPUs with optimized kernels for Hugging Face models.


| Model            | GPUs | TFLOPs/sec/GPU | Tokens/sec/GPU | Optimizations          |
| ---------------- | ---- | -------------- | -------------- | ---------------------- |
| DeepSeek V3 671B | 256  | 250            | 1,002          | TE + DeepEP            |
| GPT-OSS 20B      | 8    | 279            | 13,058         | TE + DeepEP + FlexAttn |
| Qwen3 MoE 30B    | 8    | 277            | 12,040         | TE + DeepEP            |


See the [full benchmark results](performance-summary.md) for configuration details and more models.

## Advanced Topics

Parallelism, precision, checkpointing strategies and experiment tracking.

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
Mixed-precision FP8 training with torchao.
+++
{bdg-secondary}`FP8` {bdg-secondary}`mixed-precision`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Checkpointing
:link: guides/checkpointing
:link-type: doc
Distributed checkpoints with SafeTensors output.
+++
{bdg-secondary}`DCP` {bdg-secondary}`safetensors`
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
{bdg-secondary}`QAT`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Experiment Tracking
:link: guides/mlflow-logging
:link-type: doc
Track experiments and metrics with MLflow and Wandb.
+++
{bdg-secondary}`MLflow` {bdg-secondary}`Wandb`
:::

::::

## For Developers

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Repo Internals
:link: repository-structure
:link-type: doc
Components, recipes, and CLI architecture.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: apidocs/index
:link-type: doc
Auto-generated Python API documentation.
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Use as a Library
:link: about/index
:link-type: doc
Drop-in accelerated backend for TRL, lm-eval-harness, OpenRLHF, or any code that loads Hugging Face models.
:::

::::

---

::::{toctree}
:hidden:
:caption: Get Started
About <about/index.md>
Key Features <about/key-features.md>
Installation <guides/installation.md>
Configuration <guides/configuration.md>
🤗 HF Compatibility <guides/huggingface-api-compatibility.md>
Repo Structure <repository-structure.md>
::::

::::{toctree}
:hidden:
:caption: Announcements
announcements.md
::::

::::{toctree}
:hidden:
:caption: Performance
performance-summary.md
::::

::::{toctree}
:hidden:
:caption: Model Coverage
Overview <model-coverage/overview.md>
Release Log <model-coverage/latest-models.md>
Large Language Models <model-coverage/llm/index.md>
Vison Language Models <model-coverage/vlm/index.md>
Omni <model-coverage/omni/index.md>
Diffusion <model-coverage/diffusion/index.md>
::::

::::{toctree}
:hidden:
:caption: Recipes & E2E Examples
Overview <guides/overview.md>
SFT & PEFT <guides/llm/finetune.md>
Function Calling <guides/llm/toolcalling.md>
guides/llm/knowledge-distillation.md
Large MoE Fine-Tuning <guides/llm/large-moe-finetune.md>
DeepSeek V4 Flash <guides/llm/dsv4-flash.md>
Pretraining <guides/llm/pretraining.md>
NanoGPT Pretraining <guides/llm/nanogpt-pretraining.md>
Sequence Classification <guides/llm/sequence-classification.md>
Gemma 3 / 3n <guides/omni/gemma3-3n.md>
Gemma 4 <guides/vlm/gemma4.md>
Qwen3.5-VL <guides/vlm/qwen3-5.md>
Diffusion Fine-Tuning <guides/diffusion/finetune.md>
dLLM Fine-Tuning <guides/dllm/finetune.md>
QAT <guides/quantization-aware-training.md>
Databricks <guides/llm/databricks.md>
::::

::::{toctree}
:hidden:
:caption: Datasets
Overview <guides/dataset-overview.md>
Text Dataset <guides/llm/dataset.md>
Retrieval Dataset <guides/llm/retrieval-dataset.md>
ColumnMapped Dataset <guides/llm/column-mapped-text-instruction-dataset.md>
ColumnMapped Iterable <guides/llm/column-mapped-text-instruction-iterable-dataset.md>
Multi-Modal Dataset <guides/vlm/dataset.md>
Diffusion Dataset <guides/diffusion/dataset.md>
::::

::::{toctree}
:hidden:
:caption: Job Launchers
Overview <launcher/overview.md>
Local Workstation <launcher/local-workstation.md>
SLURM Cluster <launcher/slurm.md>
NeMo-Run <launcher/nemo-run.md>
SkyPilot <launcher/skypilot.md>
SkyPilot k8s <launcher/skypilot-kubernetes.md>
::::

::::{toctree}
:hidden:
:caption: Development
guides/checkpointing.md
Gradient Checkpointing <guides/gradient-checkpointing.md>
Pipeline Parallelism <guides/pipelining.md>
guides/fp8-training.md
guides/mlflow-logging.md
API Reference <apidocs/index.rst>
::::
