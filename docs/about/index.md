---
description: "Overview of NeMo AutoModel, a PyTorch DTensor-native SPMD library with optimized model implementations and a Hugging Face-compatible API for training, fine-tuning, and as an accelerated backend for other frameworks"
categories: ["getting-started"]
tags: ["overview", "spmd", "dtensor", "distributed", "getting-started"]
personas: ["machine-learning-engineer", "researcher", "devops"]
difficulty: "beginner"
content_type: "concept"
---

(about-overview)=

# About NeMo AutoModel

NeMo AutoModel is a PyTorch DTensor-native SPMD open-source library under [NVIDIA NeMo Framework](https://github.com/NVIDIA-NeMo). It provides **optimized model implementations** with a **Hugging Face-compatible API**, so any model on the Hub works out of the box with no checkpoint conversion. On top of that, it ships ready-made **recipes** for training and fine-tuning LLMs and VLMs at scale.

Because AutoModel exposes the same Autoclass interface as `transformers`, it can also be used as a **drop-in accelerated backend for other libraries** -- reinforcement learning frameworks, evaluation harnesses, or any codebase that loads Hugging Face models.

## Target Users

- **Machine learning engineers**: Fine-tune and pre-train LLMs and VLMs at scale with minimal boilerplate.
- **Researchers**: Rapidly prototype with hackable, linear training scripts and YAML-driven configuration.
- **Library and framework authors**: Use AutoModel's optimized model implementations as a drop-in replacement for `transformers` to accelerate RL, alignment, evaluation, or any downstream workflow.

## How It Works

NeMo AutoModel is built around two core ideas: **recipes** and **components**.

- **Recipes** are executable Python scripts paired with YAML configs. Each recipe defines an end-to-end workflow -- model loading, data preparation, training loop, and checkpointing -- and can be launched with a single command.
- **Components** are modular, self-contained building blocks (datasets, optimizers, loss functions, distribution strategies) that recipes compose together. Swap any component by changing a `_target_` field in your YAML.

This design means the training loop is always visible and hackable -- no hidden abstractions. You configure parallelism, precision, and scaling through config, not code changes.

### SPMD and DTensor

NeMo AutoModel uses PyTorch's native SPMD (Single Program, Multiple Data) model with DTensor and DeviceMesh:

- **One program, any scale**: The same training script runs on 1 GPU or 1000+ by changing the mesh configuration.
- **Parallelism is configuration**: Mix tensor, sequence, pipeline, and data parallelism by editing placements -- no model rewrites.
- **Decoupled concerns**: Model code stays pure PyTorch; the parallel strategy lives in config.

### Key Technologies

- **FSDP2 and MegatronFSDP**: Memory-efficient sharded data parallelism for large-scale training, including Hybrid Sharding (HSDP).
- **Pipeline Parallelism**: Torch-native pipelining composable with FSDP2 and DTensor for 3D parallelism.
- **Custom CUDA Kernels**: Fused attention, TransformerEngine, DeepEP, and FlexAttn for optimized throughput.
- **FP8 Mixed Precision**: FP8 training via torchao for supported models.
- **Distributed Checkpointing (DCP)**: Sharded SafeTensors checkpoints with merge and reshard utilities, interoperable with Hugging Face.

## Hugging Face Integration

NeMo AutoModel builds on top of `transformers` rather than replacing it:

- Load any `AutoModelForCausalLM` or `AutoModelForImageTextToText` model directly from the Hub.
- Use Hugging Face tokenizers, datasets, and chat templates as-is.
- Checkpoints stay in the native Hugging Face format -- no conversion step before or after training.
- New models released on the Hub get day-0 support because AutoModel tracks the latest `transformers` version.

See the [Hugging Face API Compatibility](../guides/huggingface-api-compatibility.md) guide and [Model Coverage](../model-coverage/overview.md) for details.

## Optimized Model Implementations

AutoModel ships optimized implementations for supported architectures (fused attention, TransformerEngine layers, DeepEP for MoE routing, FlexAttn) while keeping the standard `transformers` API surface. This means:

- **Faster training and inference** with no code changes -- load a model the same way you would with `transformers` and get accelerated kernels automatically.
- **No checkpoint conversion** -- weights are loaded from and saved to the native Hugging Face format.
- **Day-0 model support** -- because AutoModel builds on `transformers`, newly released models on the Hub work immediately. Optimized kernels are added incrementally for popular architectures.

## Use as a Library

NeMo AutoModel is not limited to its built-in training recipes. Because it implements the Hugging Face `AutoModel` API, any library or framework that loads models through `transformers` can swap in AutoModel to get optimized performance:

- **Reinforcement learning** (e.g., TRL, OpenRLHF) -- replace the policy or reference model with an AutoModel instance for faster rollouts and gradient steps.
- **Evaluation and benchmarking** -- plug into lm-evaluation-harness or custom eval loops with no API changes.
- **Custom training loops** -- import individual components (optimizers, loss functions, distributed strategies) without using recipes at all.

```python
from nemo_automodel import NeMoAutoModelForCausalLM

model = NeMoAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
```

The returned model is a standard `nn.Module` with the same forward signature as the `transformers` equivalent, so it works anywhere a Hugging Face model is expected.

## What's Next

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Key Features and Concepts
:link: about-key-features
:link-type: ref
Explore the main features, supported workflows, and core concepts.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: ../index
:link-type: doc
Jump to the quickstart table to find the right guide for your task.
:::

::::
