<div align="center">

# NeMo AutoModel

</div>

<div align="center">

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->
[![codecov](https://codecov.io/github/NVIDIA-NeMo/Automodel/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Automodel)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Automodel/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Automodel/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Automodel.svg?style=social&label=Star)](https://github.com/NVIDIA-NeMo/Automodel/stargazers/)

<!-- **Day-0 integration with Hugging Face models automating fine-tuning and pretraining with pytorch-native parallelism, custom-kernels and optimized recipes** -->
**DTensor‑native SPMD library for large‑scale training, with Hugging Face‑native fine‑tuning and pretraining.**

[📖 Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html) • [🔥 Ready-to-Use Recipes](https://github.com/NVIDIA-NeMo/Automodel/#-ready-to-use-recipes) • [💡 Examples](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples) • [🤝 Contributing](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md)

</div>

---

NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. It enables seamless scaling of training (both pretraining and post-training) workloads from single GPU to thousand-node clusters for both 🤗Hugging Face/PyTorch and Megatron models. It includes a suite of libraries and recipe collections to help users train models from end to end. The **AutoModel library ("NeMo AutoModel")** provides GPU-accelerated PyTorch training for 🤗Hugging Face models on **Day-0**. Users can start training and fine-tuning models instantly without conversion delays, scale effortlessly with PyTorch-native parallelisms, optimized custom kernels, and memory-efficient recipes-all while preserving the original checkpoint format for seamless use across the Hugging Face ecosystem.

> ⚠️ Note: NeMo AutoModel is under active development. New features, improvements, and documentation updates are released regularly. We are working toward a stable release, so expect the interface to solidify over time. Your feedback and contributions are welcome, and we encourage you to follow along as new updates roll out.

## Table of Contents
- [Feature Roadmap](#feature-roadmap)
- [Design Principles](#design-principles)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Mesh‑Aware Checkpointing](#mesh-aware-checkpointing)
- [Training Examples](#training-examples)
- [Supported Models](#supported-models)
- [Performance](#performance)
- [Interoperability](#interoperability)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)


## Feature Roadmap

✅ _Available now_ | 🔜 _Coming in 25.11_

- ✅ **HuggingFace Integration** - Works with 1-70B models (Qwen, Llama).
- ✅ **Distributed Training** - Fully Sharded Data Parallel (FSDP2) support.
- ✅ **Environment Support** - Support for SLURM and interactive training.
- ✅ **Learning Algorithms** - SFT (Supervised Fine-Tuning), and PEFT (Parameter Efficient Fine-Tuning).
- ✅ **Large Model Support** - Native PyTorch support for models up to 70B parameters.
- ✅ **Advanced Parallelism** - PyTorch native FSDP2, TP, CP, and SP for efficient training.
- ✅ **Sequence Packing** - Sequence packing in both DTensor and MCore for huge training perf gains.
- ✅ **DCP** - Distributed Checkpoint support with SafeTensors output.
- ✅ **HSDP** - Hybrid Sharding Data Parallelism based on FSDP2.
- ✅ **Pipeline Support** - Torch-native support for pipelining composable with FSDP2 and DTensor (3D Parallelism).
- ✅ **Pre-training** - Support for model pre-training, including DeepSeekV3.
- ✅ **Knowledge Distillation** - Support for knowledge distillation with LLMs; VLM support will be added post 25.09.

- 🔜 **Extended MoE support** - GPT-OSS, Qwen3 (Coder-480B-A35B, etc), Qwen-next.

## Design Principles

- **DTensor‑native**: Partition model/optimizer states with `DeviceMesh` + placements (`Shard`, `Replicate`).
- **SPMD first**: Parallelism is configuration. No model rewrites when scaling up or changing strategy.
- **HF integration**: Operate on native 🤗 checkpoints/configs for frictionless fine‑tuning and pretraining.
- **Minimal ceremony**: YAML‑driven recipes; override any field via CLI.

## Why SPMD instead of framework‑specific parallel?

- **One program, any scale**: The same training script runs on 1 GPU or 100+ by changing the mesh.
- **Decoupled concerns**: Model code stays pure PyTorch; parallel strategy lives in config.
- **Composability**: Mix **tensor**, **sequence**, and **data** parallel by editing placements.
- **Portability**: Fewer bespoke abstractions; easier to reason about failure modes and restarts.
- **Interoperability**: HF models/tokenizers/optimizers plug in directly; no format round‑trips.

> TL;DR: SPMD turns “how to parallelize” into a *runtime layout choice*, not a code fork.

## Key Features

- **Mesh‑defined parallelism**: Compose tensor/sequence/data parallel by changing placements and sizes.
- **FSDP2 on DTensor**: Memory‑efficient sharding (HSDP included) for large models.
- **Pretraining & fine‑tuning**: Day‑0 support for both regimes with shared configs/utilities.
- **HF‑native I/O**: Train from 🤗 configs/weights; export consolidated HF checkpoints.
- **Mixed precision**: BF16/FP16/FP8; sequence packing; optimized CUDA kernels.
- **PEFT built‑in**: LoRA and hooks for custom adapters.
- **Mesh‑aware DCP**: Sharded SafeTensors with merge/reshard utilities.
- **Flexible Configuration**: YAML-based configuration system for reproducible experiments
- **FP8 Precision**: Native FP8 training & inference for higher throughput and lower memory use
- **Day-0 Hugging Face Support**: Instantly fine-tune any model from the Hugging Face Hub
- **Large-Scale Distributed Training**: Built-in FSDP2 and Megatron-FSDP for seamless multi-node scaling
- **Vision-Language Model Ready**: Native support for VLMs (Qwen2-VL, Gemma-3-VL, etc)

## Getting Started

We recommend **uv** for reproducible environments.

```bash
uv venv
uv pip install nemo_automodel # latest release
# or: uv pip install git+https://nam11.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgithub.com%2FNVIDIA-NeMo%2FAutomodel.git&data=05%7C02%7Cakoumparouli%40nvidia.com%7C88f177d2fa09407d30c408ddf316ef59%7C43083d15727340c1b7db39efd9ccc17a%7C0%7C0%7C638934002060056854%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=5CnwNAdBiwubNW9zK5QC%2FGLAiDqVIJoNFix47upiIeE%3D&reserved=0
uv run python -c "import nemo_automodel; print('AutoModel ready')"
```

> Ensure recent CUDA/PyTorch. Some kernels (e.g., FlashAttention‑style) may JIT on first run.

## Mesh‑Aware Checkpointing

AutoModel writes **Distributed Checkpoints (DCP)** with SafeTensors
shards. Checkpoints carry partition metadata to:

- **Merge** into a single HF‑compatible checkpoint for inference.
- **Reshard** when loading onto a different mesh/topology.

YAML sketch:
```yaml
checkpoint:
enabled: true
checkpoint_dir: ./checkpoints
save_consolidated: true
model_save_format: safetensors
```

## Training Examples

**LLM**
```bash
# Memory‑efficient SFT with LoRA
uv run examples/llm_finetune/finetune.py \
--config examples/llm_finetune/llama/llama3_2_1b_hellaswag_peft.yaml
```

**VLM**
```bash
# Qwen2.5‑VL on a single GPU
uv run examples/vlm_finetune/finetune.py \
--config examples/vlm_finetune/qwen2_5_vl_3b_rdr.yaml
```


## Supported Models
NeMo AutoModel provides native support for a wide range of models available on the Hugging Face Hub, enabling efficient fine-tuning for various domains. Below is a small sample of ready‑to‑use families (train as‑is or swap any compatible 🤗 causal LM):

| Domain | Model Family | Model ID | Recipes |
|--------|--------------|----------|---------|
| **LLM** |  **LLaMA** | [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml) |
| | | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_squad_peft.yaml) |
| | | [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) | [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_fp8.yaml) |
| **LLM** | **Mistral** | [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_hellaswag_fp8.yaml) |
|  |  | [`mistralai/Mistral-Nemo-Base-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_hellaswag_fp8.yaml) |
|  |  | [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) |[PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mixtral_8x7b_instruct_squad_peft.yaml) |
| **LLM** | **Qwen** | [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwen2_5_7b_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwen2_5_7b_hellaswag_fp8.yaml) |
|  |  | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag_peft.yaml) |
|  |  | [`Qwen/QwQ-32B`](https://huggingface.co/Qwen/QwQ-32B) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwq_32b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/qwen/qwq_32b_squad_peft.yaml) |
| **LLM** | **Gemma** | [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_3_270m_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_3_270m_squad_peft.yaml) |
| | | [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_2_9b_it_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_2_9b_it_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_2_9b_it_hellaswag_fp8.yaml) |
| | | [`google/gemma-7b`](https://huggingface.co/google/gemma-7b) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_7b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gemma/gemma_7b_squad_peft.yaml) |
| **LLM** | **Phi** | [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_2_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_2_squad_peft.yaml) |
|  |  | [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_3_mini_it_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_3_mini_it_squad_peft.yaml) |
|  |  | [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_4_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_4_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/phi/phi_4_hellaswag_fp8.yaml) |
| **LLM** | **Seed** | [`ByteDance-Seed/Seed-Coder-8B-Instruct`](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/seed/seed_coder_8b_instruct_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/seed/seed_coder_8b_instruct_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/seed/seed_coder_8b_instruct_hellaswag_fp8.yaml) |
|  |  | [`ByteDance-Seed/Seed-OSS-36B-Instruct`](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/seed/seed_oss_36B_hellaswag.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/seed/seed_oss_36B_hellaswag_peft.yaml) |
| **LLM** | **Baichuan** | [`baichuan-inc/Baichuan2-7B-Chat`](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/baichuan/baichuan_2_7b_mock_fp8.yaml) |
| **VLM** | **Gemma** | [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml) |
|  |  | [`google/gemma-3n-e4b-it`](https://huggingface.co/google/gemma-3n-e4b-it) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix_peft.yaml) |

**And more**: Check out more [LLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune) and [VLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune) examples! Any causal LM on Hugging Face Hub can be used with the base recipe template!

### Run a Recipe
To run a NeMo AutoModel recipe, you need a recipe script (e.g., [LLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/finetune.py), [VLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/finetune.py)) and a YAML config file (e.g., [LLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama/llama3_2_1b_squad.yaml), [VLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml)):
```
# Command invocation format:
uv run <recipe_script_path> --config <yaml_config_path>

# LLM example: multi-GPU with FSDP2
uv run torchrun --nproc-per-node=8 recipes/llm_finetune/finetune.py --config recipes/llm_finetune/llama/llama3_2_1b_hellaswag.yaml

# VLM example: single GPU fine-tuning (Gemma-3-VL) with LoRA
uv run recipes/vlm_finetune/finetune.py --config recipes/vlm_finetune/gemma3/gemma3_vl_3b_cord_v2_peft.yaml
```

See `examples/` for the latest matrix.

---
## Performance

Coming soon..

## Interoperability

- **Hugging Face**: Train from and export to native 🤗 formats.
- **Megatron Bridge**: Optional conversions to/from Megatron formats
for specific workflows.
- **NeMo RL**: Use AutoModel checkpoints directly as starting points
for DPO/RM/GRPO pipelines.

--

## 🗂️ Project Structure

```
NeMo-Automodel/
├── nemo_automodel/              # Core library
│   ├── _peft/                   # PEFT implementations (LoRA)
│   ├── _transformers/           # HF model integrations  
│   ├── checkpoint/              # Distributed checkpointing
│   ├── datasets/                # Dataset loaders
│   │   ├── llm/                 # LLM datasets (HellaSwag, SQuAD, etc.)
│   │   └── vlm/                 # VLM datasets (CORD-v2, rdr etc.)
│   ├── distributed/             # FSDP2, Megatron FSDP, parallelization
│   ├── loss/                    # Optimized loss functions
│   └── training/                # Training recipes and utilities
├── recipes/                     # Ready-to-use training recipes
│   ├── llm/                     # LLM fine-tuning recipes
│   └── vlm/                     # VLM fine-tuning recipes  
└── tests/                       # Comprehensive test suite
```

---
## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md) for details.
---

## 📄 License

NVIDIA NeMo AutoModel is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Automodel/blob/main/LICENSE).

---


## 🔗 Links

- **Documentation**: https://docs.nvidia.com/nemo-framework/user-guide/latest/automodel/index.html
- **Hugging Face Hub**: https://huggingface.co/models
- **Issues**: https://github.com/NVIDIA-NeMo/Automodel/issues
- **Discussions**: https://github.com/NVIDIA-NeMo/Automodel/discussions
