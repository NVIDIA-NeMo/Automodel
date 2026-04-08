<div align="center">

# 🚀 NeMo AutoModel

</div>

<div align="center">

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->
[![codecov](https://codecov.io/github/NVIDIA-NeMo/Automodel/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Automodel)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Automodel/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Automodel/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Automodel.svg?style=social&label=Star)](https://github.com/NVIDIA-NeMo/Automodel/stargazers/)

<!-- **Day-0 integration with Hugging Face models automating fine-tuning and pretraining with pytorch-native parallelism, custom-kernels and optimized recipes**
**Pytorch DTensor‑native SPMD library for large‑scale training**-->

[📖 Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html) • [🔥 Ready-to-Use Recipes](https://github.com/NVIDIA-NeMo/Automodel/#supported-models) • [💡 Examples](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples) • [Model Coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/overview.html) • [Performance](https://docs.nvidia.com/nemo/automodel/latest/performance-summary.html) • [🤝 Contributing](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md)

</div>

## 📣 News and Discussions
- [04/07/2026][**GLM-5.1**](https://huggingface.co/zai-org/GLM-5.1) We now support finetuning `zai-org/GLM-5.1`. GLM-5.1 is Zhipu AI's latest open-source MoE model featuring MLA + Dynamic Sparse Attention. Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/glm/glm_5.1_hellaswag_pp.yaml) and [discussion](https://github.com/NVIDIA-NeMo/Automodel/discussions/1719).
- [04/02/2026][**Gemma 4**](https://huggingface.co/collections/google/gemma-4) We support fine-tuning for Gemma4 (2B, 4B, 31B, 26BA4B)! Check out our [recipes](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune/gemma4).
- [03/30/2026]**NeMo AutoModel** ships with **agent-friendly skills** in [skills/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/skills) to help you with common development tasks (e.g., running a recipe, model onboarding, development) across the repo. We welcome PRs that improve existing skills or add new ones.
- [03/16/2026][**Mistral Small 4**](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) We support fine-tuning for Mistral4 119B! Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/mistral4/mistral4_medpix.yaml).
- [03/11/2026][**Nemotron Super v3**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) We support fine-tuning for `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`. Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron/nemotron_super_v3_hellaswag.yaml).
- [03/11/2026][**GLM-5**](https://huggingface.co/zai-org/GLM-5) We now support finetuning `zai-org/GLM-5`. Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/glm/glm_5_hellaswag_pp.yaml).
- [03/02/2026][**Qwen3.5 small models**](https://huggingface.co/collections/Qwen/qwen35) We support finetuning for Qwen3.5 small models 0.8B, 2B, 4B ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml)) and 9B ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/qwen3_5/qwen3_5_9b.yaml))
- [02/16/2026][**Qwen3.5 MoE**](https://huggingface.co/collections/Qwen/qwen35) We support finetuning for `Qwen/Qwen3.5-397B-A17B` ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml)) and `Qwen/Qwen3.5-35B-A3B` ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/qwen3_5_moe/qwen3_5_35b.yaml))
- [02/13/2026][**MiniMax-M2.5**](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) We support finetuning for `MiniMaxAI/MiniMax-M2.5`. Checkout our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml)
- [02/11/2026][**GLM-4.7-Flash**](https://huggingface.co/zai-org/GLM-4.7-Flash) We now support finetuning GLM-4.7-Flash. Checkout our [packed sequence recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/glm/glm_4.7_flash_te_packed_sequence.yaml)
- [02/09/2026][**MiniMax-M2**](https://huggingface.co/MiniMaxAI/MiniMax-M2) We support finetuning for `MiniMaxAI/MiniMax-M2`. Checkout our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/5f63eb428bacf4146e9a5ae9949d58c5751df7b9/examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml)
- [02/06/2026][**Qwen3 VL 235B**](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) We support finetuning for `Qwen/Qwen3-VL-235B-A22B-Instruct`. Checkout our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/qwen3/qwen3_vl_moe_235b.yaml)
- [02/06/2026][**GLM4.7**](https://huggingface.co/zai-org/GLM-4.7) We now support finetuning GLM4.7. Checkout our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/glm/glm_4.7_te_deepep.yaml)
- [02/06/2026][**Step3.5-flash**](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is out! Finetune it with our [finetune recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml)
- [02/05/2026][**DeepSeek-V3.2**](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) is out! Checkout out [the finetune recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml)!
- [02/04/2026][**Kimi K2.5 VL**](https://huggingface.co/moonshotai/Kimi-K2.5) is out! Finetune it with [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/discussions/1161)
- [01/30/2026][**Kimi VL**](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) We support fine-tuning for `moonshotai/Kimi-VL-A3B-Instruct`. Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml).
- [01/12/2026][**Nemotron Flash**](https://huggingface.co/nvidia/Nemotron-Flash-1B) We support fine-tuning for `nvidia/Nemotron-Flash-1B`. Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml).
- [01/12/2026][**Nemotron Parse**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) We support fine-tuning `nvidia/NVIDIA-Nemotron-Parse-v1.1` ([recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml), [tutorial](https://github.com/NVIDIA-NeMo/Automodel/blob/main/tutorials/nemotron-parse/finetune.ipynb)) <a href="https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3BxDyGV2RLNqyftNxJO5rV5sp8x"><img
  src="https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg" alt="Launch on Brev" height="23" style="vertical-align: middle;"></a>.
- [01/07/2026][**Devstral-Small**](https://huggingface.co/mistralai/Devstral-Small-2512) We support fine-tuning for `mistralai/Devstral-Small-2512`. Check out our [recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/devstral/devstral2_small_2512_squad.yaml).
- [12/18/2025][**FunctionGemma**](https://huggingface.co/google/functiongemma-270m-it) is out! Finetune it with [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/llm/toolcalling.md)!
- [12/15/2025][**NVIDIA-Nemotron-3-Nano-30B-A3B**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) is out! Finetune it with [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/discussions/976)!
- [11/6/2025][Accelerating Large-Scale Mixture-of-Experts Training in PyTorch](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/)
- [10/6/2025][Enabling PyTorch Native Pipeline Parallelism for 🤗 Hugging Face Transformer Models](https://github.com/NVIDIA-NeMo/Automodel/discussions/589)
- [9/22/2025][Fine-tune Hugging Face Models Instantly with Day-0 Support with NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/discussions/477)
- [9/18/2025][🚀 NeMo Framework Now Supports Google Gemma 3n: Efficient Multimodal Fine-tuning Made Simple](https://github.com/NVIDIA-NeMo/Automodel/discussions/494)

Overview
---

Nemo AutoModel is a Pytorch DTensor‑native SPMD open-source training library under [NVIDIA NeMo Framework](https://github.com/NVIDIA-NeMo), designed to streamline and scale training and finetuning for LLMs and VLMs. Designed for flexibility, reproducibility, and scale, NeMo AutoModel enables both small-scale experiments and massive multi-GPU, multi-node deployments for fast experimentation in research and production environments.
<p align="center">
<a href="https://github.com/NVIDIA-NeMo/Automodel"><picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/refs/heads/main/docs/automodel_diagram.png">
    <img alt="AutoModel Logo" src="https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/refs/heads/main/docs/automodel_diagram.png">
</picture></a>
</p>


What you can expect:

- **Hackable** with a modular design that allows easy integration, customization, and quick research prototypes.
- **Minimal ceremony**: YAML-driven recipes; override any field using CLI.
- **High performance and flexibility** with custom kernels and DTensor support.
- **Seamless integration** with Hugging Face for day-0 model support, ease of use, and wide range of supported models.
- **Efficient resource management** using Kubernetes and Slurm, enabling scalable and flexible deployment across configurations.
- **Documentation** with step-by-step guides and runnable examples.

<!-- Please refer to our design documents for more details on the architecture and design philosophy. -->

<!-- NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. It enables seamless scaling of training (both pretraining and post-training) workloads from single GPU to thousand-node clusters for both 🤗Hugging Face/PyTorch and Megatron models. It includes a suite of libraries and recipe collections to help users train models from end to end. The **AutoModel library ("NeMo AutoModel")** provides GPU-accelerated PyTorch training for 🤗Hugging Face models on **Day-0**. Users can start training and fine-tuning models instantly without conversion delays, scale effortlessly with PyTorch-native parallelisms, optimized custom kernels, and memory-efficient recipes-all while preserving the original checkpoint format for seamless use across the Hugging Face ecosystem. -->

> ⚠️ Note: NeMo AutoModel is under active development. New features, improvements, and documentation updates are released regularly. We are working toward a stable release, so expect the interface to solidify over time. Your feedback and contributions are welcome, and we encourage you to follow along as new updates roll out.

### Why PyTorch Distributed and SPMD

- **One program, any scale**: The same training script runs on 1 GPU or 1000+ by changing the mesh.
- **PyTorch Distributed native**: Partition model/optimizer states with `DeviceMesh` + placements (`Shard`, `Replicate`).
- **SPMD first**: Parallelism is configuration. No model rewrites when scaling up or changing strategy.
- **Decoupled concerns**: Model code stays pure PyTorch; parallel strategy lives in config.
- **Composability**: Mix **tensor**, **sequence**, and **data** parallel by editing placements.
- **Portability**: Fewer bespoke abstractions; easier to reason about failure modes and restarts.
<!-- - **Interoperability**: HF models/tokenizers/optimizers plug in directly; no format round‑trips. -->

<!-- ### Key Features -->

<!-- - **Mesh‑defined parallelism**: Compose tensor/sequence/pipeline/data parallel by changing placements and sizes. -->
<!-- - **FSDP2 on DTensor**: Memory‑efficient sharding (HSDP included) for large scale training. -->
<!-- - **Pretraining, SFT & PEFT**: Day‑0 support for LLMs both regimes with shared configs/utilities.
- **Mixed precision**: BF16/FP16/FP8; sequence packing; optimized CUDA kernels. -->
<!-- - **Mesh‑aware DCP**: Sharded SafeTensors with merge/reshard utilities; interoperable with HF. -->
<!-- - **Large-Scale Distributed Training**: Built-in FSDP2 and Megatron-FSDP for seamless multi-node scaling. -->
<!-- - **Vision-Language Model Ready**: Native support for VLMs (Qwen2-VL, Gemma-3-VL, etc). -->
<!-- - **Day-0 Hugging Face Support**: Instantly fine-tune any model from the Hugging Face Hub. -->


## Table of Contents
- [Feature Roadmap](#feature-roadmap)
- [Getting Started](#getting-started)
- [LLM](#llm-pre-training)
  - [Pre-training](#llm-pre-training)
  - [Supervised Fine-Tuning (SFT)](#llm-supervised-fine-tuning-sft)
  - [Parameter-Efficient Fine-Tuning (PEFT)](#llm-parameter-efficient-fine-tuning-peft)
- [VLM](#vlm-supervised-fine-tuning-sft)
  - [Supervised Fine-Tuning (SFT)](#vlm-supervised-fine-tuning-sft)
  - [Parameter-Efficient Fine-Tuning (PEFT)](#vlm-parameter-efficient-fine-tuning-peft)
- [Supported Models](#supported-models)
- [Performance](#performance)
- [Interoperability](#-interoperability)
- [Contributing](#-contributing)
- [License](#-license)

> TL;DR: SPMD turns “how to parallelize” into a *runtime layout choice*, not a code fork.

## Feature Roadmap

✅ _Available now_ | 🔜 _Coming in 26.02_

- ✅ **Advanced Parallelism** - PyTorch native FSDP2, TP, CP, and SP for distributed training.
- ✅ **HSDP** - Multi-node Hybrid Sharding Data Parallelism based on FSDP2.
- ✅ **Pipeline Support** - Torch-native support for pipelining composable with FSDP2 and DTensor (3D Parallelism).
- ✅ **Environment Support** - Support for SLURM and interactive training.
- ✅ **Learning Algorithms** - SFT (Supervised Fine-Tuning), and PEFT (Parameter Efficient Fine-Tuning).
- ✅ **Pre-training** - Support for model pre-training, including DeepSeekV3.
- ✅ **Knowledge Distillation** - Support for knowledge distillation with LLMs; VLM support will be added post 25.09.
- ✅ **HuggingFace Integration** - Works with dense models (e.g., Qwen, Llama3, etc) and large MoEs (e.g., DSv3).
- ✅ **Sequence Packing** - Sequence packing for huge training perf gains.
- ✅ **FP8 and mixed precision** - FP8 support with torchao, requires torch.compile-supported models.
- ✅ **DCP** - Distributed Checkpoint support with SafeTensors output.
- ✅ **VLM**: Support for finetuning VLMs (e.g., Qwen2-VL, Gemma-3-VL). More families to be included in the future.
- ✅ **Extended MoE support** - GPT-OSS, Qwen3 (Coder-480B-A35B, etc), Qwen-next.

- 🔜 **Transformers v5 🤗** - Support for transformers v5 🤗 with device-mesh driven parallelism.
- 🔜 **Muon & Dion** - Support for Muon and Dion optimizers.
- 🔜 **SonicMoE** - Optimized MoE implementation for faster expert computation.
- 🔜 **FP8 MoE** - FP8 precision training and inference for MoE models.
- 🔜 **Cudagraph with MoE** - CUDA graph support for MoE layers to reduce kernel launch overhead.
- 🔜 **Extended VLM Support** - DeepSeek OCR, Qwen3 VL 235B, Kimi-VL, GLM4.5V
- 🔜 **Extended LLM Support** - QWENCoder 480B Instruct, MiniMax2.1, and more
- 🔜 **Kubernetes** - Multi-node job launch with k8s.


## Getting Started

We recommend using **uv** for reproducible Python environments.

```bash
# Setup environment before running any recipes
uv venv

# Choose ONE:
uv sync --frozen  # LLM recipes (default)
# uv sync --frozen --extra vlm  # VLM recipes (fixes: ImportError: qwen_vl_utils is not installed)
# uv sync --frozen --extra cuda  # Optional CUDA deps (e.g., Transformer Engine, bitsandbytes)
# uv sync --frozen --extra all  # Most optional deps (includes `vlm` and `cuda`)
# uv sync --frozen --all-extras  # Everything (includes `fa`, `moe`, etc.)

# One-off runs (examples):
# uv run --extra vlm <command>
# uv run --extra cuda <command>

uv run python -c "import nemo_automodel; print('NeMo AutoModel ready')"
```


### Run a Recipe
All recipes are launched via the `automodel` CLI (or its short alias `am`). Each YAML config specifies the recipe class and all training parameters:
```bash
# LLM example: multi-GPU fine-tuning with FSDP2
automodel examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml --nproc-per-node 8

# VLM example: single-GPU fine-tuning (Gemma-3-VL) with LoRA
automodel examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml

# Both commands also work with uv run:
uv run automodel examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml --nproc-per-node 8
```

> [!TIP]
> **Login-node / CI installs:** If you only need to submit jobs (SLURM, k8s, NeMo-Run) and don't need to train locally, install the lightweight CLI package: `pip install nemo-automodel[cli]`


## LLM Pre-training
### LLM Pre-training Single Node
We provide an example SFT experiment using the [FineWeb dataset](https://arxiv.org/abs/2406.17557/) with a nano-GPT model, ideal for quick experimentation on a single node.
```sh
automodel examples/llm_pretrain/nanogpt_pretrain.yaml --nproc-per-node 8
```

<!-- ### LLM Pre-training Multi Node -->

## LLM Supervised Fine-Tuning (SFT)
We provide an example SFT experiment using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).

<!-- Refer to `examples/llm_finetune/annotated.yaml` for a full list of parameters that can be overridden. -->

### LLM SFT Single Node

The default SFT configuration is set to run on a single GPU. To start the experiment:

```sh
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

This fine-tunes the `Llama3.2-1B` model on the SQuAD dataset using a single GPU.

To use multiple GPUs on a single node, add the `--nproc-per-node` argument:

```sh
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node 8
```

### LLM SFT Multi Node
To launch on a SLURM cluster, copy the reference sbatch script and adapt it to your cluster:
```sh
cp slurm.sub my_cluster.sub
# Edit my_cluster.sub — change CONFIG, #SBATCH directives, container, mounts, etc.
sbatch my_cluster.sub
```

All cluster-specific settings (nodes, GPUs, partition, container, mounts) live in your sbatch script.
NeMo-Run (`nemo_run:`) sections are also supported -- see our
[cluster guide](https://docs.nvidia.com/nemo/automodel/latest/launcher/cluster.html) for details.

## LLM Parameter-Efficient Fine-Tuning (PEFT)

We provide a PEFT example using the [HellaSwag dataset](https://rowanzellers.com/hellaswag/).

### LLM PEFT Single Node
```bash
# Memory-efficient SFT with LoRA
automodel examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml

# Override any YAML parameter via the command line:
automodel examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml \
  --step_scheduler.local_batch_size 16
```

> [!NOTE]
> Launching a multi-node PEFT example uses the same `sbatch slurm.sub` workflow as the SFT case above.


## VLM Supervised Fine-Tuning (SFT)

We provide a VLM SFT example using Qwen2.5-VL for end-to-end fine-tuning on image-text data.

### VLM SFT Single Node
```bash
# Qwen2.5-VL on 8 GPUs
automodel examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml --nproc-per-node 8
```

## VLM Parameter-Efficient Fine-Tuning (PEFT)

We provide a VLM PEFT (LoRA) example for memory-efficient adaptation with Gemma3 VLM.

### VLM PEFT Single Node
```bash
# Gemma-3-VL PEFT on 8 GPUs
automodel examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix_peft.yaml --nproc-per-node 8
```


## Supported Models
NeMo AutoModel provides native support for a wide range of models available on the Hugging Face Hub, enabling efficient fine-tuning for various domains. Below is a small sample of ready-to-use families (train as-is or swap any compatible 🤗 causal LM), you can specify nearly any LLM/VLM model available on 🤗 hub:

| Domain | Model Family | Model ID | Recipes |
|--------|--------------|----------|---------|
| **LLM** | **GPT-OSS** | [`GPT-OSS-20B`](https://huggingface.co/openai/gpt-oss-20b) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gpt_oss/gpt_oss_20b.yaml) |
|  |  | [`GPT-OSS-120B`](https://huggingface.co/openai/gpt-oss-120b) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/gpt_oss/gpt_oss_120b.yaml) |
| **LLM** | **DeepSeek** | [`DeepSeek-V3`](https://huggingface.co/deepseek-ai/DeepSeek-V3) | [Pretrain](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/deepseekv3_pretrain.yaml) |
| **LLM** | **Moonlight** | [`Moonlight-16B-TE`](https://huggingface.co/moonshotai/Moonlight-16B-A3B) | [Pretrain](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/megatron_pretrain_moonlight_16b_te_slurm.yaml), [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/moonlight/moonlight_16b_te.yaml) |
| **LLM** |  **LLaMA** | [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml) |
| | | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_squad_peft.yaml) |
| | | [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) | [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_fp8.yaml) |
| | | [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad_peft.yaml) |
| **LLM** | **Mistral** | [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_7b_hellaswag_fp8.yaml) |
|  |  | [`mistralai/Mistral-Nemo-Base-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_squad_peft.yaml), [FP8](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mistral_nemo_2407_hellaswag_fp8.yaml) |
|  |  | [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | [SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad.yaml), [PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad_peft.yaml) |
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

> [!NOTE]
> Check out more [LLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune) and [VLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune) examples. Any causal LM on Hugging Face Hub can be used with the base recipe template, just overwrite `--model.pretrained_model_name_or_path <model-id>` in the CLI or in the YAML config.


## Performance

NeMo AutoModel achieves great training performance on NVIDIA GPUs. Below are highlights from our benchmark results:

| Model | #GPUs | Seq Length | Model TFLOPs/sec/GPU | Tokens/sec/GPU | Kernel Optimizations |
|-------|------:|-----------:|---------------------:|---------------:|----------------------|
| DeepSeek V3 671B | 256 | 4096 | 250 | 1,002 | TE + DeepEP |
| GPT-OSS 20B | 8 | 4096 | 279 | 13,058 | TE + DeepEP + FlexAttn |
| Qwen3 MoE 30B | 8 | 4096 | 212 | 11,842 | TE + DeepEP |

For complete benchmark results including configuration details, see the [Performance Summary](docs/performance-summary.md).

<!--
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
``` -->

## 🔌 Interoperability

- **[NeMo RL](https://github.com/NVIDIA-NeMo/RL)**: Use AutoModel checkpoints directly as starting points for DPO/RM/GRPO pipelines.
- **[Hugging Face](https://github.com/huggingface/transformers)**: Train any LLM/VLM from 🤗 without format conversion.
- **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)**: Optional conversions to/from Megatron formats for specific workflows.


## 🗂️ Project Structure

```
NeMo-Automodel/
├── cli/                            # `automodel` / `am` CLI entry-point
│   └── app.py
├── docker/                         # Container build files
├── docs/                           # Documentation and guides
├── examples/
│   ├── benchmark/                  # Benchmarking scripts
│   ├── llm_finetune/              # LLM finetune YAML configs
│   ├── llm_kd/                    # LLM knowledge-distillation configs
│   ├── llm_pretrain/              # LLM pretrain configs
│   ├── llm_seq_cls/               # LLM sequence classification configs
│   ├── vlm_finetune/             # VLM finetune configs
│   └── vlm_generate/             # VLM generation configs
├── nemo_automodel/
│   ├── _transformers/             # HF model integrations
│   ├── components/                # Core library
│   │   ├── _peft/                 # PEFT implementations (LoRA)
│   │   ├── attention/             # Attention implementations
│   │   ├── checkpoint/            # Distributed checkpointing
│   │   ├── config/
│   │   ├── datasets/              # LLM (HellaSwag, etc.) & VLM datasets
│   │   ├── distributed/           # FSDP2, Megatron FSDP, Pipelining, etc.
│   │   ├── launcher/              # Launcher backends (SLURM, k8s, NeMo-Run)
│   │   ├── loggers/               # Loggers
│   │   ├── loss/                  # Optimized loss functions
│   │   ├── models/                # User-defined model examples
│   │   ├── moe/                   # Optimized kernels for MoE models
│   │   ├── optim/                 # Optimizer/LR scheduler components
│   │   ├── quantization/          # FP8
│   │   ├── training/              # Train utils
│   │   └── utils/                 # Misc utils
│   ├── recipes/
│   │   ├── llm/                   # Main LLM train loop
│   │   └── vlm/                   # Main VLM train loop
│   └── shared/
├── tools/                          # Developer tooling
└── tests/                          # Comprehensive test suite
```


## Citation
If you use NeMo AutoModel in your research, please cite it using the following BibTeX entry:
```
@misc{nemo-automodel,
title = {NeMo AutoModel: DTensor-native SPMD library for scalable and efficient training},
howpublished = {\url{https://github.com/NVIDIA-NeMo/Automodel}},
year = {2025--2026},
note = {GitHub repository},
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md) for details.


## 📄 License

NVIDIA NeMo AutoModel is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Automodel/blob/main/LICENSE).
