<div align="center">

# 🚀 NeMo AutoModel

</div>

<div align="center">

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->
[![codecov](https://codecov.io/github/NVIDIA-NeMo/Automodel/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Automodel)
[![CICD NeMo](https://github.com/NVIDIA-NeMo/Automodel/actions/workflows/cicd-main.yml/badge.svg)](https://github.com/NVIDIA-NeMo/Automodel/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Automodel.svg?style=social&label=Star)](https://github.com/NVIDIA-NeMo/Automodel/stargazers/)

<!-- **Day-0 integration with Hugging Face models automating fine-tuning and pretraining with pytorch-native parallelism, custom-kernels and optimized recipes**
**Pytorch DTensor‑native SPMD library for large‑scale training**-->

[📖 Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html) • [🔥 Ready-to-Use Recipes](https://github.com/NVIDIA-NeMo/Automodel/#supported-models) • [💡 Examples](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples) • [🤝 Contributing](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md)

</div>

## 📣 News and Discussions
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

- **Hackable** with a modular design that allows easy integration, customization and quick research prototypes.
- **Minimal ceremony**: YAML‑driven recipes; override any field via CLI.
- **High performance and flexibility** with custom kernels and DTensor support.
- **Seamless integration** with Hugging Face for day-0 model support, ease of use, and wide range of supported models.
- **Efficient resource management** using k8s and Slurm, enabling scalable and flexible deployment across configurations.
- **Comprehensive documentation** that is both detailed and user-friendly, with practical examples.

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

✅ _Available now_ | 🔜 _Coming in 25.11_

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


- 🔜 **Extended MoE support** - GPT-OSS, Qwen3 (Coder-480B-A35B, etc), Qwen-next.
- 🔜 **Kubernetes** - MUlti-node job launch with k8s.


## Getting Started

We recommend using **uv** for reproducible Python environments.

```bash
# Setup environment before running any commands
uv venv
uv sync --frozen --all-extras

uv pip install nemo_automodel # latest release
# or: uv pip install git+https://github.com/NVIDIA-NeMo/Automodel.git
uv run python -c "import nemo_automodel; print('AutoModel ready')"
```


### Run a Recipe
To run a NeMo AutoModel recipe, you need a recipe script (e.g., [LLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/finetune.py), [VLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/finetune.py)) and a YAML config file (e.g., [LLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama/llama3_2_1b_squad.yaml), [VLM](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml)):
```
# Command invocation format:
uv run <recipe_script_path> --config <yaml_config_path>

# LLM example: multi-GPU with FSDP2
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml

# VLM example: single GPU fine-tuning (Gemma-3-VL) with LoRA
uv run examples/vlm_finetune/finetune.py --config examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml
```


## LLM Pre-training
### LLM Pre-training Single Node
We provide an example SFT experiment using the [Fineweb dataset](https://arxiv.org/abs/2406.17557/) with a nano-GPT model, ideal for quick experimentation on a single node.
```sh
uv run torchrun --nproc-per-node=8 \
  examples/llm_pretrain/pretrain.py \
  -c examples/llm_pretrain/nanogpt_pretrain.yaml
```

<!-- ### LLM Pre-training Multi Node -->

## LLM Supervised Fine-Tuning (SFT)
We provide an example SFT experiment using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).

<!-- Refer to `examples/llm_finetune/annotated.yaml` for a full list of parameters that can be overridden. -->

### LLM SFT Single Node

The default SFT configuration is set to run on a single GPU. To start the experiment:

```sh
uv run python3 \
  examples/llm_finetune/finetune.py \
  -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

This fine-tunes the `Llama3.2-1B` model on the SQuAD dataset using a 1 GPU.

To use multiple GPUs on a single node in an interactive environment, you can run the same command
using torchrun and adjust the `--proc-per-node` argument to the number of needed GPUs.

```sh
uv run torchrun --nproc-per-node=8 \
  examples/llm_finetune/finetune.py \
  -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

Alternatively, you can use the `automodel` CLI application to launch the same job, for example:
```sh
uv run automodel finetune llm \
  --nproc-per-node=8 \
  -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

### LLM SFT Multi Node
You can use the `automodel` CLI application to launch a job on a SLURM cluster, for example:
```sh
# First you need to specify the SLURM section in your YAML config, for example:

cat << EOF > examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
slurm:
  job_name: llm-finetune # set to the job name you want to use
  nodes: 2 # set to the needed number of nodes
  ntasks_per_node: 8
  time: 00:30:00
  account: your_account
  partition: gpu
  container_image: nvcr.io/nvidia/nemo:25.07
  gpus_per_node: 8 # This adds "#SBATCH --gpus-per-node=8" to the script
  # Optional: Add extra mount points if needed
  extra_mounts:
    - /lustre:/lustre
  # Optional: Specify custom HF_HOME location (will auto-create if not specified)
  hf_home: /path/to/your/HF_HOME
  # Optional : Specify custom env vars
  # env_vars:
  #   ENV_VAR: value
  # Optional: Specify custom job directory (defaults to cwd/slurm_jobs)
  # job_dir: /path/to/slurm/jobs
EOF

# using the updated YAML you can launch the job.
uv run automodel finetune llm \
  -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

## LLM Parameter-Efficient Fine-Tuning (PEFT)

We provide a PEFT example using the [HellaSwag dataset](https://rowanzellers.com/hellaswag/).

### LLM PEFT Single Node
```bash
# Memory‑efficient SFT with LoRA
uv run examples/llm_finetune/finetune.py \
--config examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml

# You can always overwrite parameters by appending them to the command, for example,
# if you want to increase the micro-batch size you can do
uv run examples/llm_finetune/finetune.py \
  --config examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml \
  --step_scheduler.local_batch_size 16

# The above command will modify the `local_batch_size` variable to have value 16 in the
# section `step_scheduler` of the yaml file.
```

> [!NOTE]
> Launching a multi-node PEFT example requires only adding a `slurm` section to your config, similarly to the SFT case.


## VLM Supervised Fine-Tuning (SFT)

We provide a VLM SFT example using Qwen2.5‑VL for end‑to‑end fine‑tuning on image‑text data.

### VLM SFT Single Node
```bash
# Qwen2.5‑VL on a 8 GPUs
uv run torchrun --nproc-per-node=8 \
  examples/vlm_finetune/finetune.py \
  --config examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml
```

## VLM Parameter-Efficient Fine-Tuning (PEFT)

We provide a VLM PEFT (LoRA) example for memory‑efficient adaptation with Gemma3 VLM.

### VLM PEFT Single Node
```bash
# Qwen2.5‑VL on a 8 GPUs
uv run torchrun --nproc-per-node=8 \
  examples/vlm_finetune/finetune.py \
  --config examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix_peft.yaml
```


## Supported Models
NeMo AutoModel provides native support for a wide range of models available on the Hugging Face Hub, enabling efficient fine-tuning for various domains. Below is a small sample of ready‑to‑use families (train as‑is or swap any compatible 🤗 causal LM), you can specify nearly any LLM/VLM model available on 🤗 hub:

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
├── examples
│   ├── llm_finetune            # LLM finetune recipes
│   ├── llm_kd                  # LLM knowledge-distillation recipes
│   ├── llm_pretrain            # LLM pretrain recipes
│   ├── vlm_finetune            # VLM finetune recipes
│   └── vlm_generate            # VLM generate recipes
├── nemo_automodel
│   ├── _cli
│   │   └── app.py              # the `automodel` CLI job launcher
│   ├── components              # Core library
│   │   ├── _peft               # PEFT implementations (LoRA)
│   │   ├── _transformers       # HF model integrations
│   │   ├── checkpoint          # Distributed checkpointing
│   │   ├── config
│   │   ├── datasets            # LLM (HellaSwag, etc.) & VLM datasets
│   │   ├── distributed         # FSDP2, Megatron FSDP, Pipelining, etc.
│   │   ├── launcher            # The job launcher component (SLURM)
│   │   ├── loggers             # loggers
│   │   ├── loss                # Optimized loss functions
│   │   ├── models              # User-defined model examples
│   │   ├── moe                 # Optimized kernels for MoE models
│   │   ├── optim               # Optimizer/LR scheduler components
│   │   ├── quantization        # FP8
│   │   ├── training            # Train utils
│   │   └── utils               # Misc utils
│   ├── recipes
│   │   ├── llm                 # Main LLM train loop
│   │   └── vlm                 # Main VLM train loop
│   └── shared
└── tests/                      # Comprehensive test suite
```


## Citation
If you use NeMo AutoModel in your research, please cite it using the following BibTeX entry:
```
@misc{nemo-automodel,
title = {NeMo AutoModel: DTensor‑native SPMD library for scalable and efficient training},
howpublished = {\url{https://github.com/NVIDIA-NeMo/Automodel}},
year = {2025},
note = {GitHub repository},
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/CONTRIBUTING.md) for details.


## 📄 License

NVIDIA NeMo AutoModel is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Automodel/blob/main/LICENSE).
