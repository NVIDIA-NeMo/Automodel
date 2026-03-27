(diffusion-finetune)=

# Diffusion Model Fine-Tuning with NeMo AutoModel

## Introduction

Diffusion models generate images and videos by learning to reverse a noise process — starting from random noise and iteratively refining it into coherent visual output guided by a text prompt. Pretrained diffusion models (like [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) for images or [Wan 2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) for video) produce impressive general-purpose results, but they know nothing about your particular visual domain, style, or subject matter. Fine-tuning bridges that gap — you adapt the model on your own data so it produces outputs that match your requirements, without the cost of training from scratch.

Under the hood, NeMo AutoModel uses [flow matching](https://arxiv.org/abs/2210.02747), a modern generative framework that learns to transform noise into data by regressing a velocity field along straight interpolation paths. It integrates with [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) to provide distributed fine-tuning for text-to-image and text-to-video models. This guide walks you through the process end-to-end — from installation through training and inference — using [Wan 2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) as a running example.

### Workflow Overview

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Install   │--->│ 2. Prepare   │--->│ 3. Configure │--->│  4. Train    │--->│ 5. Generate  │
│              │    │    Data      │    │              │    │              │    │              │
│ pip install  │    │ Encode to    │    │ YAML recipe  │    │ torchrun     │    │ Run inference│
│ or Docker    │    │ .meta files  │    │              │    │              │    │ with ckpt    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

| Step | Section | What You Do |
|------|---------|-------------|
| **1. Install** | [Install NeMo AutoModel](#install-nemo-automodel) | Install the package via pip or Docker |
| **2. Prepare Data** | [Prepare Your Dataset](#prepare-your-dataset) | Encode raw images/videos into `.meta` latent files |
| **3. Configure** | [Configure Your Training Recipe](#configure-your-training-recipe) | Write a YAML config specifying model, data, and training settings |
| **4. Train** | [Fine-Tune the Model](#fine-tune-the-model) | Launch training with `torchrun` on a single node |
| **4b. Multi-Node** | [Multi-Node Training](#multi-node-training) | Scale training across multiple nodes |
| **5. Generate** | [Generation / Inference](#generation--inference) | Run inference using the fine-tuned checkpoint |

For model-specific configuration (FLUX.1-dev, HunyuanVideo), see [Model-Specific Notes](#model-specific-notes).

### Supported Models

| Model | HF Model ID | Task | Parameters | Example Config |
|-------|-------------|------|------------|----------------|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-Video | 1.3B | [wan2_1_t2v_flow.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml) |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | Text-to-Image | 12B | [flux_t2i_flow.yaml](../../../examples/diffusion/finetune/flux_t2i_flow.yaml) |
| HunyuanVideo 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | Text-to-Video | — | [hunyuan_t2v_flow.yaml](../../../examples/diffusion/finetune/hunyuan_t2v_flow.yaml) |
| LTX-Video | `Lightricks/LTX-Video` | Text-to-Video | 2B–22B | [ltx_t2v_flow.yaml](../../../examples/diffusion/finetune/ltx_t2v_flow.yaml) |

All models use FSDP2 for distributed training and flow matching for loss computation.

## Install NeMo AutoModel

```bash
pip3 install nemo-automodel
```

Alternatively, if you run into dependency or driver issues, use the pre-built Docker container:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02.00
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:26.02.00
```

:::{important}
**Docker users:** Checkpoints are lost when the container exits unless you bind-mount the checkpoint directory to the host. See [Install with NeMo Docker Container](../installation.md#install-with-nemo-docker-container) and [Saving Checkpoints When Using Docker](../checkpointing.md#saving-checkpoints-when-using-docker).
:::

For the full set of installation methods, see the [installation guide](../installation.md).

## Prepare Your Dataset

Diffusion models operate in latent space — a compressed representation of visual data — rather than directly on raw images or videos. To avoid re-encoding data on every training step, the preprocessing
  pipeline encodes all inputs ahead of time and saves them as .meta files.

 Each .meta file contains:
 - Latent representations produced by a VAE (Variational Autoencoder) from the raw visual data
 - Text embeddings produced by a text encoder from the associated captions/prompts

Fine-tuning then operates entirely on these pre-encoded .meta files, which is significantly faster than encoding on the fly.

Preprocess your data using the built-in tool at [`tools/diffusion/preprocessing_multiprocess.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/tools/diffusion/preprocessing_multiprocess.py). The script provides `image` and `video` subcommands:

**Video preprocessing (using Wan 2.1 as a running example):**
```bash
python -m tools.diffusion.preprocessing_multiprocess video \
    --video_dir /data/videos \
    --output_dir /cache \
    --processor wan \
    --resolution_preset 512p \
    --caption_format sidecar
```

**Image preprocessing (FLUX):**
```bash
python -m tools.diffusion.preprocessing_multiprocess image \
    --image_dir /data/images \
    --output_dir /cache \
    --processor flux
```

**Video preprocessing (HunyuanVideo):**
```bash
python -m tools.diffusion.preprocessing_multiprocess video \
    --video_dir /data/videos \
    --output_dir /cache \
    --processor hunyuan \
    --target_frames 121 \
    --caption_format meta_json
```

For the full set of arguments and input format details, see the [Diffusion Dataset Preparation](dataset.md) guide.

## Configure Your Training Recipe

Fine-tuning is driven by two components:

1. A recipe script (e.g., [`train.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/diffusion/train.py)) — the Python entry point that orchestrates the training loop: loading the model, building the dataloader, running forward/backward passes, computing the flow matching loss, checkpointing, and logging.
2. A YAML configuration file — a text file in YAML format that specifies all settings the recipe uses: which model to fine-tune, where the data lives, optimizer hyperparameters, parallelism strategy, etc.
  You customize training by editing this file rather than modifying code, allowing you to scale from 1 to 100s of GPUs seamlessly.

Below is the annotated [wan2_1_t2v_flow.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml), with each section explained:

```yaml
seed: 42

# Weights & Biases experiment tracking
wandb:
  project: wan-t2v-flow-matching
  mode: online
  name: wan2_1_t2v_fm_v2

dist_env:
  backend: nccl
  timeout_minutes: 30

# Model configuration
# pretrained_model_name_or_path: Hugging Face model ID
# mode: "finetune" loads pretrained weights and adapts them to your dataset
model:
  pretrained_model_name_or_path: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  mode: finetune

# Training schedule
step_scheduler:
  global_batch_size: 8       # Effective batch size across all GPUs
  local_batch_size: 1        # Per-GPU batch size (gradient accumulation = global/local/num_gpus)
  ckpt_every_steps: 1000     # Checkpoint frequency
  num_epochs: 100
  log_every: 2               # Log metrics every N steps

# Data: uses pre-encoded .meta files
data:
  dataloader:
    _target_: nemo_automodel.components.datasets.diffusion.build_video_multiresolution_dataloader
    cache_dir: PATH_TO_YOUR_DATA
    model_type: wan # "wan" for Wan 2.1, "hunyuan" for HunyuanVideo
    base_resolution: [512, 512]
    dynamic_batch_size: false
    shuffle: true
    drop_last: false
    num_workers: 0

# Optimizer
optim:
  learning_rate: 5e-6
  optimizer:
    weight_decay: 0.01
    betas: [0.9, 0.999]

# Learning rate scheduler
lr_scheduler:
  lr_decay_style: cosine
  lr_warmup_steps: 0
  min_lr: 1e-6

# Flow matching configuration
flow_matching:
  adapter_type: "simple"          # Model-specific adapter (simple, flux, hunyuan)
  adapter_kwargs: {}
  timestep_sampling: "uniform"    # How timesteps are sampled during training
  logit_mean: 0.0
  logit_std: 1.0
  flow_shift: 3.0                # Shifts the flow schedule
  mix_uniform_ratio: 0.1
  sigma_min: 0.0
  sigma_max: 1.0
  num_train_timesteps: 1000
  i2v_prob: 0.3                  # Probability of image-to-video conditioning
  use_loss_weighting: true
  log_interval: 100
  summary_log_interval: 10

# FSDP2 distributed training
fsdp:
  tp_size: 1      # Tensor parallelism
  cp_size: 1      # Context parallelism
  pp_size: 1      # Pipeline parallelism
  dp_replicate_size: 1
  dp_size: 8      # Data parallelism (number of GPUs)

# Checkpointing
checkpoint:
  enabled: true
  checkpoint_dir: PATH_TO_YOUR_CKPT_DIR
  model_save_format: torch_save
  save_consolidated: false
  restore_from: null
```

### Config Field Reference

| Section | Required? | What to Change |
|---------|-----------|----------------|
| `model` | Yes | Set `pretrained_model_name_or_path` to the Hugging Face model ID. Set `mode: finetune`. |
| `step_scheduler` | Yes | `global_batch_size` is the effective batch size across all GPUs. `ckpt_every_steps` controls checkpoint frequency. |
| `data` | Yes | Set `cache_dir` to the path containing your preprocessed `.meta` files. Change `model_type` and `_target_` for different models (see [Model-Specific Notes](#model-specific-notes)). |
| `optim` | Yes | `learning_rate: 5e-6` is a good default for fine-tuning. |
| `flow_matching` | Yes | `adapter_type` must match the model (`simple` for Wan, `flux` for FLUX, `hunyuan` for HunyuanVideo). |
| `fsdp` | Yes | Set `dp_size` to the number of GPUs on your node. |
| `checkpoint` | Recommended | Set `checkpoint_dir` to a persistent path, especially in Docker. |
| `wandb` | Optional | Configure to enable Weights & Biases logging. |

(fine-tune-the-model)=
## Fine-Tune the Model

Launch fine-tuning with `torchrun`:

```bash
torchrun --nproc-per-node=8 \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow.yaml
```

Adjust `--nproc-per-node` to match the number of GPUs on your node, and ensure `fsdp.dp_size` in the YAML matches.

(multi-node-training)=
## Multi-Node Training

When a single node doesn't provide enough GPUs or memory for your workload, you can scale training across multiple nodes. NeMo AutoModel handles multi-node distributed training through `torchrun` rendezvous and FSDP2 — the same recipe script works on one node or many.

### YAML Configuration Changes

The main change is in the `fsdp` section. Set `dp_size` to the **total number of GPUs across all nodes**, and optionally increase `dp_replicate_size` for gradient replication across nodes.

For example, to train on 2 nodes with 8 GPUs each (16 GPUs total):

```yaml
fsdp:
  tp_size: 1
  cp_size: 1
  pp_size: 1
  dp_replicate_size: 2   # Replicate across 2 nodes for robustness
  dp_size: 16             # Total GPUs: 2 nodes × 8 GPUs
```

A complete multi-node config is provided at [wan2_1_t2v_flow_multinode.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow_multinode.yaml).

### Launch with torchrun

Run the following command on **each node**, setting `NODE_RANK` to `0` on the first node, `1` on the second, and so on:

```bash
export MASTER_ADDR=node0.hostname   # hostname or IP of the first node
export MASTER_PORT=29500
export NODE_RANK=0                  # 0 on master, 1 on second node, etc.

torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow_multinode.yaml
```

(model-specific-notes)=
## Model-Specific Notes

Use the table below to pick the right model for your use case:

| Use Case | Model | Why Choose It |
|----------|-------|---------------|
| **Video generation on limited hardware** | [Wan 2.1 T2V 1.3B](#wan-21-t2v-13b) | Smallest model (1.3B params) — fast iteration, fits on a single A100 40GB |
| **High-quality image generation** | [FLUX.1-dev](#flux1-dev-text-to-image) | State-of-the-art text-to-image with 12B params and guidance-based control |
| **High-quality video generation** | [HunyuanVideo 1.5](#hunyuanvideo-15) | Larger video model with condition-latent support for richer motion and detail |
| **Fast video generation** | [LTX-Video](#ltx-video) | DiT-based video model with high compression ratio VAE (1:192) for efficient generation |

### Wan 2.1 T2V 1.3B

- **Adapter type**: `simple`
- **Dataloader**: `build_video_multiresolution_dataloader` with `model_type: wan`
- **Config**: [wan2_1_t2v_flow.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml)

### FLUX.1-dev (Text-to-Image)

- **Adapter type**: `flux`
- **Dataloader**: `build_text_to_image_multiresolution_dataloader`
- **Key differences**:
  - Uses `pipeline_spec` to specify the transformer architecture:
    ```yaml
    model:
      pipeline_spec:
        transformer_cls: "FluxTransformer2DModel"
        subfolder: "transformer"
        load_full_pipeline: false
    ```
  - Requires `guidance_scale` in adapter kwargs:
    ```yaml
    flow_matching:
      adapter_type: "flux"
      adapter_kwargs:
        guidance_scale: 3.5
        use_guidance_embeds: true
    ```
  - Uses `logit_normal` timestep sampling instead of `uniform`
- **Config**: [flux_t2i_flow.yaml](../../../examples/diffusion/finetune/flux_t2i_flow.yaml)

### HunyuanVideo 1.5

- **Adapter type**: `hunyuan`
- **Dataloader**: `build_video_multiresolution_dataloader` with `model_type: hunyuan`
- **Key differences**:
  - Requires `activation_checkpointing: true` in FSDP config due to model size
  - Uses condition latents in adapter kwargs:
    ```yaml
    flow_matching:
      adapter_type: "hunyuan"
      adapter_kwargs:
        use_condition_latents: true
        default_image_embed_shape: [729, 1152]
    ```
  - Uses `logit_normal` timestep sampling
- **Config**: [hunyuan_t2v_flow.yaml](../../../examples/diffusion/finetune/hunyuan_t2v_flow.yaml)

### LTX-Video

- **Adapter type**: `ltx`
- **Dataloader**: `build_video_multiresolution_dataloader` with `model_type: ltx`
- **Key differences**:
  - Frame count must satisfy **8n+1** constraint (9, 17, 25, 33, ...)
  - Resolution must be divisible by **32** (not 16 like Wan)
  - Uses 3D latent packing/unpacking for the DiT transformer
  - VAE has 128 latent channels with 8x8x8 spatio-temporal compression
  - Uses `logit_normal` timestep sampling
  - Adapter accepts `patch_size` and `patch_size_t` kwargs (default 1):
    ```yaml
    flow_matching:
      adapter_type: "ltx"
      adapter_kwargs:
        patch_size: 1
        patch_size_t: 1
    ```
- **Config**: [ltx_t2v_flow.yaml](../../../examples/diffusion/finetune/ltx_t2v_flow.yaml)

## Generation / Inference

Once training is complete, you can use the model to generate images or videos from text prompts. This step is called inference — as opposed to training, where the model learns from data, inference is where it produces new outputs.

In diffusion models, generation works by starting from random noise and iteratively denoising it, guided by your text prompt, until a clean image or video emerges.

The generation script ([`generate.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/diffusion/generate/generate.py)) handles this: it loads your model weights (pretrained or fine-tuned), configures the diffusion sampler, and produces outputs for one or more prompts.

**Single-GPU (Wan 2.1 1.3B):**
```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_wan.yaml
```

**Multi-GPU (Wan 2.1 1.3B):**

Wan 2.1 supports tensor parallelism for inference, which shards the transformer across GPUs to reduce per-GPU memory. Pass the `distributed` config via CLI overrides:

```bash
torchrun --nproc-per-node=8 \
  examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_wan.yaml \
  --distributed.backend nccl \
  --distributed.parallel_scheme.transformer.tp_size 8
```

**With a fine-tuned checkpoint:**
```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_wan.yaml \
  --model.checkpoint ./checkpoints/step_1000 \
  --inference.prompts '["A dog running on a beach"]'
```

**FLUX image generation:**
```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_flux.yaml
```

**HunyuanVideo:**
```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_hunyuan.yaml
```

**LTX-Video:**
```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_ltx.yaml
```

### Available Generation Configs

| Config | Model | Output | GPUs |
|--------|-------|--------|------|
| [`generate_wan.yaml`](../../../examples/diffusion/generate/configs/generate_wan.yaml) | Wan 2.1 1.3B | Video | 1 |
| [`generate_flux.yaml`](../../../examples/diffusion/generate/configs/generate_flux.yaml) | FLUX.1-dev | Image | 1 |
| [`generate_hunyuan.yaml`](../../../examples/diffusion/generate/configs/generate_hunyuan.yaml) | HunyuanVideo | Video | 1 |
| [`generate_ltx.yaml`](../../../examples/diffusion/generate/configs/generate_ltx.yaml) | LTX-Video | Video | 1 |

:::{note}
You can use `--model.checkpoint ./checkpoints/LATEST` to automatically load the most recent checkpoint.
:::

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A100 40GB | A100 80GB / H100 |
| GPUs | 4 | 8 |
| RAM | 128 GB | 256 GB+ |
| Storage | 500 GB SSD | 2 TB NVMe |
