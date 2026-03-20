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
| **5. Generate** | [Generation / Inference](#generation--inference) | Run inference using the fine-tuned checkpoint |

For model-specific configuration (FLUX.1-dev, HunyuanVideo), see [Model-Specific Notes](#model-specific-notes).

### Supported Models

| Model | HF Model ID | Task | Parameters | Example Config |
|-------|-------------|------|------------|----------------|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-Video | 1.3B | [wan2_1_t2v_flow.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml) |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | Text-to-Image | 12B | [flux_t2i_flow.yaml](../../../examples/diffusion/finetune/flux_t2i_flow.yaml) |
| HunyuanVideo 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | Text-to-Video | — | [hunyuan_t2v_flow.yaml](../../../examples/diffusion/finetune/hunyuan_t2v_flow.yaml) |

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

**Video preprocessing (Wan 2.1):**
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

Fine-tuning is configured through a YAML file. Below is the annotated [wan2_1_t2v_flow.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml) configuration:

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
    model_type: wan
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

(model-specific-notes)=
## Model-Specific Notes

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

## Generation / Inference

Use the unified generation script to run inference with any supported diffusion model:

**Single-GPU (Wan 2.1 1.3B):**
```bash
python examples/diffusion/generate/generate.py \
  -c examples/diffusion/generate/configs/generate_wan.yaml
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

### Available Generation Configs

| Config | Model | Output | GPUs |
|--------|-------|--------|------|
| `generate_wan.yaml` | Wan 2.1 1.3B | Video | 1 |
| `generate_flux.yaml` | FLUX.1-dev | Image | 1 |
| `generate_hunyuan.yaml` | HunyuanVideo | Video | 1 |

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
