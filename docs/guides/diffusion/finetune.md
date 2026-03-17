(diffusion-finetune)=

# Diffusion Model Training and Fine-Tuning

## Introduction

NeMo AutoModel supports training and fine-tuning diffusion models using flow matching. Flow matching is a generative modeling framework that learns to transform noise into data by regressing a velocity field along straight interpolation paths. NeMo AutoModel integrates with [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) to provide distributed training for text-to-image and text-to-video models.

The `TrainDiffusionRecipe` orchestrates the full training pipeline: model loading, dataset preparation, optimizer setup, distributed training with FSDP2, flow matching loss computation, checkpointing, and logging.

### Supported Workflows

- **Fine-tuning** (`model.mode: finetune`): Loads pretrained weights and adapts them to your dataset
- **Pretraining** (`model.mode: pretrain`): Initializes random weights for training from scratch

## Prerequisites

:::{important}
Before proceeding, ensure NeMo AutoModel is installed:
```bash
pip3 install nemo-automodel
```
For additional options, see the [Installation Guide](../installation.md).
:::

Data must be preprocessed into cache files before training using the built-in preprocessing tool at `tools/diffusion/preprocessing_multiprocess.py`. See the [Diffusion Dataset Preparation](dataset.md) guide for instructions.

## Model and Dataset Context

In this guide, we use **Wan 2.1 T2V 1.3B** as the representative model. Wan 2.1 is a text-to-video diffusion model from Wan-AI that generates video from text prompts. The 1.3B variant is compact enough to train on a single node of 8 GPUs while demonstrating all key training features.

The training data consists of pre-encoded `.meta` files containing VAE latents and text embeddings. See the [dataset guide](dataset.md) for the preprocessing pipeline.

:::{tip}
Wan 2.1 is used as a representative example. The same recipe structure applies to FLUX.1-dev and HunyuanVideo with model-specific configuration changes described in [Model-Specific Notes](#model-specific-notes).
:::

## Recipe Walkthrough

The diffusion training recipe is configured through a YAML file. Below is the annotated [wan2_1_t2v_flow.yaml](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml) configuration:

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
# mode: "finetune" loads pretrained weights, "pretrain" initializes random weights
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

## Pretraining vs Fine-Tuning

The primary difference between pretraining and fine-tuning is the `model.mode` setting and the hyperparameters used:

| Setting | Fine-Tuning | Pretraining |
|---------|-------------|-------------|
| `model.mode` | `finetune` | `pretrain` |
| `learning_rate` | 5e-6 | 5e-5 |
| `weight_decay` | 0.01 | 0.1 |
| `flow_shift` | 3.0 | 2.5 |
| `logit_std` | 1.0 | 1.5 |
| Dataset size | 100s--1000s of samples | 10K+ samples |

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

## Running the Recipe

### Single-Node Fine-Tuning (8 GPUs)

```bash
torchrun --nproc-per-node=8 \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow.yaml
```

### Single-Node Pretraining (8 GPUs)

```bash
torchrun --nproc-per-node=8 \
  examples/diffusion/pretrain/pretrain.py \
  -c examples/diffusion/pretrain/wan2_1_t2v_flow.yaml
```

### Multi-Node with SLURM

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NUM_GPUS=8

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=$NUM_GPUS \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/wan2_1_t2v_flow_multinode.yaml
```

For multi-node training, adjust the FSDP config accordingly:

```yaml
fsdp:
  dp_size: 16           # 2 nodes x 8 GPUs
  dp_replicate_size: 2  # Replicate across 2 nodes
```

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
| `generate_wan_distributed.yaml` | Wan 2.2 14B | Video | 8 |
| `generate_flux.yaml` | FLUX.1-dev | Image | 1 |
| `generate_hunyuan.yaml` | HunyuanVideo | Video | 1 |

:::{note}
You can use `--model.checkpoint ./checkpoints/LATEST` to automatically load the most recent checkpoint.
:::

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | A100 40GB | A100 80GB / H100 |
| GPUs | 4 | 8+ |
| RAM | 128 GB | 256 GB+ |
| Storage | 500 GB SSD | 2 TB NVMe |
