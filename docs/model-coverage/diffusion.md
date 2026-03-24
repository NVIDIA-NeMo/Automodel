(diffusion-models)=

# Diffusion Models

## Introduction

Diffusion models are a class of generative models that learn to produce images or videos by iteratively denoising samples from a noise distribution. NeMo AutoModel supports training diffusion models using **flow matching**, a framework that regresses velocity fields along straight interpolation paths between noise and data.

NeMo AutoModel integrates with [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) for model loading and generation, while providing its own distributed training infrastructure via the `TrainDiffusionRecipe`. This recipe handles FSDP2 parallelization, flow matching loss computation, multiresolution bucketed dataloading, and checkpoint management.

## Supported Models

| Model | HF Model ID | Task | Parameters | Parallelization | Example YAMLs |
|-------|-------------|------|------------|-----------------|---------------|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Text-to-Video | 1.3B | FSDP2 | [finetune](../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml), [pretrain](../../examples/diffusion/pretrain/wan2_1_t2v_flow.yaml) |
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | Text-to-Image | 12B | FSDP2 | [finetune](../../examples/diffusion/finetune/flux_t2i_flow.yaml), [pretrain](../../examples/diffusion/pretrain/flux_t2i_flow.yaml) |
| HunyuanVideo 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | Text-to-Video | 13B | FSDP2 | [finetune](../../examples/diffusion/finetune/hunyuan_t2v_flow.yaml) |

## Supported Workflows

- **Pretraining**: Train from randomly initialized weights on large-scale datasets
- **Fine-tuning**: Adapt pretrained model weights to a specific dataset or style
- **Generation**: Run inference with pretrained or fine-tuned checkpoints

## Dataset

Diffusion training requires pre-encoded `.meta` files containing VAE latents and text embeddings. Raw videos or images must be preprocessed before training.

For detailed instructions on data preparation, see the [Diffusion Dataset Preparation](../guides/diffusion/dataset.md) guide.

## Train Diffusion Models

For a complete walkthrough of training configuration, model-specific settings, and launch commands, see the [Diffusion Training and Fine-Tuning Guide](../guides/diffusion/finetune.md).
