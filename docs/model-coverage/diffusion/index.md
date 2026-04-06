(diffusion-models)=

# Diffusion Models

## Introduction

Diffusion models are a class of generative models that learn to produce images or videos by iteratively denoising samples from a noise distribution. NeMo AutoModel supports training diffusion models using **flow matching**, a framework that regresses velocity fields along straight interpolation paths between noise and data.

NeMo AutoModel integrates with [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) for model loading and generation, while providing its own distributed training infrastructure via the `TrainDiffusionRecipe`. This recipe handles FSDP2 parallelization, flow matching loss computation, multiresolution bucketed dataloading, and checkpoint management.

## Supported Models

| Owner | Model | Task | Architecture |
|---|---|---|---|
| Wan AI | [Wan 2.1 T2V](wan-ai/wan2-1-t2v.md) | Text-to-Video | DiT (Flow Matching) |
| Black Forest Labs | [FLUX.1-dev](black-forest-labs/flux.md) | Text-to-Image | DiT (Flow Matching) |
| Hunyuan Community | [HunyuanVideo 1.5](hunyuanvideo-community/hunyuanvideo.md) | Text-to-Video | DiT (Flow Matching) |

## Supported Workflows

- **Pretraining**: Train from randomly initialized weights on large-scale datasets
- **Fine-tuning**: Adapt pretrained model weights to a specific dataset or style
- **Generation**: Run inference with pretrained or fine-tuned checkpoints

## Dataset

Diffusion training requires pre-encoded `.meta` files containing VAE latents and text embeddings. Raw videos or images must be preprocessed before training. See the [Diffusion Dataset Preparation](../../guides/diffusion/dataset.md) guide.

## Train Diffusion Models

For a complete walkthrough of training configuration, model-specific settings, and launch commands, see the [Diffusion Training and Fine-Tuning Guide](../../guides/diffusion/finetune.md).

```{toctree}
:hidden:

wan-ai/wan2-1-t2v
black-forest-labs/flux
hunyuanvideo-community/hunyuanvideo
```
