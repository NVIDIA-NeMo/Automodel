# Wan 2.1 T2V

[Wan 2.1](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) is a text-to-video diffusion model from Wan AI, trained with flow matching on a large-scale video dataset. It generates high-quality short video clips from text prompts.

:::{card}
| | |
|---|---|
| **Task** | Text-to-Video |
| **Architecture** | DiT (Flow Matching) |
| **Parameters** | 1.3B |
| **HF Org** | [Wan-AI](https://huggingface.co/Wan-AI) |
:::

## Available Models

- **Wan2.1-T2V-1.3B**: 1.3B parameters

## Task

- Text-to-Video (T2V)

## Example HF Models

| Model | HF ID |
|---|---|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`wan2_1_t2v_flow.yaml`](../../../examples/diffusion/finetune/wan2_1_t2v_flow.yaml) | Fine-tune — Wan 2.1 T2V with flow matching |
| [`wan2_1_t2v_flow.yaml`](../../../examples/diffusion/pretrain/wan2_1_t2v_flow.yaml) | Pretrain — Wan 2.1 T2V with flow matching |

## Training

See the [Diffusion Training and Fine-Tuning Guide](../../../guides/diffusion/finetune.md) and [Dataset Preparation](../../../guides/diffusion/dataset.md).

## Hugging Face Model Cards

- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
