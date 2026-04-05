# FLUX.1-dev

[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is a 12B parameter text-to-image diffusion transformer from Black Forest Labs, trained with flow matching. It produces high-fidelity images and is designed for non-commercial research and development use.

:::{card}
| | |
|---|---|
| **Task** | Text-to-Image |
| **Architecture** | DiT (Flow Matching) |
| **Parameters** | 12B |
| **HF Org** | [black-forest-labs](https://huggingface.co/black-forest-labs) |
:::

## Available Models

- **FLUX.1-dev**: 12B parameters

## Task

- Text-to-Image (T2I)

## Example HF Models

| Model | HF ID |
|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`flux_t2i_flow.yaml <../../../examples/diffusion/finetune/flux_t2i_flow.yaml>` | Fine-tune — FLUX.1-dev with flow matching |
| {download}`flux_t2i_flow.yaml <../../../examples/diffusion/pretrain/flux_t2i_flow.yaml>` | Pretrain — FLUX.1-dev with flow matching |

## Training

See the [Diffusion Training and Fine-Tuning Guide](../../../guides/diffusion/finetune.md) and [Dataset Preparation](../../../guides/diffusion/dataset.md).

## Hugging Face Model Cards

- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
