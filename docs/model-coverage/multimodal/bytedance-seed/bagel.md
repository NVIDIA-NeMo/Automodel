# BAGEL

[BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) is a unified multimodal model from ByteDance Seed. It combines a Qwen2 language backbone, a SigLIP-NaViT vision encoder, and mixture-of-transformations layers for mixed understanding and visual-generation training.

:::{card}
| | |
|---|---|
| **Task** | Multimodal Input/Output |
| **Architecture** | `BagelForUnifiedMultimodal`, `BagelForConditionalGeneration` |
| **Parameters** | 14B (two 7B towers) |
| **HF Org** | [ByteDance-Seed](https://huggingface.co/ByteDance-Seed) |
:::

## Available Models

- **BAGEL-7B-MoT**

## Architecture

- `BagelForUnifiedMultimodal`
- `BagelForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| BAGEL-7B-MoT | [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`bagel_pretrain.yaml <../../../../examples/multimodal_pretrain/bagel/bagel_pretrain.yaml>` | BAGEL-style packed multimodal data | Joint text-understanding and image-generation pretraining |
| {download}`bagel_sft.yaml <../../../../examples/multimodal_finetune/bagel/bagel_sft.yaml>` | BAGEL-style packed multimodal data | Joint understanding + generation fine-tuning |

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/multimodal_pretrain/bagel/bagel_pretrain.yaml
```

## Hugging Face Model Cards

- [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
