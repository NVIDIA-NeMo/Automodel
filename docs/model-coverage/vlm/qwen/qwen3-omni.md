# Qwen3-Omni

[Qwen3-Omni](https://qwenlm.github.io/blog/qwen3/) is Alibaba Cloud's omnimodal model supporting text, image, audio, and video inputs in a single unified architecture.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Qwen3OmniForConditionalGeneration` |
| **Parameters** | 30B total / 3B active |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3-Omni-30B-A3B**: 30B total, 3B activated (MoE)

## Architecture

- `Qwen3OmniForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen3-Omni 30B A3B | `Qwen/Qwen3-Omni-30B-A3B` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`qwen3_omni_moe_30b_te_deepep.yaml <../../../examples/vlm_finetune/qwen3/qwen3_omni_moe_30b_te_deepep.yaml>` | MedPix-VQA | SFT — Qwen3-Omni 30B with TE + DeepEP |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen/Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B)
