# Qwen3.5-VL

Qwen3.5-VL is Alibaba Cloud's next-generation vision language model series, including dense and MoE variants for image and multimodal understanding tasks.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Qwen3_5VLForConditionalGeneration` |
| **Parameters** | 4B – 35B+ |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3.5-VL-4B**: 4B dense model
- **Qwen3.5-VL-9B**: 9B dense model
- **Qwen3.5-MoE**: large MoE variant (35B+)

## Architectures

- `Qwen3_5VLForConditionalGeneration` — dense models
- `Qwen3_5MoeVLForConditionalGeneration` — MoE variant

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| [`qwen3_5_4b.yaml`](../../../examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml) | MedPix-VQA | SFT — Qwen3.5-VL 4B on MedPix |
| [`qwen3_5_9b.yaml`](../../../examples/vlm_finetune/qwen3_5/qwen3_5_9b.yaml) | MedPix-VQA | SFT — Qwen3.5-VL 9B on MedPix |
| [`qwen3_5_moe_medpix.yaml`](../../../examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml) | MedPix-VQA | SFT — Qwen3.5-MoE on MedPix |
| [`qwen3_5_35b.yaml`](../../../examples/vlm_finetune/qwen3_5_moe/qwen3_5_35b.yaml) | MedPix-VQA | SFT — Qwen3.5 35B on MedPix |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen](https://huggingface.co/Qwen)
