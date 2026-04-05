# Mistral-Small-4

[Mistral-Small-4-119B](https://huggingface.co/mistralai/Mistral-Small-4-119B-Instruct-2512) is Mistral AI's multimodal MoE model supporting both text and image inputs at scale.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `MistralForConditionalGeneration` |
| **Parameters** | 119B (MoE) |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

- **Mistral-Small-4-119B-Instruct-2512**

## Architecture

- `MistralForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Mistral-Small-4 119B Instruct | `mistralai/Mistral-Small-4-119B-Instruct-2512` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| [`mistral4_medpix.yaml`](../../../examples/vlm_finetune/mistral4/mistral4_medpix.yaml) | MedPix-VQA | SFT — Mistral-Small-4 on MedPix |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [mistralai/Mistral-Small-4-119B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Small-4-119B-Instruct-2512)
