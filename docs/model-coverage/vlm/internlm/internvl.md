# InternVL

[InternVL](https://github.com/OpenGVLab/InternVL) is a vision language model from Shanghai AI Laboratory (OpenGVLab), combining a large vision encoder with an InternLM language backbone for strong multimodal performance.

## Available Models

- **InternVL3.5-4B**
- **InternVL3.5-8B**

## Architecture

- `InternVLForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| InternVL3.5 4B | `OpenGVLab/InternVL3-5-4B` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| [`internvl_3_5_4b.yaml`](../../../examples/vlm_finetune/internvl/internvl_3_5_4b.yaml) | MedPix-VQA | SFT — InternVL3.5 4B on MedPix |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- https://huggingface.co/OpenGVLab/InternVL3-5-4B
