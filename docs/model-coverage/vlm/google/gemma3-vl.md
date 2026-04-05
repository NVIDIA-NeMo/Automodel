# Gemma 3 VL / Gemma 3n

[Gemma 3 VL](https://ai.google.dev/gemma/docs/gemma3) is Google's multimodal extension of Gemma 3, supporting image-text inputs for tasks like image captioning and visual question answering. Gemma 3n is a next-generation efficiency-focused variant.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Gemma3ForConditionalGeneration` |
| **Parameters** | 4B – 27B |
| **HF Org** | [google](https://huggingface.co/google) |
:::

## Available Models

- **Gemma 3 27B IT** (VL)
- **Gemma 3 4B IT** (VL)
- **Gemma 3n 4B** (VL)

## Architecture

- `Gemma3ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Gemma 3 4B IT | `google/gemma-3-4b-it` |
| Gemma 3 27B IT | `google/gemma-3-27b-it` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| [`gemma3_vl_4b_cord_v2.yaml`](../../../examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml) | cord-v2 | SFT — Gemma 3 4B VL on CORD-v2 |
| [`gemma3_vl_4b_cord_v2_peft.yaml`](../../../examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml) | cord-v2 | LoRA — Gemma 3 4B VL on CORD-v2 |
| [`gemma3_vl_4b_cord_v2_megatron_fsdp.yaml`](../../../examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_megatron_fsdp.yaml) | cord-v2 | SFT — Gemma 3 4B VL with MegatronFSDP |
| [`gemma3_vl_4b_medpix.yaml`](../../../examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix.yaml) | MedPix-VQA | SFT — Gemma 3 4B VL on MedPix |
| [`gemma3n_vl_4b_medpix.yaml`](../../../examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix.yaml) | MedPix-VQA | SFT — Gemma 3n 4B VL on MedPix |
| [`gemma3n_vl_4b_medpix_peft.yaml`](../../../examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix_peft.yaml) | MedPix-VQA | LoRA — Gemma 3n 4B VL on MedPix |

## Fine-Tuning

See the [Gemma 3 & Gemma 3n Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md) for detailed instructions on dataset preparation, configuration, and multi-GPU training.

## Hugging Face Model Cards

- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)
