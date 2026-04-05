# Qwen2.5-VL

[Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/) is Alibaba Cloud's vision language model series supporting image and video understanding. It features dynamic resolution processing and integrates with the Qwen2.5 language backbone.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Qwen2_5VLForConditionalGeneration` |
| **Parameters** | 2B – 72B |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen2.5-VL-72B-Instruct**
- **Qwen2.5-VL-32B-Instruct**
- **Qwen2.5-VL-7B-Instruct**
- **Qwen2.5-VL-3B-Instruct**
- **Qwen2-VL-7B-Instruct**, **Qwen2-VL-2B-Instruct** (Qwen2 VL)

## Architectures

- `Qwen2_5VLForConditionalGeneration` — Qwen2.5-VL
- `Qwen2VLForConditionalGeneration` — Qwen2-VL

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen2.5-VL 3B Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Qwen2.5-VL 7B Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen2-VL 7B Instruct | `Qwen/Qwen2-VL-7B-Instruct` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| [`qwen2_5_vl_3b_rdr.yaml`](../../../examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml) | rdr-items | SFT — Qwen2.5-VL 3B on RDR Items |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
