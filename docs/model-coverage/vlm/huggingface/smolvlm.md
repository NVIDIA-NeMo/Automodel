# SmolVLM

[SmolVLM](https://huggingface.co/blog/smolvlm) is HuggingFace's compact vision language model designed for on-device and memory-constrained deployment, featuring an efficient image token compression strategy.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `SmolVLMForConditionalGeneration` |
| **Parameters** | 256M – 2B |
| **HF Org** | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) |
:::

## Available Models

- **SmolVLM-Instruct**: 2B
- **SmolVLM-256M-Instruct**: 256M

## Architecture

- `SmolVLMForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| SmolVLM Instruct | `HuggingFaceTB/SmolVLM-Instruct` |
| SmolVLM 256M Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct` |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
