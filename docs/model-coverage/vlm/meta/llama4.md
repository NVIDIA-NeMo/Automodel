# Llama 4

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) is Meta's first natively multimodal model family. Llama 4 Scout and Maverick are MoE models supporting interleaved image and text inputs.

## Available Models

- **Llama-4-Scout-17B-16E-Instruct**: 17B active / 16 experts
- **Llama-4-Maverick-17B-128E-Instruct**: 17B active / 128 experts

## Architecture

- `Llama4ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Llama-4-Scout-17B-16E-Instruct | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| Llama-4-Maverick-17B-128E-Instruct | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
- https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct
