# OLMoE

[OLMoE](https://allenai.org/olmo) is Allen AI's open Mixture-of-Experts language model. It activates 1B parameters per token from a 7B total parameter pool.

## Available Models

- **OLMoE-1B-7B-0924**: 7B total, 1B activated
- **OLMoE-1B-7B-0924-Instruct**: instruction-tuned variant

## Architecture

- `OLMoEForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| OLMoE 1B 7B | `allenai/OLMoE-1B-7B-0924` |
| OLMoE 1B 7B Instruct | `allenai/OLMoE-1B-7B-0924-Instruct` |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/allenai/OLMoE-1B-7B-0924
