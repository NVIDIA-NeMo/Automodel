# Granite MoE

IBM Granite MoE models extend the Granite architecture with Mixture-of-Experts layers for more efficient scaling. PowerMoE (IBM Research) also uses this architecture.

## Available Models

- **Granite 3.0 1B A400M Base** — 1B total, 400M activated
- **Granite 3.0 3B A800M Instruct** — 3B total, 800M activated
- **PowerMoE-3B** (IBM Research) — 3B total
- **MoE-7B-1B-Active-Shared-Experts** (IBM Research, test model)

## Architectures

- `GraniteMoeForCausalLM`
- `GraniteMoeSharedForCausalLM` — variant with shared experts

## Example HF Models

| Model | HF ID |
|---|---|
| Granite 3.0 1B A400M Base | `ibm-granite/granite-3.0-1b-a400m-base` |
| Granite 3.0 3B A800M Instruct | `ibm-granite/granite-3.0-3b-a800m-instruct` |
| PowerMoE 3B | `ibm/PowerMoE-3b` |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base
- https://huggingface.co/ibm/PowerMoE-3b
