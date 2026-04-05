# Granite

[IBM Granite](https://www.ibm.com/granite) is IBM's family of enterprise-focused language models. Granite 3.x models are trained on a mix of code and language data and are optimized for enterprise tasks including summarization, classification, and RAG. PowerLM (IBM Research) also uses this architecture.

## Available Models

- **Granite 3.3 2B Instruct**
- **Granite 3.1 8B Instruct**
- **Granite 3.0 2B Base**
- **PowerLM-3B** (IBM Research)

## Architecture

- `GraniteForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Granite 3.0 2B Base | `ibm-granite/granite-3.0-2b-base` |
| Granite 3.1 8B Instruct | `ibm-granite/granite-3.1-8b-instruct` |
| PowerLM 3B | `ibm/PowerLM-3b` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`granite_3_3_2b_instruct_squad.yaml`](../../../examples/llm_finetune/granite/granite_3_3_2b_instruct_squad.yaml) | SFT — Granite 3.3 2B Instruct on SQuAD |
| [`granite_3_3_2b_instruct_squad_peft.yaml`](../../../examples/llm_finetune/granite/granite_3_3_2b_instruct_squad_peft.yaml) | LoRA — Granite 3.3 2B Instruct on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/ibm-granite/granite-3.0-2b-base
- https://huggingface.co/ibm-granite/granite-3.1-8b-instruct
