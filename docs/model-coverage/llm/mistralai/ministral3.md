# Ministral3 / Devstral

[Ministral](https://mistral.ai/news/ministraux/) is Mistral AI's efficient small model series optimized for on-device and edge use cases. [Devstral](https://mistral.ai/news/devstral/) is a code-focused model built on the same architecture, designed for software engineering agents.

Both use the `Mistral3ForConditionalGeneration` architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Mistral3ForConditionalGeneration` |
| **Parameters** | 3B – 24B |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

**Ministral3:**
- **Ministral-3-3B-Instruct-2512**
- **Ministral-3-8B-Instruct-2512**
- **Ministral-3-14B-Instruct-2512**

**Devstral:**
- **Devstral-Small-2-24B-Instruct-2512**

## Architecture

- `Mistral3ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Ministral-3 3B Instruct | `mistralai/Ministral-3-3B-Instruct-2512` |
| Ministral-3 8B Instruct | `mistralai/Ministral-3-8B-Instruct-2512` |
| Ministral-3 14B Instruct | `mistralai/Ministral-3-14B-Instruct-2512` |
| Devstral Small 2 24B | `mistralai/Devstral-Small-2-24B-Instruct-2512` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`devstral2_small_2512_squad.yaml`](../../../examples/llm_finetune/devstral/devstral2_small_2512_squad.yaml) | SFT — Devstral Small 2 24B on SQuAD |
| [`devstral2_small_2512_squad_peft.yaml`](../../../examples/llm_finetune/devstral/devstral2_small_2512_squad_peft.yaml) | LoRA — Devstral Small 2 24B on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)
