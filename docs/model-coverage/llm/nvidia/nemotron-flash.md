# Nemotron-Flash

[NVIDIA Nemotron-Flash](https://huggingface.co/nvidia/Nemotron-Flash-1B) is a compact, fast language model designed for low-latency inference workloads.

:::{note}
This model requires `trust_remote_code: true` in your recipe YAML.
:::

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `NemotronFlashForCausalLM` |
| **Parameters** | 1B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Nemotron-Flash-1B**: 1B parameters

## Architecture

- `NemotronFlashForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Nemotron-Flash 1B | `nvidia/Nemotron-Flash-1B` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`nemotron_flash_1b_squad.yaml`](../../../examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml) | SFT — Nemotron-Flash 1B on SQuAD |
| [`nemotron_flash_1b_squad_peft.yaml`](../../../examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad_peft.yaml) | LoRA — Nemotron-Flash 1B on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [nvidia/Nemotron-Flash-1B](https://huggingface.co/nvidia/Nemotron-Flash-1B)
