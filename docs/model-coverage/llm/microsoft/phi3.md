# Phi-3 / Phi-4

[Phi-3](https://azure.microsoft.com/en-us/products/phi) and [Phi-4](https://azure.microsoft.com/en-us/products/phi) are Microsoft's high-capability small language models using a shared transformer decoder architecture (`Phi3ForCausalLM`). Phi-4-mini and Phi-4 achieve strong benchmark results at relatively small parameter counts.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Phi3ForCausalLM` |
| **Parameters** | 3.8B – 14B |
| **HF Org** | [microsoft](https://huggingface.co/microsoft) |
:::

## Available Models

- **Phi-4**: 14B
- **Phi-4-mini-instruct**: 3.8B
- **Phi-3.5-mini-instruct**: 3.8B
- **Phi-3-medium-128k-instruct**: 14B
- **Phi-3-mini-128k-instruct**: 3.8B
- **Phi-3-mini-4k-instruct**: 3.8B

## Architecture

- `Phi3ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Phi-4 | `microsoft/Phi-4` |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` |
| Phi-3-mini-128k-instruct | `microsoft/Phi-3-mini-128k-instruct` |
| Phi-3-medium-128k-instruct | `microsoft/Phi-3-medium-128k-instruct` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`phi_4_squad.yaml`](../../../examples/llm_finetune/phi/phi_4_squad.yaml) | SFT — Phi-4 on SQuAD |
| [`phi_4_squad_peft.yaml`](../../../examples/llm_finetune/phi/phi_4_squad_peft.yaml) | LoRA — Phi-4 on SQuAD |
| [`phi_3_mini_it_squad.yaml`](../../../examples/llm_finetune/phi/phi_3_mini_it_squad.yaml) | SFT — Phi-3-mini Instruct on SQuAD |
| [`phi_3_mini_it_squad_peft.yaml`](../../../examples/llm_finetune/phi/phi_3_mini_it_squad_peft.yaml) | LoRA — Phi-3-mini Instruct on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [microsoft/Phi-4](https://huggingface.co/microsoft/Phi-4)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
