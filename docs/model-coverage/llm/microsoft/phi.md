# Phi

[Microsoft's Phi](https://azure.microsoft.com/en-us/products/phi) are compact, high-capability language models designed to punch above their weight class. Phi-1.5 and Phi-2 use a standard transformer decoder architecture (`PhiForCausalLM`). For Phi-3 and Phi-4 see [Phi-3 / Phi-4](phi3.md).

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `PhiForCausalLM` |
| **Parameters** | 1.3B – 2.7B |
| **HF Org** | [microsoft](https://huggingface.co/microsoft) |
:::

## Available Models

- **Phi-2**: 2.7B
- **Phi-1.5**: 1.3B

## Architecture

- `PhiForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Phi-2 | `microsoft/phi-2` |
| Phi-1.5 | `microsoft/phi-1_5` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`phi_2_squad.yaml`](../../../examples/llm_finetune/phi/phi_2_squad.yaml) | SFT — Phi-2 on SQuAD |
| [`phi_2_squad_peft.yaml`](../../../examples/llm_finetune/phi/phi_2_squad_peft.yaml) | LoRA — Phi-2 on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5)
