# Mistral

[Mistral AI](https://mistral.ai/) models are efficient transformer decoder models featuring sliding window attention for long context support. Mistral-Nemo is a 12B model developed jointly with NVIDIA.

## Available Models

- **Mistral-7B**: v0.1, v0.2, v0.3
- **Mistral-7B-Instruct**: v0.1, v0.2, v0.3
- **Mistral-Nemo-Instruct-2407**: 12B

## Architecture

- `MistralForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Mistral 7B v0.1 | `mistralai/Mistral-7B-v0.1` |
| Mistral 7B Instruct v0.1 | `mistralai/Mistral-7B-Instruct-v0.1` |
| Mistral Nemo Instruct 2407 | `mistralai/Mistral-Nemo-Instruct-2407` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`mistral_7b_squad.yaml`](../../../examples/llm_finetune/mistral/mistral_7b_squad.yaml) | SFT — Mistral 7B on SQuAD |
| [`mistral_7b_squad_peft.yaml`](../../../examples/llm_finetune/mistral/mistral_7b_squad_peft.yaml) | LoRA — Mistral 7B on SQuAD |
| [`mistral_nemo_2407_squad.yaml`](../../../examples/llm_finetune/mistral/mistral_nemo_2407_squad.yaml) | SFT — Mistral Nemo 2407 on SQuAD |
| [`mistral_nemo_2407_squad_peft.yaml`](../../../examples/llm_finetune/mistral/mistral_nemo_2407_squad_peft.yaml) | LoRA — Mistral Nemo 2407 on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/mistralai/Mistral-7B-v0.1
- https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
