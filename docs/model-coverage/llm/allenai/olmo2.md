# OLMo2

[OLMo2](https://allenai.org/olmo) is Allen AI's second-generation open language model with improved architecture and training, including RMSNorm and rotary position embeddings.

## Available Models

- **OLMo2-0425-1B-Instruct**
- **OLMo2-7B-1124**
- **OLMo2-13B-1124**

## Architecture

- `OLMo2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| OLMo2 7B | `allenai/OLMo2-7B-1124` |
| OLMo2 0425 1B Instruct | `allenai/OLMo2-0425-1B-Instruct` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`olmo_2_0425_1b_instruct_squad.yaml`](../../../examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad.yaml) | SFT — OLMo2 0425 1B Instruct on SQuAD |
| [`olmo_2_0425_1b_instruct_squad_peft.yaml`](../../../examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad_peft.yaml) | LoRA — OLMo2 0425 1B Instruct on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/allenai/OLMo2-7B-1124
