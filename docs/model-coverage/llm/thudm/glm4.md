# GLM-4

[GLM-4](https://github.com/THUDM/GLM-4) is Tsinghua University (THUDM)'s fourth-generation General Language Model, featuring strong multilingual capabilities and tool-use support.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GlmForCausalLM` / `Glm4ForCausalLM` |
| **Parameters** | 9B – 32B |
| **HF Org** | [THUDM](https://huggingface.co/THUDM) |
:::

## Available Models

- **GLM-4-9B-Chat-HF** (`GlmForCausalLM`): 9B
- **GLM-4-32B-0414** (`Glm4ForCausalLM`): 32B

## Architectures

- `GlmForCausalLM` — GLM-4 series
- `Glm4ForCausalLM` — GLM-4-0414 series

## Example HF Models

| Model | HF ID |
|---|---|
| GLM-4-9B-Chat-HF | `THUDM/glm-4-9b-chat-hf` |
| GLM-4-32B-0414 | `THUDM/GLM-4-32B-0414` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`glm_4_9b_chat_hf_squad.yaml`](../../../examples/llm_finetune/glm/glm_4_9b_chat_hf_squad.yaml) | SFT — GLM-4 9B on SQuAD |
| [`glm_4_9b_chat_hf_hellaswag_fp8.yaml`](../../../examples/llm_finetune/glm/glm_4_9b_chat_hf_hellaswag_fp8.yaml) | SFT — GLM-4 9B on HellaSwag with FP8 |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [THUDM/glm-4-9b-chat-hf](https://huggingface.co/THUDM/glm-4-9b-chat-hf)
- [THUDM/GLM-4-32B-0414](https://huggingface.co/THUDM/GLM-4-32B-0414)
