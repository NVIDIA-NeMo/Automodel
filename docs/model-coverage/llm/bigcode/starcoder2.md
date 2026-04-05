# StarCoder2

[StarCoder2](https://huggingface.co/blog/starcoder2) is BigCode's second-generation code language model, available in 3B, 7B, and 15B sizes, trained on 600+ programming languages from The Stack v2.

## Available Models

- **starcoder2-3b**: 3B
- **starcoder2-7b**: 7B
- **starcoder2-15b**: 15B

## Architecture

- `Starcoder2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| StarCoder2 3B | `bigcode/starcoder2-3b` |
| StarCoder2 7B | `bigcode/starcoder2-7b` |
| StarCoder2 15B | `bigcode/starcoder2-15b` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`starcoder_2_7b_squad.yaml`](../../../examples/llm_finetune/starcoder/starcoder_2_7b_squad.yaml) | SFT — StarCoder2 7B on SQuAD |
| [`starcoder_2_7b_hellaswag_fp8.yaml`](../../../examples/llm_finetune/starcoder/starcoder_2_7b_hellaswag_fp8.yaml) | SFT — StarCoder2 7B on HellaSwag with FP8 |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/bigcode/starcoder2-7b
