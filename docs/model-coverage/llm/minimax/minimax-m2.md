# MiniMax-M2

[MiniMax-M2](https://huggingface.co/MiniMaxAI) is MiniMax's large Mixture-of-Experts language model with linear attention for efficient long-context inference.

## Available Models

- **MiniMax-M2.1**
- **MiniMax-M2.5**

## Architecture

- `MiniMaxM2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| MiniMax M2.1 | `MiniMaxAI/MiniMax-M2.1` |
| MiniMax M2.5 | `MiniMaxAI/MiniMax-M2.5` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`minimax_m2.1_hellaswag_pp.yaml`](../../../examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml) | SFT — MiniMax-M2.1 on HellaSwag with pipeline parallelism |
| [`minimax_m2.5_hellaswag_pp.yaml`](../../../examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml) | SFT — MiniMax-M2.5 on HellaSwag with pipeline parallelism |

## Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/MiniMaxAI/MiniMax-M2.1
