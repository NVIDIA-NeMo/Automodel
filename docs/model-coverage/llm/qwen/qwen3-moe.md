# Qwen3 MoE

[Qwen3 MoE](https://qwenlm.github.io/blog/qwen3/) is the Mixture-of-Experts variant of the Qwen3 series from Alibaba Cloud, activating a small fraction of parameters per token for efficient large-scale training.

## Available Models

- **Qwen3-30B-A3B**: 30B total parameters, 3B activated per token
- **Qwen3-235B-A22B**: 235B total parameters, 22B activated per token

## Architecture

- `Qwen3MoeForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen3 30B A3B | `Qwen/Qwen3-30B-A3B` |
| Qwen3 235B A22B | `Qwen/Qwen3-235B-A22B` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`qwen3_moe_30b_te_deepep.yaml`](../../../examples/llm_finetune/qwen/qwen3_moe_30b_te_deepep.yaml) | SFT — Qwen3 MoE 30B with TE + DeepEP |
| [`qwen3_moe_30b_lora.yaml`](../../../examples/llm_finetune/qwen/qwen3_moe_30b_lora.yaml) | LoRA — Qwen3 MoE 30B |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/Qwen/Qwen3-30B-A3B
- https://huggingface.co/Qwen/Qwen3-235B-A22B
