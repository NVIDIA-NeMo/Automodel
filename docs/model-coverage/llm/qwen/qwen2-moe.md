# Qwen2 MoE

[Qwen1.5-MoE](https://qwenlm.github.io/) is a Mixture-of-Experts variant from Alibaba Cloud that activates only a fraction of parameters per token, enabling efficient training and inference at scale.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `Qwen2MoeForCausalLM` |
| **Parameters** | 14.3B total / 2.7B active |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen1.5-MoE-A2.7B**: 14.3B total parameters, 2.7B activated per token

## Architecture

- `Qwen2MoeForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen1.5 MoE A2.7B | `Qwen/Qwen1.5-MoE-A2.7B` |
| Qwen1.5 MoE A2.7B Chat | `Qwen/Qwen1.5-MoE-A2.7B-Chat` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`qwen1_5_moe_a2_7b_qlora.yaml`](../../../examples/llm_finetune/qwen/qwen1_5_moe_a2_7b_qlora.yaml) | QLoRA — Qwen1.5 MoE A2.7B |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)
