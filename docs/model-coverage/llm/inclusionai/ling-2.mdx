# Ling 2.0

[Ling 2.0](https://huggingface.co/collections/inclusionAI/ling-20) is the Mixture-of-Experts
LLM family from inclusionAI (Ant Group), released under the `bailing_moe` HF
architecture (`BailingMoeV2ForCausalLM`).  The line spans a 16 B mini through
a 1 T flagship while sharing the same architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `BailingMoeV2ForCausalLM` |
| **Parameters** | 16 B – 1 T total |
| **HF Org** | [inclusionAI](https://huggingface.co/inclusionAI) |
:::

## Available Models

- **Ling-mini-2.0**: 16 B total / ~1.4 B activated per token (20 layers, 256 experts, 8 activated).
- **Ling-flash-2.0**: 100 B total / ~6 B activated per token (32 layers, 256 experts, 8 activated).
- **Ling-1T**: 1 T total / ~50 B activated per token (80 layers, `first_k_dense_replace=4`).
- **Ling-mini-base-2.0** / **Ling-flash-base-2.0**: base (pre-instruct) variants.

All variants share the same architecture: GQA + per-head QK-RMSNorm + half RoPE
(`partial_rotary_factor=0.5`) + sigmoid-routed grouped MoE with one shared
expert and a per-expert correction bias (aux-loss-free routing).

## Architecture

- `BailingMoeV2ForCausalLM` (HF `model_type: "bailing_moe"`)
- GQA attention; `use_qk_norm: true`
- Half RoPE (`partial_rotary_factor=0.5`)
- DeepSeek-V3-style routing: sigmoid scoring, per-expert bias, grouped top-k
  (`n_group=8`, `topk_group=4`)
- 1 shared expert at `moe_intermediate_size`
- `first_k_dense_replace` dense MLP layer(s) at the start of the stack

## Example HF Models

| Model | HF ID |
|---|---|
| Ling-mini-2.0 | [`inclusionAI/Ling-mini-2.0`](https://huggingface.co/inclusionAI/Ling-mini-2.0) |
| Ling-flash-2.0 | [`inclusionAI/Ling-flash-2.0`](https://huggingface.co/inclusionAI/Ling-flash-2.0) |
| Ling-1T | [`inclusionAI/Ling-1T`](https://huggingface.co/inclusionAI/Ling-1T) |

## Example Recipes

| Recipe | Description | Min HW |
|---|---|---|
| {download}`ling_mini_2_0_squad.yaml <../../../../examples/llm_finetune/ling/ling_mini_2_0_squad.yaml>` | LoRA SFT — Ling-mini-2.0 on SQuAD | 2× H100 80GB |
| {download}`ling_mini_2_0_hellaswag.yaml <../../../../examples/llm_finetune/ling/ling_mini_2_0_hellaswag.yaml>` | LoRA SFT — Ling-mini-2.0 on HellaSwag | 2× H100 80GB |
| {download}`ling_mini_2_0_sft.yaml <../../../../examples/llm_finetune/ling/ling_mini_2_0_sft.yaml>` | Full SFT — Ling-mini-2.0 on HellaSwag, FSDP2 + EP=8 | 8× H100 80GB |
| {download}`ling_flash_2_0_lora.yaml <../../../../examples/llm_finetune/ling/ling_flash_2_0_lora.yaml>` | LoRA SFT — Ling-flash-2.0 on HellaSwag | 8× H100 80GB |
| {download}`ling_flash_2_0_sft.yaml <../../../../examples/llm_finetune/ling/ling_flash_2_0_sft.yaml>` | Full SFT — Ling-flash-2.0 on HellaSwag, FSDP2 + EP=32 | 32× H100 80GB (4 nodes) |
| {download}`ling_1t_lora_pp.yaml <../../../../examples/llm_finetune/ling/ling_1t_lora_pp.yaml>` | LoRA SFT — Ling-1T on HellaSwag, FSDP2 + PP=8 + EP=8 | 64× H100 80GB (8 nodes) |
| {download}`ling_1t_sft.yaml <../../../../examples/llm_finetune/ling/ling_1t_sft.yaml>` | Full SFT — Ling-1T on HellaSwag, FSDP2 + PP=4 + EP=64 | 256× H100 80GB (32 nodes) |

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)).

**2. Run LoRA fine-tuning:**

```bash
automodel examples/llm_finetune/ling/ling_mini_2_0_squad.yaml --nproc-per-node 1
```

A single 80 GB H100 / A100 fits Ling-mini-2.0 in bf16 with the LoRA defaults
in the example.  Set `distributed.ep_size > 1` for multi-GPU expert
parallelism on the larger variants.
