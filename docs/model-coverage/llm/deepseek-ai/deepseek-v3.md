# DeepSeek-V3

[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) is a large-scale Mixture-of-Experts model with 671B total parameters and 37B activated per token. It features Multi-head Latent Attention (MLA), innovative load balancing, and Multi-Token Prediction (MTP). DeepSeek-V3.2 is an updated release with further improvements.

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) by Moonshot AI also uses this architecture with 16B total / 3B activated parameters.

## Available Models

- **DeepSeek-V3**: 671B total, 37B activated
- **DeepSeek-V3.2** (`DeepseekV32ForCausalLM`): updated architecture
- **Moonlight-16B-A3B** (Moonshot AI): 16B total, 3B activated

## Architectures

- `DeepseekV3ForCausalLM`
- `DeepseekV32ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` |
| DeepSeek-V3-Base | `deepseek-ai/DeepSeek-V3-Base` |
| DeepSeek-V3.2 | `deepseek-ai/DeepSeek-V3.2` |
| Moonlight 16B A3B | `moonshotai/Moonlight-16B-A3B` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`deepseek_v32_hellaswag_pp.yaml`](../../../examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml) | SFT — DeepSeek-V3.2 on HellaSwag with pipeline parallelism |
| [`moonlight_16b_te.yaml`](../../../examples/llm_finetune/moonlight/moonlight_16b_te.yaml) | SFT — Moonlight 16B with Transformer Engine |
| [`moonlight_16b_te_packed_sequence.yaml`](../../../examples/llm_finetune/moonlight/moonlight_16b_te_packed_sequence.yaml) | SFT — Moonlight 16B with packed sequences |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/deepseek-ai/DeepSeek-V3
- https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
- https://huggingface.co/moonshotai/Moonlight-16B-A3B
