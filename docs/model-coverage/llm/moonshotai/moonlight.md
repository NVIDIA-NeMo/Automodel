# Moonlight

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) is a Mixture-of-Experts language model from Moonshot AI trained using Muon optimizer. It uses the `DeepseekV3ForCausalLM` architecture with 16B total parameters and 3B activated per token.

## Available Models

- **Moonlight-16B-A3B**: 16B total, 3B activated

## Architecture

- `DeepseekV3ForCausalLM` (same architecture as DeepSeek-V3)

## Example HF Models

| Model | HF ID |
|---|---|
| Moonlight 16B A3B | `moonshotai/Moonlight-16B-A3B` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`moonlight_16b_te.yaml`](../../../examples/llm_finetune/moonlight/moonlight_16b_te.yaml) | SFT — Moonlight 16B with Transformer Engine |
| [`moonlight_16b_te_packed_sequence.yaml`](../../../examples/llm_finetune/moonlight/moonlight_16b_te_packed_sequence.yaml) | SFT — Moonlight 16B with packed sequences |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/moonshotai/Moonlight-16B-A3B
