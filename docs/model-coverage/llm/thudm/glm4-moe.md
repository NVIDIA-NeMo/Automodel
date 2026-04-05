# GLM-4 MoE (GLM-4.5 / GLM-4.7)

[GLM-4.5 and GLM-4.7](https://huggingface.co/zai-org) are Mixture-of-Experts variants of the GLM family released under the `zai-org` HuggingFace organization. GLM-4.7-Flash is a lighter variant with fewer active parameters.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `Glm4MoeForCausalLM` / `Glm4MoeLiteForCausalLM` |
| **Parameters** | varies |
| **HF Org** | [zai-org](https://huggingface.co/zai-org) |
:::

## Available Models

- **GLM-4.5-Air** (`Glm4MoeForCausalLM`)
- **GLM-4.7** (`Glm4MoeForCausalLM`)
- **GLM-4.7-Flash** (`Glm4MoeLiteForCausalLM`): lightweight MoE variant

## Architectures

- `Glm4MoeForCausalLM` — GLM-4.5, GLM-4.7
- `Glm4MoeLiteForCausalLM` — GLM-4.7-Flash

## Example HF Models

| Model | HF ID |
|---|---|
| GLM-4.5-Air | `zai-org/GLM-4.5-Air` |
| GLM-4.7 | `zai-org/GLM-4.7` |
| GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`glm_4.5_air_te_deepep.yaml`](../../../examples/llm_finetune/glm/glm_4.5_air_te_deepep.yaml) | SFT — GLM-4.5-Air with TE + DeepEP |
| [`glm_4.7_te_deepep.yaml`](../../../examples/llm_finetune/glm/glm_4.7_te_deepep.yaml) | SFT — GLM-4.7 with TE + DeepEP |
| [`glm_4.7_flash_te_deepep.yaml`](../../../examples/llm_finetune/glm/glm_4.7_flash_te_deepep.yaml) | SFT — GLM-4.7-Flash with TE + DeepEP |
| [`glm_4.7_flash_te_packed_sequence.yaml`](../../../examples/llm_finetune/glm/glm_4.7_flash_te_packed_sequence.yaml) | SFT — GLM-4.7-Flash with packed sequences |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air)
- [zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7)
- [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
