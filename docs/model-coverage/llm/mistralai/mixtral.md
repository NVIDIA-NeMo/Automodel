# Mixtral

[Mixtral](https://mistral.ai/news/mixtral-of-experts/) is Mistral AI's Mixture-of-Experts model series. Each token is processed by a subset of experts, enabling a large total parameter count with efficient per-token compute.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `MixtralForCausalLM` |
| **Parameters** | 47B total / 13B active |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

- **Mixtral-8x7B**: 8 experts, 2 active per token (~13B active)
- **Mixtral-8x7B-Instruct**: instruction-tuned variant
- **Mixtral-8x22B**: 8 experts, 2 active per token (~39B active)

## Architecture

- `MixtralForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Mixtral 8x7B v0.1 | `mistralai/Mixtral-8x7B-v0.1` |
| Mixtral 8x7B Instruct v0.1 | `mistralai/Mixtral-8x7B-Instruct-v0.1` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`mixtral-8x7b-v0-1_squad.yaml <../../../examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad.yaml>` | SFT — Mixtral 8x7B on SQuAD |
| {download}`mixtral-8x7b-v0-1_squad_peft.yaml <../../../examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad_peft.yaml>` | LoRA — Mixtral 8x7B on SQuAD |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad.yaml
```

:::{dropdown} Run with Docker
Pull the NeMo AutoModel container and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

Then inside the container:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/mistral/mixtral-8x7b-v0-1_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
