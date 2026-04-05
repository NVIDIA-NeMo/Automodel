# Qwen3

[Qwen3](https://qwenlm.github.io/blog/qwen3/) is Alibaba Cloud's third-generation dense language model series, featuring improved reasoning, instruction following, and multilingual capabilities over Qwen2.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Qwen3ForCausalLM` |
| **Parameters** | 0.6B – 32B |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3**: 0.6B, 1.7B, 4B, 8B, 14B, 32B

## Architecture

- `Qwen3ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen3 0.6B | `Qwen/Qwen3-0.6B` |
| Qwen3 8B | `Qwen/Qwen3-8B` |
| Qwen3 32B | `Qwen/Qwen3-32B` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`qwen3_0p6b_hellaswag.yaml <../../../examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml>` | SFT — Qwen3 0.6B on HellaSwag |
| {download}`qwen3_8b_squad_spark.yaml <../../../examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml>` | SFT — Qwen3 8B on SQuAD (Spark) |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/qwen/qwen3_0p6b_hellaswag.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) for full SFT and LoRA instructions.

## Hugging Face Model Cards

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
