# Qwen2

[Qwen2](https://qwenlm.github.io/) is Alibaba Cloud's second-generation large language model series. It features grouped query attention, YARN-based long-context extension, and dual chunk attention for long sequences. QwQ-32B-Preview, a reasoning-focused model, also uses this architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Qwen2ForCausalLM` |
| **Parameters** | 0.5B – 72B |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen2.5**: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- **Qwen2**: 0.5B, 1.5B, 7B, 57B-A14B (MoE), 72B
- **QwQ-32B-Preview** — reasoning model

## Architecture

- `Qwen2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen2.5 72B Instruct | `Qwen/Qwen2.5-72B-Instruct` |
| Qwen2 7B Instruct | `Qwen/Qwen2-7B-Instruct` |
| QwQ 32B Preview | `Qwen/QwQ-32B-Preview` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`qwen2_5_7b_squad.yaml <../../../examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml>` | SFT — Qwen2.5 7B on SQuAD |
| {download}`qwq_32b_squad_peft.yaml <../../../examples/llm_finetune/qwen/qwq_32b_squad_peft.yaml>` | LoRA — QwQ 32B on SQuAD |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/qwen/qwen2_5_7b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) for full SFT and LoRA instructions.

## Hugging Face Model Cards

- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview)
