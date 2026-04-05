# Seed (ByteDance)

[Seed-Coder](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct) and [Seed-OSS](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) are open-weight models from ByteDance. Both use the `Qwen2ForCausalLM` architecture under the hood.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Qwen2ForCausalLM` |
| **Parameters** | 8B – 36B |
| **HF Org** | [ByteDance-Seed](https://huggingface.co/ByteDance-Seed) |
:::

## Available Models

- **Seed-Coder-8B-Instruct**: 8B code model
- **Seed-OSS-36B-Instruct**: 36B general model

## Architecture

- `Qwen2ForCausalLM` (reuses Qwen2 architecture)

## Example HF Models

| Model | HF ID |
|---|---|
| Seed-Coder 8B Instruct | [`ByteDance-Seed/Seed-Coder-8B-Instruct`](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct) |
| Seed-OSS 36B Instruct | [`ByteDance-Seed/Seed-OSS-36B-Instruct`](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`seed_coder_8b_instruct_squad.yaml <../../../examples/llm_finetune/seed/seed_coder_8b_instruct_squad.yaml>` | SFT — Seed-Coder 8B on SQuAD |
| {download}`seed_coder_8b_instruct_squad_peft.yaml <../../../examples/llm_finetune/seed/seed_coder_8b_instruct_squad_peft.yaml>` | LoRA — Seed-Coder 8B on SQuAD |
| {download}`seed_oss_36B_hellaswag.yaml <../../../examples/llm_finetune/seed/seed_oss_36B_hellaswag.yaml>` | SFT — Seed-OSS 36B on HellaSwag |
| {download}`seed_oss_36B_hellaswag_peft.yaml <../../../examples/llm_finetune/seed/seed_oss_36B_hellaswag_peft.yaml>` | LoRA — Seed-OSS 36B on HellaSwag |


## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/seed/seed_coder_8b_instruct_squad.yaml
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** Navigate to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/seed/seed_coder_8b_instruct_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [ByteDance-Seed/Seed-Coder-8B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Instruct)
- [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct)
