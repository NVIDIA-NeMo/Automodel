# Nemotron-H

[NVIDIA Nemotron-H](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/) is a hybrid Mamba-2 / transformer architecture that interleaves selective state space layers with standard attention layers for improved efficiency on long sequences.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `NemotronHForCausalLM` |
| **Parameters** | 9B – 30B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **NVIDIA-Nemotron-Nano-9B-v2**: 9B hybrid model
- **NVIDIA-Nemotron-Nano-12B-v2**: 12B hybrid model
- **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16**: 30B total, 3B activated (sparse MoE + Mamba-2)

## Architecture

- `NemotronHForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Nemotron-Nano 9B v2 | [`nvidia/NVIDIA-Nemotron-Nano-9B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |
| Nemotron-Nano 12B v2 | [`nvidia/NVIDIA-Nemotron-Nano-12B-v2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2) |
| Nemotron-3-Nano 30B A3B | [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`nemotron_nano_9b_squad.yaml <../../../../examples/llm_finetune/nemotron/nemotron_nano_9b_squad.yaml>` | SFT — Nemotron-Nano 9B on SQuAD |
| {download}`nemotron_nano_9b_squad_peft.yaml <../../../../examples/llm_finetune/nemotron/nemotron_nano_9b_squad_peft.yaml>` | LoRA — Nemotron-Nano 9B on SQuAD |
| {download}`nemotron_nano_v3_hellaswag.yaml <../../../../examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag.yaml>` | SFT — Nemotron-3-Nano 30B on HellaSwag |
| {download}`nemotron_nano_v3_hellaswag_peft.yaml <../../../../examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml>` | LoRA — Nemotron-3-Nano 30B on HellaSwag |


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
automodel --nproc-per-node=8 examples/llm_finetune/nemotron/nemotron_nano_9b_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/nemotron/nemotron_nano_9b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [nvidia/NVIDIA-Nemotron-Nano-12B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2)
- [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
