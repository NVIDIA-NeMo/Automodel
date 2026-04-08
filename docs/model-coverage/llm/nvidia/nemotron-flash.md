# Nemotron-Flash

[NVIDIA Nemotron-Flash](https://huggingface.co/nvidia/Nemotron-Flash-1B) is a compact, fast language model designed for low-latency inference workloads.

:::{note}
This model requires `trust_remote_code: true` in your recipe YAML.
:::

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `NemotronFlashForCausalLM` |
| **Parameters** | 1B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Nemotron-Flash-1B**: 1B parameters

## Architecture

- `NemotronFlashForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Nemotron-Flash 1B | [`nvidia/Nemotron-Flash-1B`](https://huggingface.co/nvidia/Nemotron-Flash-1B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`nemotron_flash_1b_squad.yaml <../../../../examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml>` | SFT — Nemotron-Flash 1B on SQuAD |
| {download}`nemotron_flash_1b_squad_peft.yaml <../../../../examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad_peft.yaml>` | LoRA — Nemotron-Flash 1B on SQuAD |


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
automodel --nproc-per-node=8 examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [nvidia/Nemotron-Flash-1B](https://huggingface.co/nvidia/Nemotron-Flash-1B)
