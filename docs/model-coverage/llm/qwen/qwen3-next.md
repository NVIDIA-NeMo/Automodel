# Qwen3-Next

Qwen3-Next is an advanced MoE language model from Alibaba Cloud's Qwen team designed for high-throughput inference with large total parameter counts and efficient per-token activation.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `Qwen3NextForCausalLM` |
| **Parameters** | 80B total / 3B active |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3-Next-80B-A3B**: 80B total parameters, 3B activated per token

## Architecture

- `Qwen3NextForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen3-Next 80B A3B Instruct | `Qwen/Qwen3-Next-80B-A3B-Instruct` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`qwen3_next_te_deepep.yaml <../../../examples/llm_finetune/qwen/qwen3_next_te_deepep.yaml>` | SFT — Qwen3-Next with TE + DeepEP |


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

:::{note}
This recipe was validated on **4 nodes × 8 GPUs (32 H100s)**. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/qwen/qwen3_next_te_deepep.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/qwen/qwen3_next_te_deepep.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
