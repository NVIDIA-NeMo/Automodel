# MiniMax-M2

[MiniMax-M2](https://huggingface.co/MiniMaxAI) is MiniMax's large Mixture-of-Experts language model with linear attention for efficient long-context inference.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `MiniMaxM2ForCausalLM` |
| **Parameters** | varies |
| **HF Org** | [MiniMaxAI](https://huggingface.co/MiniMaxAI) |
:::

## Available Models

- **MiniMax-M2.1**
- **MiniMax-M2.5**
- **MiniMax-M2.7**
## Architecture

- `MiniMaxM2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| MiniMax M2.1 | [`MiniMaxAI/MiniMax-M2.1`](https://huggingface.co/MiniMaxAI/MiniMax-M2.1) |
| MiniMax M2.5 | [`MiniMaxAI/MiniMax-M2.5`](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) |
| MiniMax M2.7 | [`MiniMaxAI/MiniMax-M2.7`](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`minimax_m2.1_hellaswag_pp.yaml <../../../../examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml>` | SFT — MiniMax-M2.1 on HellaSwag with pipeline parallelism |
| {download}`minimax_m2.5_hellaswag_pp.yaml <../../../../examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml>` | SFT — MiniMax-M2.5 on HellaSwag with pipeline parallelism |
| {download}`minimax_m2.7_hellaswag_pp.yaml <../../../../examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_pp.yaml>` | SFT — MiniMax-M2.7 on HellaSwag with pipeline parallelism |


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
This recipe was validated on **8 nodes × 8 GPUs (64 H100s)**. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
