# MiniMax-M2

[MiniMax-M2](https://huggingface.co/MiniMaxAI) is MiniMax's large Mixture-of-Experts language model with linear attention for efficient long-context inference.

<Info>

| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `MiniMaxM2ForCausalLM` |
| **Parameters** | varies |
| **HF Org** | [MiniMaxAI](https://huggingface.co/MiniMaxAI) |

</Info>

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
| [minimax_m2.1_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml) | SFT — MiniMax-M2.1 on HellaSwag with pipeline parallelism |
| [minimax_m2.5_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.5_hellaswag_pp.yaml) | SFT — MiniMax-M2.5 on HellaSwag with pipeline parallelism |
| [minimax_m2.7_hellaswag_pp.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_pp.yaml) | SFT — MiniMax-M2.7 on HellaSwag with pipeline parallelism |


## Try with NeMo AutoModel

**1. Clone and install from source** ([full instructions](/get-started/installation)):

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
uv sync --locked --all-groups --all-extras
```

<Note>
This recipe was validated on **8 nodes × 8 GPUs (64 H100s)**. See the [Launcher Guide](../../../launcher/slurm.mdx) for multi-node setup.

</Note>

**2. Run the recipe** from inside the repo:

```bash
uv run automodel --nproc-per-node=8 examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```

<Accordion title="Run with Docker">
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.06.00
```

**2.** Navigate to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/minimax_m2/minimax_m2.1_hellaswag_pp.yaml
```
</Accordion>

See the [Installation Guide](../../../guides/installation.mdx) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.mdx).

## Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.mdx).

## Hugging Face Model Cards

- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7)
