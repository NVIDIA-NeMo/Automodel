# DeepSeek V4 Flash

[DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is DeepSeek's latest fine-grained Mixture-of-Experts language model. It uses a 43-layer all-MoE backbone with 256 routed experts plus one shared expert per block, top-6 routing, and a hybrid per-layer attention zoo (SWA / CSA / HCA) selectable through `compress_ratios`. The first `num_hash_layers` blocks use a hash-clustering gate, and every block maintains `hc_mult=4` Hyper-Connection streams mixed via a learned col-norm-first Sinkhorn router.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `DeepseekV4ForCausalLM` |
| **Parameters** | fine-grained MoE, 256 routed + 1 shared expert |
| **HF Org** | [deepseek-ai](https://huggingface.co/deepseek-ai) |
:::

## Available Models

- **DeepSeek-V4-Flash**

## Architecture

- `DeepseekV4ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| DeepSeek V4 Flash | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |

## Example Recipes

| Recipe | Description |
|---|---|
| [`deepseek_v4_flash_hellaswag.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml) | SFT — DeepSeek V4 Flash on HellaSwag with pipeline parallelism |


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
The full 43-layer schedule requires a multi-node run; see the recipe yaml header for `ep_size` / `pp_size` guidance. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [Fine-Tune DeepSeek V4 Flash](../../../guides/llm/dsv4-flash.md) guide and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.md).

## Hugging Face Model Cards

- [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
