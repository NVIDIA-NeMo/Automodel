# Step-3.5

[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is a Mixture-of-Experts language model from Stepfun AI, designed for efficient inference.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `Step3p5ForCausalLM` |
| **Parameters** | varies |
| **HF Org** | [stepfun-ai](https://huggingface.co/stepfun-ai) |
:::

## Available Models

- **Step-3.5-Flash**

## Architecture

- `Step3p5ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Step-3.5-Flash | `stepfun-ai/Step-3.5-Flash` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`step_3.5_flash_hellaswag_pp.yaml <../../../examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml>` | SFT — Step-3.5-Flash on HellaSwag with pipeline parallelism |


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
This recipe was validated on **16 nodes × 8 GPUs (128 H100s)**. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [stepfun-ai/Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash)
