# Mistral-Small-4

[Mistral-Small-4-119B](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) is Mistral AI's multimodal MoE model supporting both text and image inputs at scale.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `MistralForConditionalGeneration` |
| **Parameters** | 119B (MoE) |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

- **Mistral-Small-4-119B-2603**

## Architecture

- `MistralForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Mistral-Small-4 119B | [`mistralai/Mistral-Small-4-119B-2603`](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`mistral4_medpix.yaml <../../../../examples/vlm_finetune/mistral4/mistral4_medpix.yaml>` | MedPix-VQA | SFT — Mistral-Small-4 on MedPix |


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
automodel --nproc-per-node=8 examples/vlm_finetune/mistral4/mistral4_medpix.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/mistral4/mistral4_medpix.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [mistralai/Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
