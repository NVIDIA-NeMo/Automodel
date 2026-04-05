# FLUX.1-dev

[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is a 12B parameter text-to-image diffusion transformer from Black Forest Labs, trained with flow matching. It produces high-fidelity images and is designed for non-commercial research and development use.

:::{card}
| | |
|---|---|
| **Task** | Text-to-Image |
| **Architecture** | DiT (Flow Matching) |
| **Parameters** | 12B |
| **HF Org** | [black-forest-labs](https://huggingface.co/black-forest-labs) |
:::

## Available Models

- **FLUX.1-dev**: 12B parameters

## Task

- Text-to-Image (T2I)

## Example HF Models

| Model | HF ID |
|---|---|
| FLUX.1-dev | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`flux_t2i_flow.yaml <../../../examples/diffusion/finetune/flux_t2i_flow.yaml>` | Fine-tune — FLUX.1-dev with flow matching |
| {download}`flux_t2i_flow.yaml <../../../examples/diffusion/pretrain/flux_t2i_flow.yaml>` | Pretrain — FLUX.1-dev with flow matching |


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
torchrun --nproc-per-node=8 \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/flux_t2i_flow.yaml
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
torchrun --nproc-per-node=8 \
  examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/flux_t2i_flow.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [Diffusion Fine-Tuning Guide](../../../guides/diffusion/finetune.md).

## Training

See the [Diffusion Training and Fine-Tuning Guide](../../../guides/diffusion/finetune.md) and [Dataset Preparation](../../../guides/diffusion/dataset.md).

## Hugging Face Model Cards

- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
