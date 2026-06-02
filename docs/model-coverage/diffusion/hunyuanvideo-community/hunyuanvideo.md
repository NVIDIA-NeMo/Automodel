# HunyuanVideo 1.5

[HunyuanVideo 1.5](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v) is a 13B parameter text-to-video diffusion model from the Hunyuan community, supporting 720p resolution video generation with flow matching training.

:::{card}
| | |
|---|---|
| **Task** | Text-to-Video |
| **Architecture** | DiT (Flow Matching) |
| **Parameters** | 13B |
| **HF Org** | [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) |
:::

## Available Models

- **HunyuanVideo-1.5-Diffusers-720p_t2v**: 13B parameters

## Task

- Text-to-Video (T2V)

## Example HF Models

| Model | HF ID |
|---|---|
| HunyuanVideo 1.5 720p T2V | [`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`hunyuan_t2v_flow.yaml <../../../../examples/diffusion/finetune/hunyuan_t2v_flow.yaml>` | Fine-tune — HunyuanVideo 1.5 with flow matching |


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
  -c examples/diffusion/finetune/hunyuan_t2v_flow.yaml
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
  -c examples/diffusion/finetune/hunyuan_t2v_flow.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [Diffusion Fine-Tuning Guide](../../../guides/diffusion/finetune.md).

## Training

See the [Diffusion Training and Fine-Tuning Guide](../../../guides/diffusion/finetune.md) and [Dataset Preparation](../../../guides/diffusion/dataset.md).

## Hugging Face Model Cards

- [hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v)
