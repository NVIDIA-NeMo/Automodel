# Qwen-Image

[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) is Alibaba Cloud's text-to-image diffusion transformer. NeMo AutoModel supports Qwen-Image training via its flow-matching pipeline with a dedicated `qwen_image` adapter, enabling FSDP2 parallelization, multiresolution bucketed dataloading and LoRA-style fine-tuning.

:::{card}
| | |
|---|---|
| **Task** | Text-to-Image |
| **Architecture** | DiT (Flow Matching) |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen-Image**

## Task

- Text-to-Image (T2I)

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen-Image | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`qwen_image_t2i_flow.yaml <../../../../examples/diffusion/finetune/qwen_image_t2i_flow.yaml>` | Fine-tune — Qwen-Image with flow matching |
| {download}`qwen_image_t2i_flow.yaml <../../../../examples/diffusion/pretrain/qwen_image_t2i_flow.yaml>` | Pretrain — Qwen-Image with flow matching |


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
  -c examples/diffusion/finetune/qwen_image_t2i_flow.yaml
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
  -c examples/diffusion/finetune/qwen_image_t2i_flow.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [Diffusion Training and Fine-Tuning Guide](../../../guides/diffusion/finetune.md).

## Fine-Tuning

See the [Diffusion Training and Fine-Tuning Guide](../../../guides/diffusion/finetune.md).

## Hugging Face Model Cards

- [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
