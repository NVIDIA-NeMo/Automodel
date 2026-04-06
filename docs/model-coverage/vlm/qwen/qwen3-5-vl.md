# Qwen3.5-VL

Qwen3.5-VL is Alibaba Cloud's next-generation vision language model series, including dense and MoE variants for image and multimodal understanding tasks.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Qwen3_5VLForConditionalGeneration` |
| **Parameters** | 4B – 35B+ |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3.5-VL-4B**: 4B dense model
- **Qwen3.5-VL-9B**: 9B dense model
- **Qwen3.5-MoE**: large MoE variant (35B+)

## Architectures

- `Qwen3_5VLForConditionalGeneration` — dense models
- `Qwen3_5MoeVLForConditionalGeneration` — MoE variant

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`qwen3_5_4b.yaml <../../../../examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml>` | MedPix-VQA | SFT — Qwen3.5-VL 4B on MedPix |
| {download}`qwen3_5_9b.yaml <../../../../examples/vlm_finetune/qwen3_5/qwen3_5_9b.yaml>` | MedPix-VQA | SFT — Qwen3.5-VL 9B on MedPix |
| {download}`qwen3_5_moe_medpix.yaml <../../../../examples/vlm_finetune/qwen3_5_moe/qwen3_5_moe_medpix.yaml>` | MedPix-VQA | SFT — Qwen3.5-MoE on MedPix |
| {download}`qwen3_5_35b.yaml <../../../../examples/vlm_finetune/qwen3_5_moe/qwen3_5_35b.yaml>` | MedPix-VQA | SFT — Qwen3.5 35B on MedPix |


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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen](https://huggingface.co/Qwen)
