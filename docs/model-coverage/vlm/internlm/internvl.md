# InternVL

[InternVL](https://github.com/OpenGVLab/InternVL) is a vision language model from Shanghai AI Laboratory (OpenGVLab), combining a large vision encoder with an InternLM language backbone for strong multimodal performance.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `InternVLForConditionalGeneration` |
| **Parameters** | 4B – 8B |
| **HF Org** | [OpenGVLab](https://huggingface.co/OpenGVLab) |
:::

## Available Models

- **InternVL3.5-4B**
- **InternVL3.5-8B**

## Architecture

- `InternVLForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| InternVL3.5 4B | [`OpenGVLab/InternVL3_5-4B`](https://huggingface.co/OpenGVLab/InternVL3_5-4B) |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`internvl_3_5_4b.yaml <../../../../examples/vlm_finetune/internvl/internvl_3_5_4b.yaml>` | MedPix-VQA | SFT — InternVL3.5 4B on MedPix |


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
automodel --nproc-per-node=8 examples/vlm_finetune/internvl/internvl_3_5_4b.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/internvl/internvl_3_5_4b.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [OpenGVLab/InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B)
