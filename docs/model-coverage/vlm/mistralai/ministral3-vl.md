# Ministral3 VL

[Ministral3](https://mistral.ai/news/ministraux/) is Mistral AI's efficient small model series. The vision-capable variants support image-text inputs for multimodal tasks.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Mistral3ForConditionalGeneration` |
| **Parameters** | 3B – 14B |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

- **Ministral-3-14B-Instruct-2512**
- **Ministral-3-8B-Instruct-2512**
- **Ministral-3-3B-Instruct-2512**

## Architecture

- `Mistral3ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Ministral-3 3B Instruct | `mistralai/Ministral-3-3B-Instruct-2512` |
| Ministral-3 8B Instruct | `mistralai/Ministral-3-8B-Instruct-2512` |
| Ministral-3 14B Instruct | `mistralai/Ministral-3-14B-Instruct-2512` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`ministral3_3b_medpix.yaml <../../../examples/vlm_finetune/mistral/ministral3_3b_medpix.yaml>` | MedPix-VQA | SFT — Ministral3 3B on MedPix |
| {download}`ministral3_8b_medpix.yaml <../../../examples/vlm_finetune/mistral/ministral3_8b_medpix.yaml>` | MedPix-VQA | SFT — Ministral3 8B on MedPix |
| {download}`ministral3_14b_medpix.yaml <../../../examples/vlm_finetune/mistral/ministral3_14b_medpix.yaml>` | MedPix-VQA | SFT — Ministral3 14B on MedPix |


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
automodel --nproc-per-node=8 examples/vlm_finetune/mistral/ministral3_3b_medpix.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/mistral/ministral3_3b_medpix.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
