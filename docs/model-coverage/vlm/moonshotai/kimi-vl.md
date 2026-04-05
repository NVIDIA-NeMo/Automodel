# Kimi-VL

[Kimi-VL](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) and Kimi-K25-VL are vision language models from Moonshot AI. Kimi-VL-A3B uses a MoE language backbone (3B active parameters) with a vision encoder, supporting image understanding and multimodal reasoning.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `KimiVLForConditionalGeneration` |
| **Parameters** | ~3B active (MoE) |
| **HF Org** | [moonshotai](https://huggingface.co/moonshotai) |
:::

## Available Models

- **Kimi-VL-A3B-Instruct**
- **Kimi-K25-VL**

## Architecture

- `KimiVLForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Kimi-VL-A3B-Instruct | [`moonshotai/Kimi-VL-A3B-Instruct`](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`kimi2vl_cordv2.yaml <../../../examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml>` | cord-v2 | SFT — Kimi-VL on CORD-v2 |
| {download}`kimi25vl_medpix.yaml <../../../examples/vlm_finetune/kimi/kimi25vl_medpix.yaml>` | MedPix-VQA | SFT — Kimi-K25-VL on MedPix |


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
automodel --nproc-per-node=8 examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [moonshotai/Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)
