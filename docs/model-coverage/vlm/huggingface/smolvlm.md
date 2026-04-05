# SmolVLM

[SmolVLM](https://huggingface.co/blog/smolvlm) is HuggingFace's compact vision language model designed for on-device and memory-constrained deployment, featuring an efficient image token compression strategy.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `SmolVLMForConditionalGeneration` |
| **Parameters** | 256M – 2B |
| **HF Org** | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) |
:::

## Available Models

- **SmolVLM-Instruct**: 2B
- **SmolVLM-256M-Instruct**: 256M

## Architecture

- `SmolVLMForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| SmolVLM Instruct | `HuggingFaceTB/SmolVLM-Instruct` |
| SmolVLM 256M Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct` |


## Try with NeMo AutoModel

Install NeMo AutoModel and follow the fine-tuning guide to configure a recipe for this model.

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get example recipes you can adapt:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** The recipes are at `/opt/Automodel/examples/` — navigate there:

```bash
cd /opt/Automodel
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
