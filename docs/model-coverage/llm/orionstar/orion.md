# Orion

[Orion](https://github.com/OrionStarAI/Orion) is a bilingual (Chinese-English) language model from OrionStar AI, with 14B parameters and strong performance on Chinese benchmarks.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `OrionForCausalLM` |
| **Parameters** | 14B |
| **HF Org** | [OrionStarAI](https://huggingface.co/OrionStarAI) |
:::

## Available Models

- **Orion-14B-Base**: 14B
- **Orion-14B-Chat**: 14B instruction-tuned

## Architecture

- `OrionForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Orion 14B Base | `OrionStarAI/Orion-14B-Base` |
| Orion 14B Chat | `OrionStarAI/Orion-14B-Chat` |


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

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [OrionStarAI/Orion-14B-Base](https://huggingface.co/OrionStarAI/Orion-14B-Base)
