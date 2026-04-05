# GritLM

[GritLM](https://github.com/ContextualAI/gritlm) (Generative Representational Instruction Tuning) is a unified model that performs both generative language modeling and text embedding in a single model, from Parasail AI.

:::{card}
| | |
|---|---|
| **Task** | Text Generation + Embedding |
| **Architecture** | `GritLM` |
| **Parameters** | 7B |
| **HF Org** | [parasail-ai](https://huggingface.co/parasail-ai) |
:::

## Available Models

- **GritLM-7B-vllm**

## Architecture

- `GritLM`

## Example HF Models

| Model | HF ID |
|---|---|
| GritLM 7B vllm | `parasail-ai/GritLM-7B-vllm` |


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

- [parasail-ai/GritLM-7B-vllm](https://huggingface.co/parasail-ai/GritLM-7B-vllm)
