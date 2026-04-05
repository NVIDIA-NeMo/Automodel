# Nemotron-Super (Llama-3.3-Nemotron-Super-49B)

[Llama-3.3-Nemotron-Super-49B-v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1) is a NVIDIA model derived from Llama-3.1-70B through Neural Architecture Search (NAS)-based pruning and knowledge distillation, resulting in a 49B model with strong reasoning capabilities. It uses the `DeciLMForCausalLM` architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `DeciLMForCausalLM` |
| **Parameters** | 49B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Llama-3.3-Nemotron-Super-49B-v1**: 49B parameters

## Architecture

- `DeciLMForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Llama-3.3-Nemotron-Super-49B-v1 | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` |


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

- [nvidia/Llama-3_3-Nemotron-Super-49B-v1](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1)
