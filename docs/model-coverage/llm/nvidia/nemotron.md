# Nemotron / Minitron

[NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) and [Minitron](https://developer.nvidia.com/blog/minitron-approach-for-model-compression/) are NVIDIA's family of language models. Minitron models are produced by pruning and distilling larger Llama/Nemotron models into compact, high-performance checkpoints.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `NemotronForCausalLM` |
| **Parameters** | 8B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Minitron-8B-Base**: pruned and distilled from Llama-3.1-8B

## Architecture

- `NemotronForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Minitron 8B Base | `nvidia/Minitron-8B-Base` |


## Try with NeMo AutoModel

Install NeMo AutoModel and follow the fine-tuning guide to configure a recipe for this model.

```bash
pip install nemo-automodel
```

:::{dropdown} Run with Docker
```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [nvidia/Minitron-8B-Base](https://huggingface.co/nvidia/Minitron-8B-Base)
