# Phi-3-Small

[Phi-3-Small](https://azure.microsoft.com/en-us/products/phi) is Microsoft's 7B model using a distinct `Phi3SmallForCausalLM` architecture with blocksparse attention, separate from the standard Phi-3 family.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Phi3SmallForCausalLM` |
| **Parameters** | 7B |
| **HF Org** | [microsoft](https://huggingface.co/microsoft) |
:::

## Available Models

- **Phi-3-small-8k-instruct**: 7B, 8K context
- **Phi-3-small-128k-instruct**: 7B, 128K context

## Architecture

- `Phi3SmallForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Phi-3-small-8k-instruct | `microsoft/Phi-3-small-8k-instruct` |
| Phi-3-small-128k-instruct | `microsoft/Phi-3-small-128k-instruct` |


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

- [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct)
- [microsoft/Phi-3-small-128k-instruct](https://huggingface.co/microsoft/Phi-3-small-128k-instruct)
