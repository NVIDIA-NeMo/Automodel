# StableLM

[StableLM](https://stability.ai/stable-lm) is Stability AI's series of open language models, available in base and instruction-tuned variants across multiple sizes.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `StableLmForCausalLM` |
| **Parameters** | 3B – 7B |
| **HF Org** | [stabilityai](https://huggingface.co/stabilityai) |
:::

## Available Models

- **stablelm-3b-4e1t**: 3B
- **stablelm-base-alpha-7b-v2**: 7B

## Architecture

- `StableLmForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| StableLM 3B 4E1T | `stabilityai/stablelm-3b-4e1t` |
| StableLM Base Alpha 7B v2 | `stabilityai/stablelm-base-alpha-7b-v2` |


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

- [stabilityai/stablelm-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t)
