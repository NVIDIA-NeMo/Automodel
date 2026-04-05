# Jais

[Jais](https://huggingface.co/inceptionai/jais-13b) is an Arabic-English bilingual language model from Inception (formerly G42/Inception AI), trained on a large Arabic and English corpus.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `JAISLMHeadModel` |
| **Parameters** | 13B – 30B |
| **HF Org** | [inceptionai](https://huggingface.co/inceptionai) |
:::

## Available Models

- **jais-30b-chat-v3**: 30B
- **jais-30b-v3**: 30B base
- **jais-13b-chat**: 13B
- **jais-13b**: 13B base

## Architecture

- `JAISLMHeadModel`

## Example HF Models

| Model | HF ID |
|---|---|
| Jais 13B | `inceptionai/jais-13b` |
| Jais 13B Chat | `inceptionai/jais-13b-chat` |
| Jais 30B v3 | `inceptionai/jais-30b-v3` |
| Jais 30B Chat v3 | `inceptionai/jais-30b-chat-v3` |


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

- [inceptionai/jais-13b](https://huggingface.co/inceptionai/jais-13b)
