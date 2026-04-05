# GPT-J

[GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) is a 6B parameter transformer language model trained by EleutherAI on the Pile dataset. It was one of the earliest large open-weight models.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GPTJForCausalLM` |
| **Parameters** | 6B |
| **HF Org** | [EleutherAI](https://huggingface.co/EleutherAI) |
:::

## Available Models

- **gpt-j-6b**: 6B parameters
- **gpt4all-j**: GPT-J fine-tuned for instruction following (Nomic AI)

## Architecture

- `GPTJForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| GPT-J 6B | `EleutherAI/gpt-j-6b` |
| GPT4All-J | `nomic-ai/gpt4all-j` |


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

- [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)
