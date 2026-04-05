# Aquila / Aquila2

[Aquila](https://huggingface.co/BAAI/Aquila-7B) is a Chinese-English bilingual language model from the Beijing Academy of Artificial Intelligence (BAAI).

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `AquilaForCausalLM` |
| **Parameters** | 7B – 34B |
| **HF Org** | [BAAI](https://huggingface.co/BAAI) |
:::

## Available Models

- **Aquila-7B**
- **AquilaChat-7B**: instruction-tuned
- **Aquila2-34B**

## Architecture

- `AquilaForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Aquila 7B | `BAAI/Aquila-7B` |
| AquilaChat 7B | `BAAI/AquilaChat-7B` |


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

- [BAAI/Aquila-7B](https://huggingface.co/BAAI/Aquila-7B)
