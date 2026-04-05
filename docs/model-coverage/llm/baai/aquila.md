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

- [BAAI/Aquila-7B](https://huggingface.co/BAAI/Aquila-7B)
