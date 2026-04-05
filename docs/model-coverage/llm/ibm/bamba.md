# Bamba

[Bamba](https://huggingface.co/ibm-ai-platform/Bamba-9B) is a hybrid SSM-attention language model from IBM, combining Mamba-2 selective state space layers with standard transformer attention for efficient long-context processing.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `BambaForCausalLM` |
| **Parameters** | 9B |
| **HF Org** | [ibm-ai-platform](https://huggingface.co/ibm-ai-platform) |
:::

## Available Models

- **Bamba-9B**: 9B parameters

## Architecture

- `BambaForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Bamba 9B | `ibm-ai-platform/Bamba-9B` |


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

- [ibm-ai-platform/Bamba-9B](https://huggingface.co/ibm-ai-platform/Bamba-9B)
