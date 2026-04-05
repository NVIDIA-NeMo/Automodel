# EXAONE

[EXAONE](https://www.lgresearch.ai/exaone) is a bilingual (Korean-English) language model series from LG AI Research, with strong performance on Korean-language benchmarks.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `ExaoneForCausalLM` |
| **Parameters** | 7.8B |
| **HF Org** | [LGAI-EXAONE](https://huggingface.co/LGAI-EXAONE) |
:::

## Available Models

- **EXAONE-3.0-7.8B-Instruct**
- **EXAONE-3.5-7.8B-Instruct**

## Architecture

- `ExaoneForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| EXAONE 3.0 7.8B Instruct | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` |


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

- [LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
