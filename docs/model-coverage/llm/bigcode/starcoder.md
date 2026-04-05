# StarCoder

[StarCoder](https://huggingface.co/blog/starcoder) is BigCode's code language model trained on the Stack dataset. It uses Multi-Query Attention and Fill-in-the-Middle (FIM) objectives. WizardCoder also uses this architecture.

:::{card}
| | |
|---|---|
| **Task** | Code Generation |
| **Architecture** | `GPTBigCodeForCausalLM` |
| **Parameters** | 1B – 15.5B |
| **HF Org** | [bigcode](https://huggingface.co/bigcode) |
:::

## Available Models

- **StarCoder**: 15.5B
- **gpt_bigcode-santacoder**: 1.1B
- **WizardCoder-15B-V1.0** (WizardLM)

## Architecture

- `GPTBigCodeForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| StarCoder | `bigcode/starcoder` |
| SantaCoder | `bigcode/gpt_bigcode-santacoder` |
| WizardCoder 15B | `WizardLM/WizardCoder-15B-V1.0` |


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

- [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
