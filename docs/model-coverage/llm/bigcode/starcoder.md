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

- [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
