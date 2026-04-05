# GPT-OSS

[GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) is OpenAI's open-weight model family featuring QuickGELU activations and activation clamping for training stability.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GptOssForCausalLM` |
| **Parameters** | 20B – 120B |
| **HF Org** | [openai](https://huggingface.co/openai) |
:::

## Available Models

- **gpt-oss-20b**: 20B parameters
- **gpt-oss-120b**: 120B parameters

## Architecture

- `GptOssForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| GPT-OSS 20B | `openai/gpt-oss-20b` |
| GPT-OSS 120B | `openai/gpt-oss-120b` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`gpt_oss_20b.yaml <../../../examples/llm_finetune/gpt_oss/gpt_oss_20b.yaml>` | SFT — GPT-OSS 20B |
| {download}`gpt_oss_120b.yaml <../../../examples/llm_finetune/gpt_oss/gpt_oss_120b.yaml>` | SFT — GPT-OSS 120B |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/gpt_oss/gpt_oss_20b.yaml
```

:::{dropdown} Run with Docker
Pull the NeMo AutoModel container and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

Then inside the container:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/gpt_oss/gpt_oss_20b.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
