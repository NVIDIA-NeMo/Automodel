# GPT-2

[GPT-2](https://huggingface.co/openai-community/gpt2) is OpenAI's foundational decoder-only transformer. NeMo AutoModel uses it as a baseline for the Megatron pretraining smoke test and tutorials — its small footprint makes it a convenient target to validate data pipelines, distributed configs, and logging without needing large compute.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (pretraining baseline) |
| **Architecture** | `GPT2LMHeadModel` |
| **Parameters** | 124M – 1.5B |
| **HF Org** | [openai-community](https://huggingface.co/openai-community) |
:::

## Available Models

- **gpt2** (124M)
- **gpt2-medium** (355M)
- **gpt2-large** (774M)
- **gpt2-xl** (1.5B)

## Architecture

- `GPT2LMHeadModel`

## Example HF Models

| Model | HF ID |
|---|---|
| GPT-2 | [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`megatron_pretrain_gpt2.yaml <../../../../examples/llm_pretrain/megatron_pretrain_gpt2.yaml>` | Megatron pretraining smoke test — GPT-2 on FineWeb-Edu |


## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_pretrain/megatron_pretrain_gpt2.yaml
```

See the [Installation Guide](../../../guides/installation.md) and [LLM Pretraining Guide](../../../guides/llm/pretrain.md).

## Hugging Face Model Cards

- [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
