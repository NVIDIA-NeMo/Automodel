# DeepSeek

[DeepSeek](https://github.com/deepseek-ai) is a series of open-weight language models from DeepSeek AI. The first-generation models (V1/V2) use standard transformer decoder and Multi-head Latent Attention architectures.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `DeepseekForCausalLM` |
| **Parameters** | 7B – 67B |
| **HF Org** | [deepseek-ai](https://huggingface.co/deepseek-ai) |
:::

## Available Models

- **DeepSeek-V2**: 236B total, 21B activated (MoE)
- **DeepSeek-V2-Chat**: instruction-tuned variant
- **DeepSeek-LLM 7B/67B**: dense models

## Architecture

- `DeepseekForCausalLM` — DeepSeek v1/v2 dense models

## Example HF Models

| Model | HF ID |
|---|---|
| DeepSeek LLM 7B Chat | `deepseek-ai/deepseek-llm-7b-chat` |
| DeepSeek LLM 67B Chat | `deepseek-ai/deepseek-llm-67b-chat` |


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

- [deepseek-ai/deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
- [deepseek-ai/deepseek-llm-67b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat)
