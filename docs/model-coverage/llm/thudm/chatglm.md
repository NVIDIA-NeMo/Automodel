# ChatGLM

[ChatGLM](https://github.com/THUDM/ChatGLM-6B) is a bilingual (Chinese-English) conversational language model from Tsinghua University (THUDM). ChatGLM2 and ChatGLM3 extend the original with improved performance, longer context, and more efficient attention.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `ChatGLMModel` |
| **Parameters** | 6B |
| **HF Org** | [THUDM](https://huggingface.co/THUDM) |
:::

## Available Models

- **ChatGLM3-6B**
- **ChatGLM2-6B**

## Architecture

- `ChatGLMModel` / `ChatGLMForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| ChatGLM3 6B | `THUDM/chatglm3-6b` |
| ChatGLM2 6B | `THUDM/chatglm2-6b` |


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

- [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)
- [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
