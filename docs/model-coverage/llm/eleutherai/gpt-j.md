# GPT-J

[GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) is a 6B parameter transformer language model trained by EleutherAI on the Pile dataset. It was one of the earliest large open-weight models.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GPTJForCausalLM` |
| **Parameters** | 6B |
| **HF Org** | [EleutherAI](https://huggingface.co/EleutherAI) |
:::

## Available Models

- **gpt-j-6b**: 6B parameters
- **gpt4all-j**: GPT-J fine-tuned for instruction following (Nomic AI)

## Architecture

- `GPTJForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| GPT-J 6B | `EleutherAI/gpt-j-6b` |
| GPT4All-J | `nomic-ai/gpt4all-j` |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)
