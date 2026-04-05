# Phi-3-Small

[Phi-3-Small](https://azure.microsoft.com/en-us/products/phi) is Microsoft's 7B model using a distinct `Phi3SmallForCausalLM` architecture with blocksparse attention, separate from the standard Phi-3 family.

## Available Models

- **Phi-3-small-8k-instruct**: 7B, 8K context
- **Phi-3-small-128k-instruct**: 7B, 128K context

## Architecture

- `Phi3SmallForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Phi-3-small-8k-instruct | `microsoft/Phi-3-small-8k-instruct` |
| Phi-3-small-128k-instruct | `microsoft/Phi-3-small-128k-instruct` |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/microsoft/Phi-3-small-8k-instruct
- https://huggingface.co/microsoft/Phi-3-small-128k-instruct
