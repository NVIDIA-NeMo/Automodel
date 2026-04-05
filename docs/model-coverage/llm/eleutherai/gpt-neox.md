# GPT-NeoX / Pythia

[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) is EleutherAI's large-scale language model architecture. The same `GPTNeoXForCausalLM` architecture is used by the Pythia scaling suite, OpenAssistant, Databricks Dolly, and StableLM models.

## Available Models

- **GPT-NeoX-20B** (EleutherAI)
- **Pythia** suite: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B (EleutherAI)
- **OA-SFT-Pythia-12B** (OpenAssistant)
- **Dolly-v2-12B** (Databricks)
- **StableLM-tuned-alpha-7B** (Stability AI)

## Architecture

- `GPTNeoXForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| GPT-NeoX 20B | `EleutherAI/gpt-neox-20b` |
| Pythia 12B | `EleutherAI/pythia-12b` |
| OpenAssistant SFT Pythia 12B | `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5` |
| Dolly v2 12B | `databricks/dolly-v2-12b` |
| StableLM tuned alpha 7B | `stabilityai/stablelm-tuned-alpha-7b` |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/EleutherAI/gpt-neox-20b
- https://huggingface.co/EleutherAI/pythia-12b
