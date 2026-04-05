# Gemma

[Google's Gemma](https://ai.google.dev/gemma) is a family of open-weight language models built on the same research and technology as Gemini. Gemma models are available in multiple sizes and versions, with improvements in each generation including local sliding window attention (Gemma 2) and interleaved global/local attention (Gemma 3).

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GemmaForCausalLM` / `Gemma2ForCausalLM` / `Gemma3ForCausalLM` |
| **Parameters** | 1B – 27B |
| **HF Org** | [google](https://huggingface.co/google) |
:::

## Available Models

- **Gemma 3**: 1B, 4B, 12B, 27B
- **Gemma 2**: 2B, 9B, 27B
- **Gemma (v1)**: 2B, 7B

## Architectures

- `GemmaForCausalLM` — Gemma v1
- `Gemma2ForCausalLM` — Gemma 2
- `Gemma3ForCausalLM` — Gemma 3

## Example HF Models

| Model | HF ID |
|---|---|
| Gemma 1.1 2B IT | `google/gemma-1.1-2b-it` |
| Gemma 2B | `google/gemma-2b` |
| Gemma 2 9B IT | `google/gemma-2-9b-it` |
| Gemma 2 27B | `google/gemma-2-27b` |
| Gemma 3 1B IT | `google/gemma-3-1b-it` |
| Gemma 3 4B IT | `google/gemma-3-4b-it` |
| Gemma 3 27B IT | `google/gemma-3-27b-it` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`gemma_2_9b_it_squad.yaml`](../../../examples/llm_finetune/gemma/gemma_2_9b_it_squad.yaml) | SFT — Gemma 2 9B IT on SQuAD |
| [`gemma_2_9b_it_squad_peft.yaml`](../../../examples/llm_finetune/gemma/gemma_2_9b_it_squad_peft.yaml) | LoRA — Gemma 2 9B IT on SQuAD |
| [`gemma_3_270m_squad.yaml`](../../../examples/llm_finetune/gemma/gemma_3_270m_squad.yaml) | SFT — Gemma 3 270M on SQuAD |
| [`gemma_3_270m_squad_peft.yaml`](../../../examples/llm_finetune/gemma/gemma_3_270m_squad_peft.yaml) | LoRA — Gemma 3 270M on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) for full SFT and LoRA instructions.

## Hugging Face Model Cards

- [google/gemma-2b](https://huggingface.co/google/gemma-2b)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
