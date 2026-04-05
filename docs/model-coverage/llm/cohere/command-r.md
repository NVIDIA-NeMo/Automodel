# Command-R

[Cohere Command-R](https://cohere.com/command) is a series of enterprise-grade language models optimized for retrieval-augmented generation (RAG) and tool use. Command-R7B uses the updated `Cohere2ForCausalLM` architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `CohereForCausalLM` / `Cohere2ForCausalLM` |
| **Parameters** | 7B – 104B |
| **HF Org** | [CohereForAI](https://huggingface.co/CohereForAI) |
:::

## Available Models

- **c4ai-command-r-v01**: 35B
- **c4ai-command-r-plus**: 104B
- **c4ai-command-r7b-12-2024**: 7B (`Cohere2ForCausalLM`)

## Architectures

- `CohereForCausalLM` — Command-R v01, Plus
- `Cohere2ForCausalLM` — Command-R7B

## Example HF Models

| Model | HF ID |
|---|---|
| Command-R v01 | `CohereForAI/c4ai-command-r-v01` |
| Command-R7B | `CohereForAI/c4ai-command-r7b-12-2024` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`cohere_command_r_7b_squad.yaml <../../../examples/llm_finetune/cohere/cohere_command_r_7b_squad.yaml>` | SFT — Command-R 7B on SQuAD |
| {download}`cohere_command_r_7b_squad_peft.yaml <../../../examples/llm_finetune/cohere/cohere_command_r_7b_squad_peft.yaml>` | LoRA — Command-R 7B on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [CohereForAI/c4ai-command-r7b-12-2024](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024)
