# Falcon

[Falcon](https://falconllm.tii.ae/) is a series of open language models from the Technology Innovation Institute (TII) in Abu Dhabi, known for being trained on a high-quality curated web corpus (RefinedWeb).

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `FalconForCausalLM` |
| **Parameters** | 7B – 40B |
| **HF Org** | [tiiuae](https://huggingface.co/tiiuae) |
:::

## Available Models

- **Falcon-40B**, **Falcon-40B-Instruct**
- **Falcon-7B**, **Falcon-7B-Instruct**
- **Falcon-RW-7B**
- **Falcon3-7B-Instruct**

## Architecture

- `FalconForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Falcon 7B | `tiiuae/falcon-7b` |
| Falcon 40B | `tiiuae/falcon-40b` |
| Falcon RW 7B | `tiiuae/falcon-rw-7b` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`falcon3_7b_instruct_squad.yaml <../../../examples/llm_finetune/falcon/falcon3_7b_instruct_squad.yaml>` | SFT — Falcon3 7B Instruct on SQuAD |
| {download}`falcon3_7b_instruct_squad_peft.yaml <../../../examples/llm_finetune/falcon/falcon3_7b_instruct_squad_peft.yaml>` | LoRA — Falcon3 7B Instruct on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)
