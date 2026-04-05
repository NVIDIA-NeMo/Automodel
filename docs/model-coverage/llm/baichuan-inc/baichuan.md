# Baichuan / Baichuan2

[Baichuan](https://github.com/baichuan-inc/Baichuan2) is a Chinese-English bilingual language model series from Baichuan Inc., featuring strong Chinese language performance.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `BaiChuanForCausalLM` |
| **Parameters** | 7B – 13B |
| **HF Org** | [baichuan-inc](https://huggingface.co/baichuan-inc) |
:::

## Available Models

- **Baichuan2-13B-Chat**
- **Baichuan2-7B-Chat**
- **Baichuan-7B**

## Architecture

- `BaiChuanForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Baichuan2 13B Chat | `baichuan-inc/Baichuan2-13B-Chat` |
| Baichuan 7B | `baichuan-inc/Baichuan-7B` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`baichuan_2_7b_squad.yaml`](../../../examples/llm_finetune/baichuan/baichuan_2_7b_squad.yaml) | SFT — Baichuan2 7B on SQuAD |
| [`baichuan_2_7b_squad_peft.yaml`](../../../examples/llm_finetune/baichuan/baichuan_2_7b_squad_peft.yaml) | LoRA — Baichuan2 7B on SQuAD |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
