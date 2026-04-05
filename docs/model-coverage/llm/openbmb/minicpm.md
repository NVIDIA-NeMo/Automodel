# MiniCPM

[MiniCPM](https://github.com/OpenBMB/MiniCPM) is a compact language model series from OpenBMB / Tsinghua University, designed to deliver strong performance at small parameter counts using model merging and continuous training techniques.

## Available Models

- **MiniCPM3-4B** (`MiniCPM3ForCausalLM`): 4B
- **MiniCPM-2B-sft-bf16** (`MiniCPMForCausalLM`): 2B, SFT
- **MiniCPM-2B-dpo-bf16** (`MiniCPMForCausalLM`): 2B, DPO

## Architectures

- `MiniCPMForCausalLM` — MiniCPM v1/v2
- `MiniCPM3ForCausalLM` — MiniCPM3

## Example HF Models

| Model | HF ID |
|---|---|
| MiniCPM 2B SFT | `openbmb/MiniCPM-2B-sft-bf16` |
| MiniCPM3 4B | `openbmb/MiniCPM3-4B` |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
- https://huggingface.co/openbmb/MiniCPM3-4B
