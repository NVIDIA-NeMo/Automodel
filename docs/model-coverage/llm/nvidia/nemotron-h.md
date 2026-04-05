# Nemotron-H

[NVIDIA Nemotron-H](https://developer.nvidia.com/blog/nemotron-h-efficient-hybrid-mamba-transformer-models/) is a hybrid Mamba-2 / transformer architecture that interleaves selective state space layers with standard attention layers for improved efficiency on long sequences.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `NemotronHForCausalLM` |
| **Parameters** | 9B – 30B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **NVIDIA-Nemotron-Nano-9B-v2**: 9B hybrid model
- **NVIDIA-Nemotron-Nano-12B-v2**: 12B hybrid model
- **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16**: 30B total, 3B activated (sparse MoE + Mamba-2)

## Architecture

- `NemotronHForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Nemotron-Nano 9B v2 | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` |
| Nemotron-Nano 12B v2 | `nvidia/NVIDIA-Nemotron-Nano-12B-v2` |
| Nemotron-3-Nano 30B A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |

## Example Recipes

| Recipe | Description |
|---|---|
| [`nemotron_nano_9b_squad.yaml`](../../../examples/llm_finetune/nemotron/nemotron_nano_9b_squad.yaml) | SFT — Nemotron-Nano 9B on SQuAD |
| [`nemotron_nano_9b_squad_peft.yaml`](../../../examples/llm_finetune/nemotron/nemotron_nano_9b_squad_peft.yaml) | LoRA — Nemotron-Nano 9B on SQuAD |
| [`nemotron_nano_v3_hellaswag.yaml`](../../../examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag.yaml) | SFT — Nemotron-3-Nano 30B on HellaSwag |
| [`nemotron_nano_v3_hellaswag_peft.yaml`](../../../examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml) | LoRA — Nemotron-3-Nano 30B on HellaSwag |

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [nvidia/NVIDIA-Nemotron-Nano-12B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2)
- [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
