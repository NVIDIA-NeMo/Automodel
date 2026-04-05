# MiniCPM

[MiniCPM](https://github.com/OpenBMB/MiniCPM) is a compact language model series from OpenBMB / Tsinghua University, designed to deliver strong performance at small parameter counts using model merging and continuous training techniques.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `MiniCPMForCausalLM` / `MiniCPM3ForCausalLM` |
| **Parameters** | 2B – 4B |
| **HF Org** | [openbmb](https://huggingface.co/openbmb) |
:::

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

- [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
- [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B)
