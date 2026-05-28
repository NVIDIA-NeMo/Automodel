# MiniCPM

[MiniCPM](https://github.com/OpenBMB/MiniCPM) is a compact language model series from OpenBMB / Tsinghua University, designed to deliver strong performance at small parameter counts using model merging and continuous training techniques.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `MiniCPMForCausalLM` / `MiniCPM3ForCausalLM` / `MiniCPM5ForCausalLM` |
| **Parameters** | 1B – 4B |
| **HF Org** | [openbmb](https://huggingface.co/openbmb) |
:::

## Available Models

- **MiniCPM5-1B** (`MiniCPM5ForCausalLM`): 1B
- **MiniCPM3-4B** (`MiniCPM3ForCausalLM`): 4B
- **MiniCPM-2B-sft-bf16** (`MiniCPMForCausalLM`): 2B, SFT
- **MiniCPM-2B-dpo-bf16** (`MiniCPMForCausalLM`): 2B, DPO

## Architectures

- `MiniCPMForCausalLM` — MiniCPM v1/v2
- `MiniCPM3ForCausalLM` — MiniCPM3
- `MiniCPM5ForCausalLM` — MiniCPM5

## Example HF Models

| Model | HF ID |
|---|---|
| MiniCPM5 1B | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) |
| MiniCPM 2B SFT | [`openbmb/MiniCPM-2B-sft-bf16`](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) |
| MiniCPM3 4B | [`openbmb/MiniCPM3-4B`](https://huggingface.co/openbmb/MiniCPM3-4B) |


## Try with NeMo AutoModel

Install NeMo AutoModel and follow the fine-tuning guide to configure a recipe for this model.

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get example recipes you can adapt:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Fine-tune** using one of the provided MiniCPM5 recipes:

```bash
# Full fine-tuning
automodel examples/llm_finetune/minicpm/minicpm5_1b_squad.yaml --nproc-per-node 8

# LoRA fine-tuning
automodel examples/llm_finetune/minicpm/minicpm5_1b_squad_peft.yaml --nproc-per-node 8
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** The recipes are at `/opt/Automodel/examples/` — navigate there:

```bash
cd /opt/Automodel
```

**3. Fine-tune**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
  --model.pretrained_model_name_or_path <MODEL_HF_ID>
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
- [openbmb/MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
- [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B)
