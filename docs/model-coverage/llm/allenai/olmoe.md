# OLMoE

[OLMoE](https://allenai.org/olmo) is Allen AI's open Mixture-of-Experts language model. It activates 1B parameters per token from a 7B total parameter pool.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `OLMoEForCausalLM` |
| **Parameters** | 7B total / 1B active |
| **HF Org** | [allenai](https://huggingface.co/allenai) |
:::

## Available Models

- **OLMoE-1B-7B-0924**: 7B total, 1B activated
- **OLMoE-1B-7B-0924-Instruct**: instruction-tuned variant

## Architecture

- `OLMoEForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| OLMoE 1B 7B | `allenai/OLMoE-1B-7B-0924` |
| OLMoE 1B 7B Instruct | `allenai/OLMoE-1B-7B-0924-Instruct` |


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
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
