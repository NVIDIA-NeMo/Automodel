# Phi-3 / Phi-4

[Phi-3](https://azure.microsoft.com/en-us/products/phi) and [Phi-4](https://azure.microsoft.com/en-us/products/phi) are Microsoft's high-capability small language models using a shared transformer decoder architecture (`Phi3ForCausalLM`). Phi-4-mini and Phi-4 achieve strong benchmark results at relatively small parameter counts.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Phi3ForCausalLM` |
| **Parameters** | 3.8B – 14B |
| **HF Org** | [microsoft](https://huggingface.co/microsoft) |
:::

## Available Models

- **Phi-4**: 14B
- **Phi-4-mini-instruct**: 3.8B
- **Phi-3.5-mini-instruct**: 3.8B
- **Phi-3-medium-128k-instruct**: 14B
- **Phi-3-mini-128k-instruct**: 3.8B
- **Phi-3-mini-4k-instruct**: 3.8B

## Architecture

- `Phi3ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Phi-4 | [`microsoft/Phi-4`](https://huggingface.co/microsoft/Phi-4) |
| Phi-4-mini-instruct | [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Phi-3-mini-4k-instruct | [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Phi-3-mini-128k-instruct | [`microsoft/Phi-3-mini-128k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) |
| Phi-3-medium-128k-instruct | [`microsoft/Phi-3-medium-128k-instruct`](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`phi_4_squad.yaml <../../../../examples/llm_finetune/phi/phi_4_squad.yaml>` | SFT — Phi-4 on SQuAD |
| {download}`phi_4_squad_peft.yaml <../../../../examples/llm_finetune/phi/phi_4_squad_peft.yaml>` | LoRA — Phi-4 on SQuAD |
| {download}`phi_3_mini_it_squad.yaml <../../../../examples/llm_finetune/phi/phi_3_mini_it_squad.yaml>` | SFT — Phi-3-mini Instruct on SQuAD |
| {download}`phi_3_mini_it_squad_peft.yaml <../../../../examples/llm_finetune/phi/phi_3_mini_it_squad_peft.yaml>` | LoRA — Phi-3-mini Instruct on SQuAD |


## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/phi/phi_4_squad.yaml
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** Navigate to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/phi/phi_4_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [microsoft/Phi-4](https://huggingface.co/microsoft/Phi-4)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
