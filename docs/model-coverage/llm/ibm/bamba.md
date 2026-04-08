# Bamba

[Bamba](https://huggingface.co/ibm-ai-platform/Bamba-9B) is a hybrid SSM-attention language model from IBM, combining Mamba-2 selective state space layers with standard transformer attention for efficient long-context processing.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `BambaForCausalLM` |
| **Parameters** | 9B |
| **HF Org** | [ibm-ai-platform](https://huggingface.co/ibm-ai-platform) |
:::

## Available Models

- **Bamba-9B**: 9B parameters

## Architecture

- `BambaForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Bamba 9B | [`ibm-ai-platform/Bamba-9B`](https://huggingface.co/ibm-ai-platform/Bamba-9B) |


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

**3. Fine-tune** by adapting a base LLM recipe — override the model ID on the CLI:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
  --model.pretrained_model_name_or_path <MODEL_HF_ID>
```

Replace `<MODEL_HF_ID>` with the model ID from **Example HF Models** above.

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

- [ibm-ai-platform/Bamba-9B](https://huggingface.co/ibm-ai-platform/Bamba-9B)
