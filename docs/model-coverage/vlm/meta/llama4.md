# Llama 4

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) is Meta's first natively multimodal model family. Llama 4 Scout and Maverick are MoE models supporting interleaved image and text inputs.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Llama4ForConditionalGeneration` |
| **Parameters** | 17B active (MoE) |
| **HF Org** | [meta-llama](https://huggingface.co/meta-llama) |
:::

## Available Models

- **Llama-4-Scout-17B-16E-Instruct**: 17B active / 16 experts
- **Llama-4-Maverick-17B-128E-Instruct**: 17B active / 128 experts

## Architecture

- `Llama4ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Llama-4-Scout-17B-16E-Instruct | [`meta-llama/Llama-4-Scout-17B-16E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |
| Llama-4-Maverick-17B-128E-Instruct | [`meta-llama/Llama-4-Maverick-17B-128E-Instruct`](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |


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

**3. Fine-tune** by adapting a base VLM recipe — override the model ID on the CLI:

```bash
automodel --nproc-per-node=8 examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml \
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
automodel --nproc-per-node=8 examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml \
  --model.pretrained_model_name_or_path <MODEL_HF_ID>
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)
