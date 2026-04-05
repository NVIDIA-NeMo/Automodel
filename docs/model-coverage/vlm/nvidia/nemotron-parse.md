# Nemotron-Parse

[Nemotron-Parse-v1.1](https://huggingface.co/nvidia/Nemotron-Parse-v1.1) is NVIDIA's document parsing VLM, specializing in extracting structured information from complex documents including tables, forms, and mixed-content PDFs.

:::{card}
| | |
|---|---|
| **Task** | Document Parsing |
| **Architecture** | `NemotronParseForConditionalGeneration` |
| **Parameters** | varies |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Nemotron-Parse-v1.1**

## Architecture

- `NemotronParseForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Nemotron-Parse v1.1 | `nvidia/Nemotron-Parse-v1.1` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`nemotron_parse_v1_1.yaml <../../../examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml>` | cord-v2 | SFT — Nemotron-Parse on CORD-v2 |


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
automodel --nproc-per-node=8 examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [nvidia/Nemotron-Parse-v1.1](https://huggingface.co/nvidia/Nemotron-Parse-v1.1)
