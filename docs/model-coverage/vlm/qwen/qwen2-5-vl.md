# Qwen2.5-VL

[Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/) is Alibaba Cloud's vision language model series supporting image and video understanding. It features dynamic resolution processing and integrates with the Qwen2.5 language backbone.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Qwen2_5VLForConditionalGeneration` |
| **Parameters** | 2B – 72B |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen2.5-VL-72B-Instruct**
- **Qwen2.5-VL-32B-Instruct**
- **Qwen2.5-VL-7B-Instruct**
- **Qwen2.5-VL-3B-Instruct**
- **Qwen2-VL-7B-Instruct**, **Qwen2-VL-2B-Instruct** (Qwen2 VL)

## Architectures

- `Qwen2_5VLForConditionalGeneration` — Qwen2.5-VL
- `Qwen2VLForConditionalGeneration` — Qwen2-VL

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen2.5-VL 3B Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Qwen2.5-VL 7B Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen2-VL 7B Instruct | `Qwen/Qwen2-VL-7B-Instruct` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`qwen2_5_vl_3b_rdr.yaml <../../../examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml>` | rdr-items | SFT — Qwen2.5-VL 3B on RDR Items |


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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
