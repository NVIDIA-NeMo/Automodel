# Qwen3-Omni

[Qwen3-Omni](https://qwenlm.github.io/blog/qwen3/) is Alibaba Cloud's omnimodal model supporting text, image, audio, and video inputs in a single unified architecture with a MoE language backbone.

:::{card}
| | |
|---|---|
| **Task** | Omnimodal (Text·Image·Audio·Video) |
| **Architecture** | `Qwen3OmniForConditionalGeneration` |
| **Parameters** | 30B total / 3B active |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen3-Omni-30B-A3B-Instruct**: 30B total, 3B activated (MoE)

## Architecture

- `Qwen3OmniForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen3-Omni 30B A3B Instruct | [`Qwen/Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`qwen3_omni_moe_30b_te_deepep.yaml <../../../../examples/vlm_finetune/qwen3/qwen3_omni_moe_30b_te_deepep.yaml>` | MedPix-VQA | SFT — Qwen3-Omni 30B with TE + DeepEP |


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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen3/qwen3_omni_moe_30b_te_deepep.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/qwen3/qwen3_omni_moe_30b_te_deepep.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [Omni Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM / Omni Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
