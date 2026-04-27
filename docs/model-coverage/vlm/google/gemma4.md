# Gemma 4

[Gemma 4](https://ai.google.dev/gemma) is Google's next-generation multimodal Gemma family, supporting image-text inputs with a Mixture-of-Experts (MoE) language backbone at larger scales. NeMo AutoModel replaces the HF-native dense matmul over experts with the NeMo `GroupedExperts` backend, enabling Expert Parallelism (EP) via the standard MoE parallelizer.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Gemma4ForConditionalGeneration` |
| **Parameters** | 2B – 31B (dense) · 26B-A4B (MoE) |
| **HF Org** | [google](https://huggingface.co/google) |
:::

## Available Models

- **Gemma 4 E2B IT** (VL, dense)
- **Gemma 4 E4B IT** (VL, dense, kv-shared layers)
- **Gemma 4 31B IT** (VL, dense)
- **Gemma 4 26B-A4B IT** (VL, MoE)

## Architecture

- `Gemma4ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Gemma 4 E2B IT | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| Gemma 4 E4B IT | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) |
| Gemma 4 31B IT | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| Gemma 4 26B-A4B IT (MoE) | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`gemma4_2b.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_2b.yaml>` | SFT — Gemma 4 E2B on MedPix |
| {download}`gemma4_2b_peft.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_2b_peft.yaml>` | LoRA — Gemma 4 E2B on MedPix |
| {download}`gemma4_4b.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_4b.yaml>` | SFT — Gemma 4 E4B on MedPix |
| {download}`gemma4_4b_peft.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_4b_peft.yaml>` | LoRA — Gemma 4 E4B on MedPix |
| {download}`gemma4_31b.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_31b.yaml>` | SFT — Gemma 4 31B on MedPix |
| {download}`gemma4_31b_peft.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_31b_peft.yaml>` | LoRA — Gemma 4 31B on MedPix |
| {download}`gemma4_31b_tp4.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_31b_tp4.yaml>` | SFT — Gemma 4 31B with TP=4 |
| {download}`gemma4_31b_tp4_pp2.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_31b_tp4_pp2.yaml>` | SFT — Gemma 4 31B with TP=4, PP=2 |
| {download}`gemma4_31b_tp4_pp4.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_31b_tp4_pp4.yaml>` | SFT — Gemma 4 31B with TP=4, PP=4 (multi-node) |
| {download}`gemma4_26b_a4b_moe.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_26b_a4b_moe.yaml>` | SFT — Gemma 4 26B-A4B MoE on MedPix |
| {download}`gemma4_26b_a4b_moe_peft.yaml <../../../../examples/vlm_finetune/gemma4/gemma4_26b_a4b_moe_peft.yaml>` | LoRA — Gemma 4 26B-A4B MoE on MedPix |


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
automodel --nproc-per-node=8 examples/vlm_finetune/gemma4/gemma4_4b.yaml
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
automodel --nproc-per-node=8 examples/vlm_finetune/gemma4/gemma4_4b.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)
- [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
