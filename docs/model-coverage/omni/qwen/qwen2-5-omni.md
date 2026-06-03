# Qwen2.5-Omni

[Qwen2.5-Omni](https://qwenlm.github.io/blog/qwen2.5-omni/) is Alibaba Cloud's omnimodal model supporting text, image, audio, and video inputs in a single unified architecture with a dense language backbone. NeMo AutoModel onboards the **Thinker** stack for audio understanding tasks such as automatic speech recognition (ASR).

:::{card}
| | |
|---|---|
| **Task** | Omnimodal (Text·Image·Audio·Video) |
| **Architecture** | `Qwen2_5OmniForConditionalGeneration` |
| **Parameters** | 3B / 7B (dense) |
| **HF Org** | [Qwen](https://huggingface.co/Qwen) |
:::

## Available Models

- **Qwen2.5-Omni-3B**: 3B dense backbone
- **Qwen2.5-Omni-7B**: 7B dense backbone

## Architecture

The registry wires the Qwen2.5-Omni Thinker backbone under the following architecture keys:

- `Qwen2_5OmniForConditionalGeneration`
- `Qwen2_5OmniModel`
- `Qwen2_5OmniThinkerForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Qwen2.5-Omni 3B | [`Qwen/Qwen2.5-Omni-3B`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |
| Qwen2.5-Omni 7B | [`Qwen/Qwen2.5-Omni-7B`](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`ami_sft_3b.yaml <../../../../examples/audio_finetune/qwen2_5_omni_asr/ami_sft_3b.yaml>` | AMI | ASR SFT — Qwen2.5-Omni 3B on the AMI meeting corpus |
| {download}`ami_sft_7b.yaml <../../../../examples/audio_finetune/qwen2_5_omni_asr/ami_sft_7b.yaml>` | AMI | ASR SFT — Qwen2.5-Omni 7B on the AMI meeting corpus |


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
automodel --nproc-per-node=8 examples/audio_finetune/qwen2_5_omni_asr/ami_sft_3b.yaml
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
automodel --nproc-per-node=8 examples/audio_finetune/qwen2_5_omni_asr/ami_sft_3b.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [Omni Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM / Omni Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
- [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
