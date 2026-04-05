# Phi-4-multimodal

[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) is Microsoft's multimodal extension of Phi-4, supporting text, image, and audio inputs for versatile edge and cloud deployment.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `Phi4MultimodalForCausalLM` |
| **Parameters** | 5.6B |
| **HF Org** | [microsoft](https://huggingface.co/microsoft) |
:::

## Available Models

- **Phi-4-multimodal-instruct**: 5.6B

## Architecture

- `Phi4MultimodalForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Phi-4-multimodal-instruct | `microsoft/Phi-4-multimodal-instruct` |

## Example Recipes

| Recipe | Dataset | Description |
|---|---|---|
| {download}`phi4_mm_cv17.yaml <../../../examples/vlm_finetune/phi4/phi4_mm_cv17.yaml>` | CommonVoice 17 | SFT — Phi-4-multimodal on CommonVoice (audio-text) |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/vlm_finetune/phi4/phi4_mm_cv17.yaml
```

:::{dropdown} Run with Docker
Pull the NeMo AutoModel container and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

Then inside the container:

```bash
automodel --nproc-per-node=8 examples/vlm_finetune/phi4/phi4_mm_cv17.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
