# LLaVA

[LLaVA](https://llava-vl.github.io/) (Large Language and Vision Assistant) is a pioneering open-source multimodal model connecting a vision encoder to a language model via a projection layer. Multiple versions and variants are supported via the `llava-hf` organization on Hugging Face.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text |
| **Architecture** | `LlavaForConditionalGeneration` / `LlavaNextForConditionalGeneration` |
| **Parameters** | 7B – 34B |
| **HF Org** | [llava-hf](https://huggingface.co/llava-hf) |
:::

## Available Models

- **LLaVA-1.5** (`LlavaForConditionalGeneration`): 7B, 13B
- **LLaVA-1.6 / LLaVA-NeXT** (`LlavaNextForConditionalGeneration`): 7B, 34B
- **LLaVA-NeXT-Video** (`LlavaNextVideoForConditionalGeneration`): 7B
- **LLaVA-OneVision** (`LlavaOnevisionForConditionalGeneration`): 7B

## Architectures

- `LlavaForConditionalGeneration` — LLaVA 1.5
- `LlavaNextForConditionalGeneration` — LLaVA-NeXT / 1.6
- `LlavaNextVideoForConditionalGeneration` — LLaVA-NeXT-Video
- `LlavaOnevisionForConditionalGeneration` — LLaVA-OneVision

## Example HF Models

| Model | HF ID |
|---|---|
| LLaVA 1.5 7B | [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| LLaVA 1.5 13B | [`llava-hf/llava-1.5-13b-hf`](https://huggingface.co/llava-hf/llava-1.5-13b-hf) |
| LLaVA-NeXT Mistral 7B | [`llava-hf/llava-v1.6-mistral-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) |
| LLaVA-NeXT 34B | [`llava-hf/llava-v1.6-34b-hf`](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) |
| LLaVA-NeXT-Video 7B | [`llava-hf/LLaVA-NeXT-Video-7B-hf`](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) |
| LLaVA-OneVision 7B | [`llava-hf/llava-onevision-qwen2-7b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) |


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

- [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)
