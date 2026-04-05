# LLaVA

[LLaVA](https://llava-vl.github.io/) (Large Language and Vision Assistant) is a pioneering open-source multimodal model connecting a vision encoder to a language model via a projection layer. Multiple versions and variants are supported via the `llava-hf` organization on Hugging Face.

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
| LLaVA 1.5 7B | `llava-hf/llava-1.5-7b-hf` |
| LLaVA 1.5 13B | `llava-hf/llava-1.5-13b-hf` |
| LLaVA-NeXT Mistral 7B | `llava-hf/llava-v1.6-mistral-7b-hf` |
| LLaVA-NeXT 34B | `llava-hf/llava-v1.6-34b-hf` |
| LLaVA-NeXT-Video 7B | `llava-hf/LLaVA-NeXT-Video-7B-hf` |
| LLaVA-OneVision 7B | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |

## Fine-Tuning

See the [VLM Fine-Tuning Guide](../../../guides/omni/gemma3-3n.md).

## Hugging Face Model Cards

- https://huggingface.co/llava-hf/llava-1.5-7b-hf
- https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
- https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf
