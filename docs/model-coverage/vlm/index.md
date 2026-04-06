# Vision Language Models (VLMs)

## Introduction

Vision Language Models (VLMs) integrate vision and language processing capabilities, enabling models to understand images and generate text descriptions, answer visual questions, and perform multimodal reasoning.

NeMo AutoModel LLM APIs can be easily extended to support VLM tasks. While most of the training setup is the same as for LLMs, some additional steps are required to prepare the data and model for VLM training.

## Run VLMs with NeMo AutoModel

To run VLMs with NeMo AutoModel, use NeMo container version [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) or later. If the model you want to fine-tune requires a newer version of Transformers, you may need to upgrade:

```bash
pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/AutoModel.git
```

For other installation options, see our [Installation Guide](../../guides/installation.md).

## Supported Models

NeMo AutoModel supports [AutoModelForImageTextToText](https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForImageTextToText) in the [Image-Text-to-Text](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending) category.

| Owner | Model | Architectures |
|---|---|---|
| Moonshot AI | [Kimi-VL](moonshotai/kimi-vl.md) | `KimiVLForConditionalGeneration` |
| Google | [Gemma 3 VL / Gemma 3n](google/gemma3-vl.md) | `Gemma3ForConditionalGeneration` |
| Qwen / Alibaba Cloud | [Qwen2.5-VL](qwen/qwen2-5-vl.md) | `Qwen2VLForConditionalGeneration`, `Qwen2_5VLForConditionalGeneration` |
| Qwen / Alibaba Cloud | [Qwen3-VL / Qwen3-VL-MoE](qwen/qwen3-vl.md) | `Qwen3VLForConditionalGeneration` |
| Qwen / Alibaba Cloud | [Qwen3.5-VL](qwen/qwen3-5-vl.md) | `Qwen3_5VLForConditionalGeneration`, `Qwen3_5MoeVLForConditionalGeneration` |
| NVIDIA | [Nemotron-Parse](nvidia/nemotron-parse.md) | `NemotronParseForConditionalGeneration` |
| Mistral AI | [Ministral3 VL](mistralai/ministral3-vl.md) | `Mistral3ForConditionalGeneration` |
| Mistral AI | [Mistral-Small-4](mistralai/mistral-small-4.md) | `MistralForConditionalGeneration` |
| InternLM / Shanghai AI Lab | [InternVL](internlm/internvl.md) | `InternVLForConditionalGeneration` |
| Meta | [Llama 4](meta/llama4.md) | `Llama4ForConditionalGeneration` |
| HuggingFace | [SmolVLM](huggingface/smolvlm.md) | `SmolVLMForConditionalGeneration` |
| LLaVA | [LLaVA](llava-hf/llava.md) | `LlavaForConditionalGeneration`, `LlavaNextForConditionalGeneration`, `LlavaNextVideoForConditionalGeneration`, `LlavaOnevisionForConditionalGeneration` |

## Fine-Tuning

All supported models can be fine-tuned using either full SFT or PEFT (LoRA) approaches. See the [Gemma 3 Fine-Tuning Guide](../../guides/omni/gemma3-3n.md) for a complete walkthrough covering dataset preparation, configuration, and multi-GPU training.

:::{tip}
In these guides, we use the `quintend/rdr-items` and `naver-clova-ix/cord-v2` datasets for demonstration purposes. Update the recipe YAML `dataset` section to use your own data. See [VLM datasets](../../guides/vlm/dataset.md) and [dataset overview](../../guides/dataset-overview.md).
:::

```{toctree}
:hidden:

moonshotai/kimi-vl
google/gemma3-vl
qwen/qwen2-5-vl
qwen/qwen3-vl
qwen/qwen3-5-vl
nvidia/nemotron-parse
mistralai/ministral3-vl
mistralai/mistral-small-4
internlm/internvl
meta/llama4
huggingface/smolvlm
llava-hf/llava
```
