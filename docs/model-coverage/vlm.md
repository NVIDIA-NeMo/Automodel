# Vision Language Models (VLMs)

## Introduction

Vision Language Models (VLMs) are advanced models that integrate vision and language processing capabilities. They are trained on extensive datasets containing both interleaved images and text data, allowing them to generate text descriptions of images and answer questions related to images.

NeMo Automodel LLM APIs can be easily extended to support VLM tasks. While most of the training setup is the same, some additional steps are required to prepare the data and model for VLM training.

## Run LLMs with NeMo Automodel

To run LLMs with NeMo Automodel, use NeMo container version `25.07` or later. If the model you want to fine-tune requires a newer version of Transformers, you may need to upgrade to the latest NeMo Automodel using:

```bash

   pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/Automodel.git
```

For other installation options (e.g., uv) please see our [Installation Guide](../guides/installation.md).

## Supported Models


The following VLM models from Hugging Face have been tested and support both Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) with LoRA:


| Model                              | Dataset                     | FSDP2      | PEFT       |
|------------------------------------|-----------------------------|------------|------------|
| Gemma 3-4B & 27B                   | naver-clova-ix & rdr-items  | Supported  | Supported  |
| Gemma 3n                           | naver-clova-ix & rdr-items  | Supported  | Supported  |
| Qwen2-VL-2B-Instruct & Qwen2.5-VL-3B-Instruct | cord-v2          | Supported  | Supported  |
| llava-v1.6                         | cord-v2 & naver-clova-ix    | Supported  | Supported  |

For detailed instructions on fine-tuning these models using both SFT and PEFT approaches, please refer to the [Gemma 3 and Gemma 3n Fine-Tuning Guide](../guides/omni/gemma3-3n.md). The guide covers dataset preparation, configuration, and running both full fine-tuning and LoRA-based parameter efficient fine-tuning.


## Dataset Examples

:::{tip}
In these guides, we use the `quintend/rdr-items` and `naver-clova-ix/cord-v2` datasets for demonstation purposes, but you can specify your own data as needed.
:::

### rdr items dataset
The rdr items dataset [`quintend/rdr-items`](https://huggingface.co/datasets/quintend/rdr-items) is a small dataset containing 48 images with descriptions. This dataset serves as an example of how to prepare image-text data for VLM fine-tuning. For complete instructions on dataset preprocessing and the collate functions used, see the [Gemma Fine-Tuning Guide](../guides/omni/gemma3-3n.md).

### cord-v2 dataset
The cord-v2 dataset [`naver-clova-ix/cord-v2`](https://huggingface.co/naver-clova-ix/cord-v2) contains receipts with descriptions in JSON format. This demonstrates handling structured data in VLMs. The [Gemma Fine-Tuning Guide](../guides/omni/gemma3-3n.md) provides detailed examples of custom preprocessing and collate functions for similar datasets.

## Train VLM Models
All supported models can be fine-tuned using either full SFT or PEFT approaches. The [Gemma Fine-Tuning Guide](../guides/omni/gemma3-3n.md) provides complete instructions for:
* Configuring YAML-based training.
* Running single-GPU and multi-GPU training.
* Setting up PEFT with LoRA.
* Model checkpointing and W&B integration.
