# Vision Language Models (VLMs)

## Introduction

Vision Language Models (VLMs) are advanced models that integrate vision and language processing capabilities. They are trained on extensive datasets containing both interleaved images and text data, allowing them to generate text descriptions of images and answer questions related to images.

NeMo Automodel LLM APIs can be easily extended to support VLM tasks. While most of the training setup is the same, some additional steps are required to prepare the data and model for VLM training.

## Run VLMs with NeMo Automodel

To run VLMs with NeMo Automodel, use NeMo container version [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) or later. If the model you want to fine-tune requires a newer version of Transformers, you may need to upgrade to the latest NeMo Automodel using:

```bash

   pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/Automodel.git
```

For other installation options (e.g., uv) please see our [Installation Guide](../guides/installation.md).

## Supported Models


NeMo Automodel supports [AutoModelForImageTextToText](https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForImageTextToText) in the [Image-Text-to-Text](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending) category. Specifically, the following VLM models from Hugging Face have been tested and support both Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) with LoRA:


| Model                              | Dataset                     | FSDP2      | PEFT       | Example YAML |
|------------------------------------|-----------------------------|------------|------------|--------------|
| Kimi-VL-A3B-Instruct & Kimi-K25-VL | cord-v2, MedPix-VQA          | Supported  | Supported  | [kimi2vl_cordv2.yaml](../../examples/vlm_finetune/kimi/kimi2vl_cordv2.yaml), [kimi25vl_medpix.yaml](../../examples/vlm_finetune/kimi/kimi25vl_medpix.yaml) |
| Gemma 3-4B & 27B                   | cord-v2, MedPix-VQA          | Supported  | Supported  | [gemma3_vl_4b_cord_v2.yaml](../../examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml), [gemma3_vl_4b_cord_v2_peft.yaml](../../examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_peft.yaml), [gemma3_vl_4b_cord_v2_megatron_fsdp.yaml](../../examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2_megatron_fsdp.yaml), [gemma3_vl_4b_medpix.yaml](../../examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix.yaml), [gemma3_vl_4b_medpix_peft.yaml](../../examples/vlm_finetune/gemma3/gemma3_vl_4b_medpix_peft.yaml) |
| Gemma 3n                           | MedPix-VQA                   | Supported  | Supported  | [gemma3n_vl_4b_medpix.yaml](../../examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix.yaml), [gemma3n_vl_4b_medpix_peft.yaml](../../examples/vlm_finetune/gemma3n/gemma3n_vl_4b_medpix_peft.yaml) |
| Nemotron-Parse-v1.1                | cord-v2                      | Supported  | Supported  | [nemotron_parse_v1_1.yaml](../../examples/vlm_finetune/nemotron/nemotron_parse_v1_1.yaml) |
| Qwen2.5-VL-3B-Instruct             | rdr-items                    | Supported  | Supported  | [qwen2_5_vl_3b_rdr.yaml](../../examples/vlm_finetune/qwen2_5/qwen2_5_vl_3b_rdr.yaml) |
| Qwen3-VL-{4B,8B}-Instruct          | rdr-items                    | Supported  | Supported  | [qwen3_vl_4b_instruct_rdr.yaml](../../examples/vlm_finetune/qwen3/qwen3_vl_4b_instruct_rdr.yaml), [qwen3_vl_8b_instruct_rdr.yaml](../../examples/vlm_finetune/qwen3/qwen3_vl_8b_instruct_rdr.yaml) |
| Qwen3-VL-MoE                       | MedPix-VQA                   | Supported  | Supported  | [qwen3_vl_moe_30b_te_deepep.yaml](../../examples/vlm_finetune/qwen3/qwen3_vl_moe_30b_te_deepep.yaml), [qwen3_vl_moe_235b.yaml](../../examples/vlm_finetune/qwen3/qwen3_vl_moe_235b.yaml) |
| Qwen3-Omni-30BA3B                  | MedPix-VQA                   | Supported  | Supported  | [qwen3_omni_moe_30b_te_deepep.yaml](../../examples/vlm_finetune/qwen3/qwen3_omni_moe_30b_te_deepep.yaml) |
| InternVL3.5-4B                     | MedPix-VQA                   | Supported  | Supported  | [internvl_3_5_4b.yaml](../../examples/vlm_finetune/internvl/internvl_3_5_4b.yaml) |
| Ministral3-{3B,8B,14B}             | MedPix-VQA                   | Supported  | Supported  | [ministral3_3b_medpix.yaml](../../examples/vlm_finetune/mistral/ministral3_3b_medpix.yaml), [ministral3_8b_medpix.yaml](../../examples/vlm_finetune/mistral/ministral3_8b_medpix.yaml), [ministral3_14b_medpix.yaml](../../examples/vlm_finetune/mistral/ministral3_14b_medpix.yaml) |
| Phi-4-multimodal-instruct          | commonvoice_17_tr_fixed      | Supported  | Supported  | [phi4_mm_cv17.yaml](../../examples/vlm_finetune/phi4/phi4_mm_cv17.yaml) |

For detailed instructions on fine-tuning these models using both SFT and PEFT approaches, please refer to the [Gemma 3 and Gemma 3n Fine-Tuning Guide](../guides/omni/gemma3-3n.md). The guide covers dataset preparation, configuration, and running both full fine-tuning and LoRA-based parameter efficient fine-tuning.


## Dataset Examples

:::{tip}
In these guides, we use the `quintend/rdr-items` and `naver-clova-ix/cord-v2` datasets for demonstration purposes, but you can specify your own data as needed.
:::

### RDR Items Dataset
The rdr items dataset [`quintend/rdr-items`](https://huggingface.co/datasets/quintend/rdr-items) is a small dataset containing 48 images with descriptions. This dataset serves as an example of how to prepare image-text data for VLM fine-tuning. For complete instructions on dataset preprocessing and the collate functions used, see the [Gemma Fine-Tuning Guide](../guides/omni/gemma3-3n.md).

### CORD-v2 Dataset
The cord-v2 dataset [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2) contains receipts with descriptions in JSON format. This demonstrates handling structured data in VLMs. The [Gemma Fine-Tuning Guide](../guides/omni/gemma3-3n.md) provides detailed examples of custom preprocessing and collate functions for similar datasets.

## Train VLM Models
All supported models can be fine-tuned using either full SFT or PEFT approaches. The [Gemma Fine-Tuning Guide](../guides/omni/gemma3-3n.md) provides complete instructions for:
* Configuring YAML-based training.
* Running single-GPU and multi-GPU training.
* Setting up PEFT with LoRA.
* Model checkpointing and W&B integration.
