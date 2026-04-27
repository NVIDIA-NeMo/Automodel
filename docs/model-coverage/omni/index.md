# Omni Models

Omni models go beyond image-text understanding to support additional modalities such as audio, video, or a combination of all — text, image, audio, and video in a single unified model.

## Run Omni Models with NeMo AutoModel

To run omni models with NeMo AutoModel, use NeMo container version [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) or later. If the model you want to fine-tune requires a newer version of Transformers, you may need to upgrade:

```bash
pip3 install --upgrade git+git@github.com:NVIDIA-NeMo/AutoModel.git
```

For other installation options, see our [Installation Guide](../../guides/installation.md).

## Supported Models

| Owner | Model | Modalities | Architecture |
|---|---|---|---|
| Qwen / Alibaba Cloud | [Qwen3-Omni](qwen/qwen3-omni.md) | Text · Image · Audio · Video | `Qwen3OmniForConditionalGeneration` |
| Microsoft | [Phi-4-multimodal](microsoft/phi4-multimodal.md) | Text · Image · Audio | `Phi4MultimodalForCausalLM` |
| NVIDIA | [Nemotron-3-Nano-Omni](nvidia/nemotron-omni.md) | Text · Image · Audio | `NemotronH_Nano_Omni_Reasoning_V3` |

## Fine-Tuning

All supported omni models can be fine-tuned using full SFT or PEFT (LoRA) approaches. See the [VLM Fine-Tuning Guide](../../guides/omni/gemma3-3n.md) for general setup instructions.

```{toctree}
:hidden:

qwen/qwen3-omni
microsoft/phi4-multimodal
nvidia/nemotron-omni
```
