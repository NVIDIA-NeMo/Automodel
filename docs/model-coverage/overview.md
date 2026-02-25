# Model Coverage

NeMo AutoModel integrates with Hugging Face `transformers`. Any LLM or VLM that can be instantiated through `transformers` can also be used via NeMo AutoModel, subject to runtime, third-party software dependencies, and feature compatibility.

## Supported Hugging Face Auto Classes

| Auto Class | Task | Status | Details |
|------------|------|--------|---------|
| `AutoModelForCausalLM` | Text Generation (LLM) | Supported | See [LLM model list](llm.md). |
| `AutoModelForImageTextToText` | Image-Text-to-Text (VLM) | Supported | See [VLM model list](vlm.md). |
| `AutoModelForSequenceClassification` | Sequence Classification | WIP | Early support; interfaces may change. |

## Release Log

The table below tracks when model support and key features were added across NeMo AutoModel releases. For the full list of tested architectures and example configs, see the [LLM](llm.md) and [VLM](vlm.md) pages.

| Release | Date | New Models | Key Features |
|---------|------|------------|--------------|
| **0.3.0** (upcoming) | — | Kimi-VL, Kimi-K25-VL, Gemma 3n, Nemotron-Parse, Qwen3-VL-MoE, Qwen3-Omni, InternVL 3.5, Ministral3, Phi-4-multimodal, Devstral-Small-2, Step-3.5-Flash, Qwen3-Next, Nemotron-3-Nano-30B | MoE LoRA, expanded VLM coverage |
| **0.2.0** | Dec 2025 | GPT-OSS 20B/120B, Qwen3, Qwen3-MoE, GLM-4/4-MoE, Qwen2.5-VL, Qwen3-VL | Single- and multi-turn tool calling, streaming dataset, QAT for SFT, sequence classification, async DCP checkpointing, MLflow, CP + sequence packing for MoE |
| **0.1.0** | Oct 2025 | DeepSeek V3/V3.2, 40+ LLM architectures, Gemma 3 VLM | Pretraining, knowledge distillation, FP8 (torchao), pipeline parallelism, HSDP, auto pipelining, ColumnMapped dataset |
| **0.1.0a0** | Sep 2025 | Initial LLM and VLM support (Llama, Mistral, Qwen2, Gemma, Phi, and more) | MegatronFSDP, packed sequences, Triton LoRA kernels |


## Day-0 Support

- NeMo AutoModel closely tracks the latest `transformers` version and updates its dependency regularly.
- New models released on the Hugging Face Hub may require the latest `transformers` version, necessitating a package upgrade.
- We are working on a CI pipeline that automatically bumps the supported `transformers` version when a new release is detected, enabling even faster day-0 support.

**Note:** To use newly released models, you may need to upgrade your NeMo AutoModel installation — just as you would upgrade `transformers` to access the latest models. AutoModel mirrors the familiar `transformers` `Auto*` APIs while adding optional performance accelerations and distributed training features.

## Custom Model Registry

NeMo AutoModel includes a custom model registry that allows teams to:

- Add custom implementations to extend support to models not yet covered upstream.
- Provide optimized or faster implementations for specific models while retaining the same AutoModel interface.

## Having Issues?

If a model from the Hub doesn't work as expected, see the [Troubleshooting](troubleshooting.md) guide for common issues and solutions.
