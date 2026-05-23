# Step-3.7-Flash

[Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) is Stepfun AI's 198B-A13B Mixture-of-Experts vision-language model. It extends the Step-3.5-Flash language architecture with native vision support for image and video understanding, with an emphasis on agentic developer workflows and stable tool calling.

:::{card}
| | |
|---|---|
| **Task** | Image-Text-to-Text / Video-Text-to-Text |
| **Architecture** | 198B total / 13B active MoE VLM |
| **Language Module** | Step-3.5-Flash-derived backbone, 45 layers, 288 experts, top-8 routing |
| **Vision Module** | 1.8B ViT, 47 layers, 728x728 image size |
| **Context Window** | 256k tokens |
| **Precision** | BF16 and FP8 planned for Day 0; NVFP4 best effort |
| **HF Org** | [stepfun-ai](https://huggingface.co/stepfun-ai) |
:::

## Positioning

Step-3.7-Flash is positioned as a multimodal foundation model for agents and agentic applications. The model targets high-throughput, low-latency inference so developer workflows can use image and video context without relying on text-only requirement descriptions.

## Architecture

- **Language backbone:** derived from Step-3.5-Flash with 45 layers, 288 experts, 8 activated experts per token, and a 256k context length.
- **Vision backbone:** 1.8B-parameter ViT with 47 layers and 728x728 image inputs.
- **Optimization target:** trained on Hopper GPUs, with BF16 and FP8 support planned on Day 0 and NVFP4 listed as best effort.

## Key Strengths

- **Native multimodal input.** Designed for image and video understanding on top of a large sparse language backbone.
- **Agentic stability.** Focused on tool-call stability for agent frameworks and bounded task execution.
- **Developer workflow fit.** Targets frontend generation from mockups, data-processing tasks, and screenshot-based debugging.
- **Fast serving path.** Intended for high throughput and fast inference in real-time developer loops.

## Available Models

- **Step-3.7-Flash**

## Example HF Models

| Model | HF ID |
|---|---|
| Step-3.7-Flash | [`stepfun-ai/Step-3.7-Flash`](https://huggingface.co/stepfun-ai/Step-3.7-Flash) |

## Example Recipes

This documentation-only branch does not add a recipe YAML.

## Agent Frameworks

Step-3.7-Flash continues support for agent integrations such as OpenClaw, HermesAgent, and KiloClaw.

## Hugging Face Model Cards

- [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash)
