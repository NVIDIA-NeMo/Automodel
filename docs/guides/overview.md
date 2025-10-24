## Recipes and E2E Examples overview

NeMo AutoModel is organized using recipes and components.

A recipe is a runnable scipt, configured with YAML files, that owns its train/val loop. It wires the loop via `step_scheduler` and specifies model, data, loss, optimizer/scheduler, checkpointing, and distributed settings—so a single command trains end‑to‑end.

Components are plug‑and‑play modules referenced via `_target_` (e.g., models, datasets, losses, distribution managers). Recipes compose them; swap components to change precision, distribution, datasets, or tasks without rewriting the loop.

This page maps the ready-to-run recipes under the `examples/` directory to their purpose, representative model families, and the most relevant how-to guides.

- Examples root: [examples/ (GitHub)](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples)
- Getting started: [Installation](installation.md)

## Large Language Models (LLM)
### Finetuning

End-to-end finetuning recipes for many open models. Each subfolder contains YAMLs showing task setups (e.g., SQuAD, HellaSwag), precision options (e.g., FP8), and parameter-efficient methods (e.g., LoRA/QLoRA).

- Folder: [examples/llm_finetune](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune)
- Representative families: Llama 3.1/3.2/3.3, Gemma 2/3, Falcon 3, Mistral/Mixtral, Nemotron, Granite, Starcoder, Qwen, Baichuan, GLM, OLMo, Phi, GPT-OSS, Moonlight
- How-to guide: [LLM finetuning](llm/finetune.md)

### Pretraining

Starter configurations and scripts for pretraining with different stacks (e.g., PyTorch, Megatron-Core) and scales.

- Folder: [examples/llm_pretrain](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_pretrain)
- Examples include: GPT-2 baseline, NanoGPT, DeepSeek-V3, Moonlight 16B TE (Slurm)
- How-to guides:
  - [LLM pretraining](llm/pretraining.md)
  - [Megatron-Core pretraining](llm/mcore-pretraining.md)

### Knowledge Distillation (KD)

Recipes for distilling a teacher into a smaller student model.

- Folder: [examples/llm_kd](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_kd)
- Example: Llama 3.2 1B KD
- How-to guide: [Knowledge distillation](llm/knowledge-distillation.md)

### Benchmark configs

Curated configs for benchmarking training stacks and settings (e.g., Torch vs. TransformerEngine + DeepEP, model sizes, and MoE variants).

- Folder: [examples/benchmark/configs](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/benchmark/configs)
- Representative configs: DeepSeek-V3, GPT-OSS (20B/120B), Kimi K2, Moonlight 16B, Qwen3 MoE 30B


## Vision Language Models (VLM)
### Finetuning

Vision-language model finetuning recipes. Currently includes Gemma 3-based VLMs.

- Folder: [examples/vlm_finetune](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune)
- Representative family: Gemma 3 (various configs)
- How-to guide: [Gemma 3n: Efficient multimodal finetuning](omni/gemma3-3n.md)

### Generation

Simple generation script and configs for VLMs.

- Folder: [examples/vlm_generate](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_generate)

## Diffusion generation

WAN 2.2 example for diffusion-based image generation.

- Folder: [examples/diffusion/wan2.2](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion/wan2.2)

---

If you are new to the project, start with the [Installation](installation.md) guide, then pick a recipe category above and follow its linked guide(s). Many YAMLs can be used as templates—adapt model names, datasets, and precisions to your needs.
