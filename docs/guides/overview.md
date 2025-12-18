## Recipes and E2E Examples

NeMo Automodel is organized around two key concepts: recipes and components.

Recipes are executable scripts configured with YAML files. Each recipe defines its own training and validation loop, orchestrated through a `step_scheduler`. It specifies the model, dataset, loss function, optimizer, scheduler, checkpointing, and distributed training settings—allowing end-to-end training with a single command.

Components are modular, plug-and-play building blocks referenced using the `_target_` field. These include models, datasets, loss functions, and distribution managers. Recipes assemble these components, making it easy to swap them out to change precision, distribution strategy, dataset, or task—without modifying the training loop itself.

This page maps the ready-to-run recipes found in the `examples/` directory to their intended use cases, representative model families, and the most relevant how-to guides.

- Examples root: [examples/ (GitHub)](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples)
- Getting started: [Installation](installation.md)

## Large Language Models (LLM)
This section provides practical recipes and configurations for working with large language models across three core workflows: fine-tuning, pretraining, and knowledge distillation.

### Fine-tuning

End-to-end fine-tuning recipes for many open models. Each subfolder contains YAML configurations showing task setups (e.g., SQuAD, HellaSwag), precision options (e.g., FP8), and parameter-efficient methods (e.g., LoRA/QLoRA).

- Folder: [examples/llm_finetune](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune)
- Representative families: Llama 3.1/3.2/3.3, Gemma 2/3, Falcon 3, Mistral/Mixtral, Nemotron, Granite, Starcoder, Qwen, Baichuan, GLM, OLMo, Phi, GPT-OSS, Moonlight
- How-to guide: [LLM finetuning](llm/finetune.md)

### Pretraining

Starter configurations and scripts for pretraining with datasets from different stacks (e.g., PyTorch, Megatron Core).

- Folder: [examples/llm_pretrain](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_pretrain)
- Example models: GPT-2 baseline, NanoGPT, DeepSeek-V3, Moonlight 16B TE (Slurm)
- How-to guides:
  - [LLM pretraining](llm/pretraining.md)
  - [Pretraining with NanoGPT](llm/nanogpt-pretraining.md)

### Knowledge Distillation (KD)

Recipes for distilling knowledge from a large teacher model into a smaller, more efficient student model.

- Folder: [examples/llm_kd](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_kd)
- Example model: Llama 3.2 1B
- How-to guide: [Knowledge distillation](llm/knowledge-distillation.md)

### Benchmark Configurations

Curated configurations for benchmarking different training stacks and settings (e.g., Torch vs. TransformerEngine + DeepEP, various model sizes, MoE variants).

- Folder: [examples/benchmark/configs](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/benchmark/configs)
- Representative configurations: DeepSeek-V3, GPT-OSS (20B/120B), Kimi K2, Moonlight 16B, Qwen3 MoE 30B


## Vision Language Models (VLM)
This section provides practical recipes and configurations for working with vision-language models, covering fine-tuning and generation workflows for multimodal tasks.

### Fine-tuning

Vision-language model fine-tuning recipes.

- Folder: [examples/vlm_finetune](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune)
- Representative family: Gemma 3 (various configurations)
- How-to guide: [Gemma 3n: Efficient multimodal fine-tuning](omni/gemma3-3n.md)

### Generation

Simple generation script and configurations for VLMs.

- Folder: [examples/vlm_generate](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_generate)

## Diffusion Generation

WAN 2.2 example for diffusion-based image generation.

- Folder: [examples/diffusion/wan2.2](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion/wan2.2)

---

If you are new to the project, begin with the [Installation](installation.md) guide. Then, select a recipe category above and follow its linked how-to guide(s). The provided YAML configurations can serve as templates—customize them by adapting model names, datasets, and precision settings to match your specific needs.
