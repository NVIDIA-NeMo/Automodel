## Recipes and End-to-End Examples

NeMo Automodel is organized around two key concepts: recipes and components.

Recipes are executable scripts configured with YAML files. Each recipe defines its own training and validation loop, orchestrated through a `step_scheduler`. It specifies the model, dataset, loss function, optimizer, scheduler, checkpointing, and distributed training settings—allowing end-to-end training with a single command.

Components are modular, plug-and-play building blocks referenced using the `_target_` field. These include models, datasets, loss functions, and distribution managers. Recipes assemble these components, making it easy to swap them out to change precision, distribution strategy, dataset, or task—without modifying the training loop itself.

This page maps the ready-to-run recipes found in the `examples/` directory to their intended use cases, representative model families, and the most relevant how-to guides.

- Examples root: [examples/ (GitHub)](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples)
- Getting started: [Quickstart](quickstart.md)

## Large Language Models (LLM)
This section provides practical recipes and configurations for working with large language models across three core workflows: fine-tuning, pretraining, and knowledge distillation.

### Fine-Tuning

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
This section provides practical recipes and configurations for working with vision language models, covering fine-tuning and generation workflows for multimodal tasks.

### Fine-Tuning

Fine-tuning recipes for VLMs.

- Folder: [examples/vlm_finetune](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune)
- Representative family: Gemma 3 (various configurations)
- How-to guide: [Gemma 3n: Efficient multimodal fine-tuning](omni/gemma3-3n.md)

### Generation

Simple generation script and configurations for VLMs.

- Folder: [examples/vlm_generate](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_generate)

## Diffusion Models (Text-to-Image & Text-to-Video)

Text-to-image and text-to-video diffusion models can generate visual content from natural language descriptions. Fine-tuning lets you adapt these models to a specific style, domain, or dataset — for example, generating product videos in your brand's aesthetic. Pretraining gives you full control when no existing model fits your needs.
 
This section walks through the full workflow in NeMo AutoModel: preparing your dataset, training the model, and generating outputs.

### Fine-Tuning

Fine-tuning recipes for adapting pretrained diffusion models to your data.

- Folder: [examples/diffusion/finetune](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion/finetune)
- Representative models: FLUX.1-dev (T2I, 12B), Wan 2.1 T2V 1.3B, HunyuanVideo 1.5
- How-to guide: [Diffusion fine-tuning](diffusion/finetune.md)

### Pretraining

Pretraining recipes for training diffusion models from scratch on large-scale datasets.

- Folder: [examples/diffusion/pretrain](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion/pretrain)
- Representative models: Wan 2.1 T2V 1.3B, FLUX.1-dev
- How-to guide: [Diffusion fine-tuning (pretraining section)](diffusion/finetune.md#configure-your-training-recipe)

### Generation

Generation scripts and configs for running inference with pretrained or fine-tuned diffusion models.

- Folder: [examples/diffusion/generate](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/diffusion/generate)
- Representative models: Wan 2.1 1.3B, FLUX.1-dev, HunyuanVideo
- How-to guide: [Diffusion generation](diffusion/finetune.md#generation--inference)

### Dataset Preparation

Preprocessing pipeline to create `.meta` files containing VAE latents and text embeddings.

- How-to guide: [Diffusion dataset preparation](diffusion/dataset.md)

---

If you are new to the project, begin with the [Quickstart](quickstart.md) guide. Then, select a recipe category above and follow its linked how-to guide(s). The provided YAML configurations can serve as templates—customize them by adapting model names, datasets, and precision settings to match your specific needs.
