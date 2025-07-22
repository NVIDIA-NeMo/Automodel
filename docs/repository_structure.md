# Introduction to automodel repository

This introductory guide presents you with the NeMo Automodel repository's structure, provides a brief
overview of its parts, introduces the concepts such as components and recipes and explains how everything fits together.

## What is NeMo Automodel?
NeMo Automodel is a PyTorch-native machine learning framework/library for finetuning and pre-training models available on the Hugging Face Hub. The Hugging Face Hub integration provides day-0 support for nearly all LLMs and most VLMs, while seamlessly integrating with the rest of the HF ecosystem. NeMo Automodel provides optimized model implementations and training infrastructure.


## Repository structure, components and recipes
The Automodel source code is availabe under the `nemo_automodel` directory. It is organized in three main parts:
- `components/`
- `recipes/`
- `cli/`

We will proceed with presenting each part.

### Components
The `components/` directory contains self-contained modules used in training and fine-tuning loops.
To keep dependencies minimal, maximize re-use, and localize bugs, each module is completely isolated,
by not allowing cross-module imports. This guarantees that any component can be dropped into another project without pulling additional code from other modules.

Key points:
- One directory per component.
- Each component owns its own dependencies.
- Unit tests live beside the component they cover.

The following directory listing shows all components with explaination of their contents:
```
$ tree -L 1 nemo_automodel/components/

├── _peft/          - Implementations of PEFT methods (e.g., LoRA).
├── _transformers/  - Optimized model implementations for Hugging Face models.
├── checkpoint/     - Checkpoint save / load related logic.
├── config/         - Utils to load yamls and CLI parsing helpers.
├── datasets/       - LLM and VLM datasets and utils (collate functions, preprocessing).
├── distributed/    - Distributed processing primited (DDP, FSDP2, nvFSDP).
├── launcher/       - Job launcher for interactive and batch (slurm, k8s) processing.
├── loggers/        - Metric/event logging for Weights&Biases, etc.
├── loss/           - Loss function (e.g., cross-entropy, linear cross-entropy, etc).
├── optim/          - Optimizers and LR schedulers, including fused or second-order variants.
├── training/       - Training and finetuning utils.
└── utils/          - Small, dependency-free helpers (seed, profiler, timing, fs).
```

### Recipes
In NeMo Automodel, the term "recipe" describes a full training/finetuning/knowledge distillation recipe,
which includes data-preparation, model training/finetuning and evaluation.

The following directory listing shows all components with explaination of their contents:
```
$ tree -L 2 nemo_automodel/recipes/
├── llm
│   └── finetune.py   - Finetune recipe for LLMs (SFT, PEFT).
└── vlm
    └── finetune.py   - Finetune recipe for VLMs (SFT, PEFT).
```

Each recipe can be executed directly using torchrun, for example (from the root directory):
```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```
Will finetune the Llama3.2-1B model on the SQuaD dataset with two GPUs. While the main recipes
live under `nemo_automodel/recipes/`, to make them importable from third-party source, tiny utils
also exist under `examples/` for convenience.

Each recipe, imports the components it needs from the `nemo_automodel/components/` catalog.
The recipe/components structure enables users to:
- Decouple most components and replace them with their own if needed.
- Avoid rigid, structured trainer classes and instead opt for linear scripts, which surface the training logic to users for maximum control and flexibility.

<!-- For an in-depth explanation of the LLM recipe please also see the [LLM recipe deep-dive guide](docs/llm_recipe_deep_dive.md). -->

#### Recipe configuration
For recipe 

### CLI
NeMo Automodel enables users to run jobs from a single GPU to multiple nodes. Currently, it supports
SLURM clusters, with Kuberneters support coming soon.
