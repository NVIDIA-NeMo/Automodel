# Introduction to automodel repository

This introductory guide presents you with the NeMo Automodel repository's structure, provides a brief
overview of the parts, introduces the concepts such as components and recipes and explains how everything fits together.

## What is NeMo Automodel?
NeMo Automodel is a PyTorch-native machine learning framework/library for finetuning and pre-training models available on the Hugging Face Hub. The Hugging Face Hub integration provides day-0 support for nearly all LLMs and most VLMs, while seamlessly playing with the rest of the HF ecosystem.


## Repository structure, components and recipes
The Automodel source code is availabe under the `nemo_automodel` directory. It is organized in three main parts:
- `components/`
- `recipes/`
- `cli/`

### Components
The `components/` directory contains self-contained modules used in training and fine-tuning loops.
To keep dependencies minimal, maximize re-use, and localize bugs, each module is completely isolated: no cross-module imports are allowed. This guarantees that any component can be dropped into another project without pulling in unexpected code.

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
├── launcher/       - Job launcher (slurm, k8s, local); imports only stdlib + config.
├── loggers/        - Metric/event logging for Weights-&-Biases, etc.
├── loss/           - Loss function (e.g., cross-entropy, linear cross-entropy, etc).
├── optim/          - Optimizers and LR schedulers, including fused or second-order variants.
├── training/       - Base recipe and high-level training/finetuning utils.
└── utils/          - Small, dependency-free helpers (seed, profiler, timing, fs).
```

### Recipes
In NeMo Automodel, the term "recipe" describes a full training/finetuning recipe, which includes
data-preparation, model training/finetuning and evaluation.

### CLI
NeMo Automodel enables users to run jobs from a single GPU to multiple nodes. Currently, it supports
SLURM clusters, with Kuberneters support coming soon.


### Recipesaaaaa

### CLI
