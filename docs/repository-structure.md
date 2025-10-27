# Introduction to the NeMo Automodel Repository

This introductory guide presents the structure of the NeMo Automodel repository, provides a brief overview of its parts, introduces concepts such as components and recipes, and explains how everything fits together.

## What is NeMo Automodel?
NeMo Automodel is a PyTorch library for fine-tuning and pre-training large scale models. In particular, it provides:
- **Optimized implementations** for training efficiency, including fused kernels and memory-saving techniques.
- [**Day-0 support**](model-coverage/overview.md) for LLMs and VLMs available on the Hugging Face Hub.
- **Seamless integration** with Hugging Face datasets, tokenizers, and related tools.
- **Distributed training strategies** using FSDP2 and MegatronFSDP across multi-GPU and multi-node environments.
- **End-to-end workflows** with recipes for data preparation, training, and evaluation.


## Repository Structure
The Automodel source code is available under the [`nemo_automodel`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel) directory. It is organized into three directories:
- [`components/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components)  - Self-contained modules
- [`recipes/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/recipes) - End-to-end training workflows
- [`cli/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/_cli) - Job launcher.

### Components Directory
The `components/` directory contains isolated modules used in training loops.
Each component is designed to be dependency-light and reusable without cross-module imports.

#### Directory Structure
The following directory listing shows all components along with explanations of their contents.
```
$ tree -L 1 nemo_automodel/components/

├── _peft/          - Implementations of PEFT methods, such as LoRA.
├── attention/      - Efficient attention modules and related utilities (e.g., flash attention, rotary embeddings).
├── checkpoint/     - Checkpoint save and load-related logic.
├── config/         - Utils to load YAML files and CLI-parsing helpers.
├── datasets/       - LLM and VLM datasets and utils (collate functions, preprocessing).
├── distributed/    - Distributed processing primitives (DDP, FSDP2, MegatronFSDP).
├── launcher/       - Job launcher for interactive and batch (Slurm, K8s) processing.
├── loggers/        - Metric/event logging for Weights & Biases and other tools.
├── loss/           - Loss functions (such as cross-entropy and linear cross-entropy, etc.).
├── models/         - Optimized model implementations for LLMs and VLMs.
├── moe/            - Mixture of Experts modules and routing utilities for scalable model architectures.
├── optim/          - Optimizers and LR schedulers, including fused or second-order variants.
├── quantization/   - Quantization layers and helpers for 4-bit/8-bit or other reduced-precision training and inference.
├── training/       - Training and fine-tuning utils.
└── utils/          - Small, dependency-free helpers (seed, profiler, timing, fs).
```

#### Key Features
- Each component can be used independently in other projects.
- Each component has its own dependencies, without cross-module imports.
- Unit tests are colocated with the component they cover.

### Recipes Directory
Recipes define **end-to-end workflows** (data and model loading → training with custom loop → saving the output checkpoint)
for a variety of tasks, such as, training, fine-tuning, knowledge distillation, and combining components into usable pipelines.

#### Available Recipes
The following directory listing shows all components along with explanations of their contents.
```
$ tree -L 2 nemo_automodel/recipes/
├── llm
│   ├── benchmark.py  - Benchmark recipe for LLMs
│   ├── kd.py         - Knowledge Distillation for LLMs
│   └── train_ft.py   - Train recipe for LLMs (Pretrain & Finetune SFT, PEFT).
└── vlm
    └── finetune.py   - Finetune recipe for VLMs (SFT, PEFT).
```

#### Run a Recipe

Each recipe script can be executed directly using `torchrun`, for example, from the root directory:
```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

The above command will fine-tune the Llama3.2-1B model on the SQuaD dataset with two GPUs using the [`llama3_2_1b_squad.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/824408f007c42e11471a1f9e1c975b570514d2a8/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml) config
If you want to execute on a single GPU replace `torchrun --nproc-per-node` with `python3`:
```bash
python3 nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

Each recipe, imports the components it needs from the `nemo_automodel/components/` catalog.
The recipe/components structure enables you to:
- Decouple individual components and replace them with custom implementations when needed.
- Avoid rigid, class-based trainer structures by using linear scripts that expose training logic for maximum flexibility and control.

<!-- For an in-depth explanation of the LLM recipe please also see the [LLM recipe deep-dive guide](docs/llm_recipe_deep_dive.md). -->

#### Configure a Recipe
An example YAML configuration is shown below. The complete config is available [here](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml):
```yaml
step_scheduler:
  grad_acc_steps: 4
  ckpt_every_steps: 1000
  val_every_steps: 10  # will run every x number of gradient steps
  num_epochs: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train
```

More recipe examples are available under the [`examples/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples) directory.

### CLI Directory
The `automodel` CLI application simplifies job execution across different environments, from
single-GPU interactive sessions to batch multi-node runs. Currently, it supports Slurm clusters, with Kubernetes support coming soon.


## Next steps

Learn how to train models with NeMo AutoModel on:
- **Your local workstation**: See [`docs/launcher/local-workstation.md`](launcher/local-workstation.md).
- **A cluster**: See [`docs/launcher/cluster.md`](launcher/cluster.md).
