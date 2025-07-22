# Introduction to the NeMo Automodel Repository

This introductory guide presents the structure of the NeMo Automodel repository, provides a brief overview of its parts, introduces concepts such as components and recipes, and explains how everything fits together.

## What is NeMo Automodel?
NeMo Automodel is a PyTorch library for fine-tuning and pre-training models from the Hugging Face Hub. It provides:
- **Day-0 support** for most LLMs/VLMs on Hugging Face Hub.
- **Optimized implementations** for training efficiency, including fused kernels and memory-saving techniques.
- **Seamless integration** with Hugging Face datasets, tokenizers, and related tools.
- **FSDP2/nvFSDP**: Distributed training strategies for multi-GPU/node workloads.
- **Recipes**: End-to-end workflows combining data prep, training, and evaluation.


## Repository Structure
The Automodel source code is availabe under the [`nemo_automodel`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel) directory. It is organized into three parts:
- [`components/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components)  - Self-contained modules
- [`recipes/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/recipes) - End-to-end training workflows
- [`cli/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/_cli_) - Jon launching utility.


### Components
The `components/` directory contains isolated modules used in training loops.
Each component is designed to be dependency-light and reusable without cross-module imports.

#### Directory Structure
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

Key features:
    - Each component can be used independently in other projects.
    - Each component has its own dependencies, wihtout cross-module imports.
    - Unit tests live beside the component they cover.

### Recipes
Recipes define **end-to-end workflows** (data → training → eval) for a variety of tasks, such as,
training, finetuning and knowledge distillation, combining components into usable pipelines.

#### Available Recipes
The following directory listing shows all components with explaination of their contents:
```
$ tree -L 2 nemo_automodel/recipes/
├── llm
│   └── finetune.py   - Finetune recipe for LLMs (SFT, PEFT).
└── vlm
    └── finetune.py   - Finetune recipe for VLMs (SFT, PEFT).
```

#### Run a Recipe

Each recipe can be executed directly using torchrun, for example (from the root directory):
```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

The above command will finetune the Llama3.2-1B model on the SQuaD dataset with two GPUs.

Each recipe, imports the components it needs from the `nemo_automodel/components/` catalog.
The recipe/components structure enables users to:
    - Decouple most components and replace them with their own if needed.
    - Avoid rigid, structured trainer classes and instead opt for linear scripts, which surface the training logic to users for maximum control and flexibility.

<!-- For an in-depth explanation of the LLM recipe please also see the [LLM recipe deep-dive guide](docs/llm_recipe_deep_dive.md). -->

#### Recipe configuration
Example YAML configuration (complete config available [here](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm/llama_3_2_1b_squad.yaml)):
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

More recipe examples are availble under the [`examples/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples) directory.

### Automodel CLI application
The `automodel` CLI application simplifies job execution across different environments,
single-GPU interactive to batch multi-node. Currently, it supports SLURM clusters, with Kuberneters support coming soon.

#### Basic Usage
For example, to run the same torchrun llm finetune shown in the recipes section above:
```bash
automodel llm finetune -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

#### Launching a batch job on SLURM

The `automodel` CLI application further enables users to launch batch jobs. For example, to run
a job on a SLURM cluster, the YAML file needs to be extended with:
```yaml
slurm:
  job_name: llm-finetune  # if no job_name is provided will use {domain}_{command} from invocation
  nodes: 1
  ntasks_per_node: 8
  time: 00:05:00
  account: coreai_dlalgo_llm
  partition: batch
  container_image: nvcr.io/nvidia/nemo:dev # can also use path to sqsh, e.g.: /foo/bar/image.sqsh
  gpus_per_node: 8
  extra_mounts:
    - /a/b/c:/d/e
```
The above section defines the SLURM hyper-parameters necessary to launch a batch job on a SLURM
cluster using one node (`nodes` argument) and eight gpus (`ntasks_per_node`).

#### Launching a batch job on SLURM with modified code

The above `slurm` YAML configuration will use the automodel installation contained in the `container_image`.
However, if the command is run from within a git repo (that is accessible from SLURM workers), then
the SBATCH script will use the git repo for running the experiments, instead of installation in the container.

For example,
```bash
git clone git@github.com:NVIDIA-NeMo/Automodel.git automodel_test_repo
cd automodel_test_repo/
automodel llm finetune -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

Will launch the job using the source code contained in the `automodel_test_repo` directory, instead
of the one contained in the docker image.
<!-- The [Automodel CLI guide](docs/automodel_cli.md) provides an in-depth explanation of the automodel util. -->
