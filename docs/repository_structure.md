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

The above command will finetune the Llama3.2-1B model on the SQuaD dataset with two GPUs.
While the main source of each recipes live under `nemo_automodel/recipes/`,
to make them importable from third-party source, tiny utils also exist under `examples/` for convenience.

Each recipe, imports the components it needs from the `nemo_automodel/components/` catalog.
The recipe/components structure enables users to:
- Decouple most components and replace them with their own if needed.
- Avoid rigid, structured trainer classes and instead opt for linear scripts, which surface the training logic to users for maximum control and flexibility.

<!-- For an in-depth explanation of the LLM recipe please also see the [LLM recipe deep-dive guide](docs/llm_recipe_deep_dive.md). -->

#### Recipe configuration


### Automodel CLI application
The `automodel` CLI application enables users to run jobs from a single GPU to multiple nodes.
Currently, it supports SLURM clusters, with Kuberneters support coming soon.

For example, to run the torchrun llm finetune shown in the recipes section above:
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
