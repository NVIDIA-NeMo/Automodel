# Run on a Cluster (Slurm / Multi-node)

In this guide you will learn how to submit distributed training jobs on Slurm clusters (single- or multi-node). For single-node workstation usage, see [Run on Your Local Workstation](./local-workstation.md). For setup details, refer to our [Installation Guide](../guides/installation.md).

NeMo Automodel uses recipes to run end-to-end workflows. If you're new to recipes, see the [Repository Structure](../repository-structure.md) guide.


:::{note}
Kubernetes support is coming soon.
:::

## Quick start: Choose your job launch option

Slurm jobs support two modes of execution: `batch` and `interactive`. In `batch` mode, the job is submitted to the cluster queue and
is executed without any other input from the user (e.g., no keyboard input), while the `interactive` mode, as the name implies, enables keyboard input.

- **CLI (recommended for Slurm)**
  This only requires to configure the `slurm` section in the YAML file, and the launcher will render the SBATCH script.
  ```bash
  automodel finetune llm -c your_config_with_slurm.yaml
  ```

- **Direct recipe script (typically for interactive testing)**
  You can also launch an interactive job on a Slurm node, and on the node run:
  - Single node, single GPU
    ```bash
    python3 examples/llm_finetune/finetune.py -c your_config.yaml
    ```
  - Single node, multiple GPUs
    ```bash
    torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py -c your_config.yaml
    ```
  - Note: For multi-node, prefer the CLI with `slurm` configuration.

## Submit a Batch Job with Slurm

For distributed training on Slurm clusters, add a `slurm` section to your YAML configuration:

```yaml
# Your existing model, dataset, training config...
step_scheduler:
  grad_acc_steps: 4
  num_epochs: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

# Add Slurm configuration
slurm:
  job_name: llm-finetune
  nodes: 1
  ntasks_per_node: 8
  time: 00:30:00
  account: your_account
  partition: gpu
  container_image: nvcr.io/nvidia/nemo:25.07
  gpus_per_node: 8 # Adds an SBATCH line: "#SBATCH --gpus-per-node=8"
  # Optional: Add extra mount points if needed
  extra_mounts:
    - /lustre:/lustre
  # Optional: Specify custom HF_HOME location (will auto-create if not specified)
  hf_home: /path/to/your/HF_HOME
  # Optional : Specify custom env vars
  # env_vars:
  #   ENV_VAR: value
  # Optional: Specify custom job directory (defaults to cwd/slurm_jobs)
  # job_dir: /path/to/slurm/jobs
```

Then submit the job:
```bash
automodel finetune llm -c your_config_with_slurm.yaml
```

The AutoModel CLI is the preferred method for most users. It provides a unified interface to submit Slurm batch jobs without deep knowledge of cluster specifics.
The CLI will automatically submit the job to Slurm and handle the distributed setup.
The above example launches one node with eight workers per node inside torchrun (`--nproc_per_node=8`).
The Slurm script itself uses `#SBATCH --ntasks-per-node 1`, and when `gpus_per_node` is set, it adds `#SBATCH --gpus-per-node=8` as well.

The CLI follows this format:
```bash
automodel <command> <domain> -c <config_file> [options]
```

Where:
- `<command>`: The operation to perform (`finetune`)
- `<domain>`: The model domain (`llm` or `vlm`)
- `<config_file>`: Path to your YAML configuration file

### Launch a Batch Job on Slurm with Modified Code

If the command is executed from within a Git repository accessible to Slurm workers, the generated SBATCH script will prioritize the repository source over the Automodel installation inside the container image.

For example:
```bash
git clone git@github.com:NVIDIA-NeMo/Automodel.git automodel_test_repo
cd automodel_test_repo/
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node=2
```

This will launch the job using the source code in the `automodel_test_repo` directory instead of the version bundled in the Docker image.

## Standalone Slurm Script (advanced)

If you prefer to submit with your own Slurm script, here is a standalone bash script adapted from the Automodel launcher template. See the upstream template for the authoritative reference: [Automodel Slurm template](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/launcher/slurm/template.py).

```bash
#!/bin/bash
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -N <nodes>
#SBATCH --ntasks-per-node 1<gpus_per_node_directive>
#SBATCH --time <HH:MM:SS>
#SBATCH --mail-type=FAIL
#SBATCH --exclusive
#SBATCH --output=<job_dir>/slurm_%x_%j.out
#SBATCH -J <job_name>

# Multi-node env
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=<master_port>
export NUM_GPUS=<num_gpus>
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

# Experiment env
export WANDB_API_KEY=<wandb_key>
export HF_HOME=<hf_home>
export HF_TOKEN=<hf_token>
# Add any custom env vars below, e.g.:
# export MY_ENV_VAR=value

# User command
read -r -d '' CMD <<'EOF'
cd <chdir>; whoami; date; pwd;
<command>
EOF
echo "$CMD"

srun \
    --mpi=pmix \
    --container-entrypoint \
    --no-container-mount-home \
    --container-image=<container_image> \
    --container-mounts=<container_mounts> \
    --export=ALL \
    bash -c "$CMD"
```

Replace bracketed placeholders (e.g., `<account>`, `<container_image>`, `<command>`) with your values. For multi-node training, ensure your `<command>` uses `torchrun` with `--nnodes=$SLURM_NNODES --nproc-per-node=$NUM_GPUS --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT` or rely on the Automodel CLI which configures this for you.

## Run with uv (Development Mode)

When developing on clusters, you can use `uv` to prepare and test scripts locally. For single-node `torchrun` examples, see [Run on Your Local Workstation](./local-workstation.md). Cluster execution should be done through the CLI with `slurm` configs above.

For Slurm-based execution, rely on the `slurm` section in your YAML and submit with the CLI.

### Why use uv?

uv provides several advantages for development and experimentation:

- **Automatic environment management**: uv automatically creates and manages virtual environments, ensuring consistent dependencies without manual setup.
- **Lock file synchronization**: Keeps your local environment perfectly synchronized with the project's `uv.lock` file.
- **No installation required**: Run scripts directly from the repository without installing packages system-wide.
- **Development flexibility**: Direct access to Python scripts for debugging, profiling, and customization.
- **Dependency isolation**: Each project gets its own isolated environment, preventing conflicts.

## Run with Torchrun

For cluster usage, prefer submitting via the CLI with `slurm` configuration. Direct `torchrun` is recommended for single-node development; see [Run on Your Local Workstation](./local-workstation.md).

### Standard PyTorch multi-node example

If you need a straightforward reference for manual multi-node launching with PyTorch (outside of Slurm helpers), use the pattern below. Run this on each node, updating `NODE_RANK` per node and adjusting `--nnodes`/`--nproc-per-node` as needed for your setup.

```bash
export MASTER_ADDR=node0.hostname   # master node's host/IP
export MASTER_PORT=29500
export NODE_RANK=0                  # node0 -> 0, node1 -> 1, ...

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py --batch_size 32
```

Notes:
- Set `NODE_RANK=0` on the master node (where `MASTER_ADDR` resolves), `NODE_RANK=1` on the second node, and so on.
- Ensure `--nproc_per_node` matches the number of GPUs per node.
- When launching under Slurm, prefer the CLI `slurm` configuration above or ensure equivalent rendezvous/env settings are provided via the scheduler.

## Customize Configuration Settings

All approaches use the same YAML configuration files. You can easily customize training by following the steps in this section.

1. **Override config values**: Use command-line arguments to directly replace default settings.
For example, if you want to fine-tune `Qwen/Qwen3-0.6B` instead of `meta-llama/Llama-3.2-1B`, you can use:
   ```bash
   automodel finetune llm -c config.yaml --model.pretrained_model_name_or_path Qwen/Qwen3-0.6B
   ```

2. **Edit the config file**: Modify the YAML directly for persistent changes.

3. **Create custom configs**: Copy and modify existing configurations from the `examples/` directory.

## When to Use Which Approach

**Use the Automodel CLI when:**
- You want a simple, unified interface
- You are submitting jobs to production clusters (Slurm)

**Use uv when:**
- You're developing or debugging the codebase
- You want automatic dependency management

**Use Torchrun when:**
- You have a stable, pre-configured environment
- You prefer explicit control over Python execution

All approaches use the same configuration files. For single-node workflows, see our [Run on Your Local Workstation](./local-workstation.md) guide.
