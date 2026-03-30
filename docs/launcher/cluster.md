# Run on a Cluster

In this guide, you will learn how to submit distributed training jobs on Slurm clusters (single- or multi-node), Kubernetes clusters, or via NeMo-Run. For single-node workstation usage, see [Run on Your Local Workstation](./local-workstation.md). For setup details, refer to our [Installation Guide](../guides/installation.md).

NeMo AutoModel uses recipes to run end-to-end workflows. If you're new to recipes, see the [Repository Structure](../repository-structure.md) guide.


## Quickstart: Choose Your Job Launch Option

All cluster launch methods use the same CLI command. The launcher is selected by which section is present in the YAML config:

| Launcher | YAML section | When to use |
|---|---|---|
| SLURM | `slurm:` | HPC clusters with Slurm scheduler |
| Kubernetes | `k8s:` | Kubeflow / k8s clusters with PyTorchJob operator |
| NeMo-Run | `nemo_run:` | NeMo ecosystem workflows |
| Interactive | *(none)* | Local workstation or interactive Slurm node |

```bash
automodel your_config.yaml
# or use the short alias:
am your_config.yaml
# both also work with uv run:
uv run automodel your_config.yaml
```

### Lightweight CLI-Only Install for Login Nodes

If you only need to submit jobs from a login node (no local training), install
the lightweight CLI package:

```bash
pip install nemo-automodel[cli]
```

This installs only `pyyaml` -- no PyTorch or CUDA dependencies. It supports
SLURM and Kubernetes submission out of the box. For NeMo-Run submission,
also install nemo-run (`pip install nemo-run`). If you accidentally try to run a local
job, you will get a clear error with install instructions.

For interactive testing on a Slurm node (without the `slurm:` YAML section):
  - Single node, single GPU
    ```bash
    python3 nemo_automodel/recipes/llm/train_ft.py -c your_config.yaml
    ```
  - Single node, multiple GPUs
    ```bash
    torchrun --nproc-per-node=8 nemo_automodel/recipes/llm/train_ft.py -c your_config.yaml
    ```

## Submit a Batch Job with Slurm

SLURM clusters vary widely: some use Pyxis containers, others use
Singularity/Apptainer, and many run bare-metal with environment modules.
Instead of trying to cover all variations in code, AutoModel provides a
reference sbatch script that you copy and adapt to your cluster.

### Getting Started

1. Copy the reference script:

```bash
cp slurm.sub my_cluster.sub
```

2. Edit `my_cluster.sub` — change `#SBATCH` directives (account, partition,
   nodes, time), container runtime, mounts, and secrets for your cluster.

3. Add a `slurm:` section to your YAML config:

```yaml
recipe:
  _target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

slurm:
  script: my_cluster.sub
```

4. Submit the job:

```bash
automodel your_config.yaml
```

The CLI generates the `torchrun` command from your recipe config and makes it
available to your script as `$AUTOMODEL_COMMAND`.  All cluster-specific
configuration (SBATCH directives, container runtime, mounts, NCCL tuning,
secrets) lives in your sbatch script where you can see and edit it directly.


### How It Works

When you run `automodel config.yaml` with a `slurm:` section, the CLI:

1. Writes `job_config.yaml` (the recipe config) to a timestamped job directory
2. Generates the `torchrun` command from your recipe config
3. Copies your script into the job directory for reproducibility
4. Exports `AUTOMODEL_*` environment variables
5. Runs `sbatch` on your script

Your script receives these environment variables:

| Variable | Description |
|---|---|
| `AUTOMODEL_COMMAND` | Full torchrun invocation (ready to `eval` or pass to `srun`) |
| `AUTOMODEL_CONFIG` | Absolute path to `job_config.yaml` |
| `AUTOMODEL_JOB_DIR` | Directory where job artifacts are stored |
| `AUTOMODEL_REPO_ROOT` | Path to the AutoModel source repo |

The generated `AUTOMODEL_COMMAND` uses SLURM environment variables for node
count and GPUs per node, so it stays in sync with your `#SBATCH` directives:

```
PYTHONPATH=/opt/Automodel:$PYTHONPATH torchrun \
    --nproc_per_node=${SLURM_GPUS_PER_NODE:-8} \
    --nnodes=${SLURM_NNODES:-1} \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    nemo_automodel/recipes/llm/train_ft.py -c /path/to/job_config.yaml
```


### YAML Options

The `slurm:` section supports these fields:

| Field | Required | Description |
|---|---|---|
| `script` | **yes** | Path to your sbatch script |
| `job_dir` | no | Directory for job artifacts (default: `./slurm_jobs`) |
| `repo_root` | no | Path to AutoModel source (auto-detected from cwd, falls back to `/opt/Automodel`) |
| `nsys_enabled` | no | Prepend `nsys profile` to the torchrun command (default: `false`) |

Everything else (nodes, GPUs, time, partition, container image, mounts,
secrets, NCCL tuning) belongs in your sbatch script.


### Examples

**Pyxis container (NVIDIA clusters):**

```bash
#!/bin/bash
#SBATCH -A my_account
#SBATCH -p batch
#SBATCH -t 01:00:00
#SBATCH -N 8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH -J automodel-finetune
#SBATCH --output=slurm_jobs/%x_%j.out
#SBATCH --error=slurm_jobs/%x_%j.err
#SBATCH --dependency=singleton

echo "Running on hosts: $(echo $(scontrol show hostname))"

CONT=/lustre/fsw/images/automodel.sqsh
CONT_NAME=automodel-training
CONT_MOUNT="\
/home/$USER/Automodel:/opt/Automodel,\
/home/$USER/.cache/huggingface:/root/.cache/huggingface"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=13742
export NUM_GPUS=${SLURM_GPUS_PER_NODE:-8}
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))
export WANDB_API_KEY=${WANDB_API_KEY}
export HF_TOKEN=${HF_TOKEN}

srun \
    --container-name="${CONT_NAME}" \
    --container-image="${CONT}" \
    --container-mounts="${CONT_MOUNT}" \
    --container-entrypoint \
    --no-container-mount-home \
    --export=ALL \
    bash -c "cd /opt/Automodel && ${AUTOMODEL_COMMAND}"
```

**Bare-metal (no container):**

```bash
#!/bin/bash
#SBATCH -A my_account
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=13742

module load cuda/12.8
source /opt/venvs/automodel/bin/activate

srun bash -c "$AUTOMODEL_COMMAND"
```

**Apptainer / Singularity:**

```bash
#!/bin/bash
#SBATCH -A my_account
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=13742

srun apptainer exec --nv /shared/images/automodel.sif \
    bash -c "$AUTOMODEL_COMMAND"
```

To use any of these scripts:

```yaml
slurm:
  script: my_cluster.sub
```

```bash
automodel my_config.yaml
```

If you prefer to write the torchrun command yourself (skipping the CLI's
command generation), just replace `${AUTOMODEL_COMMAND}` with a hardcoded
command in your script.


### Launch with Modified Code

If the command is executed from within a Git repository accessible to Slurm workers, `AUTOMODEL_COMMAND` will use the repository source over the AutoModel installation inside the container image.

```bash
git clone git@github.com:NVIDIA-NeMo/Automodel.git automodel_test_repo
cd automodel_test_repo/
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

## Submit a Job with Kubernetes

For Kubernetes clusters running the [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/), add a `k8s` section to your YAML configuration:

```yaml
recipe:
  _target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction

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

# Add Kubernetes configuration
k8s:
  num_nodes: 2
  gpus_per_node: 8
  image: nvcr.io/nvidia/nemo-automodel:latest
  namespace: default
  pvc_mounts:
    - claim: data-pvc
      mount_path: /data
  env_vars:
    HF_HOME: /data/.hf_home
  node_selector:
    nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
```

Then submit the job:
```bash
automodel your_config_with_k8s.yaml
```

The CLI generates a Kubeflow `PyTorchJob` manifest and submits it via `kubectl apply`. The config is stored in a `ConfigMap` and mounted into each pod at `/etc/automodel/config.yaml`.

### Monitor the Job

```bash
kubectl -n default get pytorchjobs
kubectl -n default logs -f -l training.kubeflow.org/job-name=automodel-<timestamp>
```

## Submit a Job with NeMo-Run

[NeMo-Run](https://github.com/NVIDIA/NeMo-Run) provides a higher-level abstraction for launching distributed jobs across multiple backends. Add a `nemo_run` section to your YAML configuration:

```yaml
recipe:
  _target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  grad_acc_steps: 4
  num_epochs: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# NeMo-Run with Slurm executor
nemo_run:
  executor: slurm          # "local", "slurm", or "k8s"
  num_nodes: 2
  num_gpus_per_node: 8
  container_image: nvcr.io/nvidia/nemo-automodel:latest
  account: your_account
  partition: gpu
  time: "01:00:00"
```

Then submit the job:
```bash
automodel your_config_with_nemo_run.yaml
```

:::{note}
NeMo-Run is an optional dependency. Install it with:
```bash
pip install nemo-run
```
:::

### NeMo-Run Executors

| Executor | Description |
|---|---|
| `local` | Run on the current machine (like interactive mode, but through NeMo-Run's API) |
| `slurm` | Submit to a Slurm cluster via NeMo-Run's SlurmExecutor |
| `k8s` | Submit to Kubernetes via NeMo-Run's K8sExecutor |

## Customize Configuration Settings

All approaches use the same YAML configuration files. You can easily customize training by following the steps in this section.

1. **Override config values**: Use command-line arguments to directly replace default settings.
For example, if you want to fine-tune `Qwen/Qwen3-0.6B` instead of `meta-llama/Llama-3.2-1B`, you can use:
   ```bash
   automodel config.yaml --model.pretrained_model_name_or_path Qwen/Qwen3-0.6B
   ```

2. **Edit the config file**: Modify the YAML directly for persistent changes.

3. **Create custom configs**: Copy and modify existing configurations from the `examples/` directory.

## When to Use Which Approach

**Use the AutoModel CLI when:**
- You want a simple, unified interface
- You are submitting jobs to production clusters (Slurm, Kubernetes)

**Use NeMo-Run when:**
- You need NeMo ecosystem integration
- You want a single API that spans local, Slurm, and Kubernetes

**Use uv when:**
- You're developing or debugging the codebase
- You want automatic dependency management

**Use Torchrun when:**
- You have a stable, pre-configured environment
- You prefer explicit control over Python execution

All approaches use the same configuration files. For single-node workflows, see our [Run on Your Local Workstation](./local-workstation.md) guide.
