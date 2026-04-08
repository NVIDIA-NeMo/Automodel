# Run with NeMo-Run

In this guide, you will learn how to launch NeMo AutoModel training jobs using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run). NeMo-Run supports multiple backends including Slurm, Kubernetes, Docker, and local execution. For cloud-based training, see [Run on Any Cloud with SkyPilot](./skypilot.md). For direct sbatch usage, see [Run on a Cluster (Slurm)](./slurm.md). For single-node workstation usage, see [Run on Your Local Workstation](./local-workstation.md).

NeMo-Run is an open-source tool from NVIDIA that manages job submission across different execution backends. You define your compute configuration once in a Python file and reuse it across all your training jobs.

## Before You Begin

1. **Install NeMo-Run** (it is not bundled with AutoModel):

```bash
pip install nemo-run
```

2. **Create an executor definitions file** at `$NEMORUN_HOME/executors.py`. `NEMORUN_HOME` defaults to `~/.nemo_run`; set the environment variable to use a different location. This file tells NeMo-Run how to reach your compute target. Every executor you reference in a YAML config must be defined here. See [Executor Setup](#executor-setup) for a complete example.

3. **Verify connectivity** to the target in your executor (e.g. SSH for Slurm, kubeconfig for Kubernetes).

4. **Set required environment variables** (if needed by your training config):

```bash
export HF_TOKEN=hf_...          # Required for gated models (e.g. Llama)
export WANDB_API_KEY=...        # Optional: Weights & Biases logging
```

## Executor Setup

The `executor:` field in your YAML config is a name that maps to an entry in `$NEMORUN_HOME/executors.py`. This file must define a module-level `EXECUTOR_MAP` dictionary. NeMo-Run supports several executor types -- here are examples of the most common ones:

### Slurm Executor

```python
import nemo_run as run

def my_slurm_cluster():
    executor = run.SlurmExecutor(
        account="my_account",
        partition="batch",
        tunnel=run.SSHTunnel(
            user="myuser",
            host="login-node.example.com",
            job_dir="/remote/path/nemo_run/jobs",
        ),
        nodes=1,
        ntasks_per_node=8,
        gpus_per_node=8,
        mem="0",
        exclusive=True,
        packager=run.Packager(),
    )
    executor.container_image = "nvcr.io/nvidia/nemo-automodel:26.02"
    executor.container_mounts = ["/data:/data", "/checkpoints:/checkpoints"]
    executor.env_vars = {"HF_HOME": "/data/hf_cache"}
    executor.time = "04:00:00"
    return executor

EXECUTOR_MAP = {
    "my_slurm": my_slurm_cluster(),
}
```

### Kubernetes Executor

```python
import nemo_run as run

def my_k8s_cluster():
    return run.KubeflowExecutor(
        namespace="training",
        image="nvcr.io/nvidia/nemo-automodel:26.02",
        num_nodes=1,
        nprocs_per_node=8,
        gpus_per_node=8,
    )

EXECUTOR_MAP = {
    "my_k8s": my_k8s_cluster(),
}
```

### Multiple Executors

You can define as many executors as you need for different backends, clusters, or resource configurations:

```python
EXECUTOR_MAP = {
    "slurm_dev": my_slurm_dev(),
    "slurm_prod": my_slurm_prod(),
    "k8s": my_k8s_cluster(),
}
```

- Keys in `EXECUTOR_MAP` are names you reference in YAML (`executor: slurm_dev`).
- Values can be executor instances or zero-argument callables that return one.
- Override fields in the YAML (`nodes`, `devices`, `container_image`, etc.) are applied on top of the executor defaults.

## Quickstart

Any existing AutoModel YAML config can be run via NeMo-Run by adding a `nemo_run:` section at the top. For example, given an existing config that you run locally:

```bash
automodel examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence.yaml
```

Add a `nemo_run:` block to submit it to a remote executor instead:

```yaml
# -- Add this section to any existing config ----------------------------------
nemo_run:
  executor: my_slurm             # Name from EXECUTOR_MAP in $NEMORUN_HOME/executors.py
  container_image: /images/custom.sqsh  # Override executor's default image
  nodes: 1                       # Override number of nodes
  ntasks_per_node: 8             # GPUs per node
  time: "04:00:00"               # Override time limit
  job_name: qwen3_moe_finetune   # Experiment and job name

# -- Everything below is your existing training config (unchanged) ------------
recipe: TrainFinetuneRecipeForNextTokenPrediction

step_scheduler:
  global_batch_size: 32
  # ... rest of your config ...
```

Then run the same command:

```bash
automodel your_config.yaml
```

The CLI detects the `nemo_run:` key, strips it from the training config, loads the named executor from `$NEMORUN_HOME/executors.py`, and submits the job -- all in one command.

## Configuration Reference

### All `nemo_run:` Fields

| Field | Default | Description |
|---|---|---|
| `executor` | `"local"` | Name from `EXECUTOR_MAP` in `$NEMORUN_HOME/executors.py`, or `"local"` for local execution |
| `job_name` | `<recipe_class_name>` | Experiment and job name |
| `detach` | `true` | Return immediately after submission |
| `tail_logs` | `false` | Stream logs after submission |
| `executors_file` | `$NEMORUN_HOME/executors.py` | Path to the executor definitions file |
| `job_dir` | `./nemo_run_jobs` | Local directory for job artifacts (config snapshot) |
| *(any other key)* | *(from executor)* | Applied directly to the executor via `setattr`. Use the executor's native attribute names (e.g. `nodes`, `ntasks_per_node`, `partition`, `container_image`, `time`, `env_vars`). Dicts are merged, lists are extended. |

## Examples

### Single-Node Fine-Tuning (1 x 8 GPUs)

```yaml
nemo_run:
  executor: my_slurm
  nodes: 1
  ntasks_per_node: 8
  job_name: single_node_finetune
```

### Multi-Node Distributed Training (2 x 8 GPUs)

```yaml
nemo_run:
  executor: my_slurm
  nodes: 2
  ntasks_per_node: 8
  time: "08:00:00"
  job_name: multinode_pretrain
```

For multi-node jobs the launcher automatically adds `--nnodes`, `--node-rank`, `--rdzv-backend`, `--master-addr`, and `--master-port` to the `torchrun` command.

### Custom Container Image and Mounts

```yaml
nemo_run:
  executor: my_slurm
  container_image: /images/automodel_nightly.sqsh
  container_mounts:
    - /scratch/datasets:/datasets
    - /scratch/checkpoints:/checkpoints
  env_vars:
    HF_HOME: /datasets/hf_cache
    NCCL_DEBUG: INFO
```

### Local Execution (No Cluster)

Use `executor: local` to run on the current machine. No `$NEMORUN_HOME/executors.py` entry is needed:

```yaml
nemo_run:
  executor: local
  ntasks_per_node: 2
  job_name: local_test
```

## Monitor and Manage Jobs

NeMo-Run stores experiment metadata under `$NEMORUN_HOME/experiments/`. Set `tail_logs: true` in the YAML to stream job output after submission.

For Slurm-based executors, standard Slurm commands also work:

```bash
squeue -u $USER                 # List your queued and running jobs
scancel <job_id>                # Cancel a running or pending job
sacct -j <job_id>               # View job accounting information
```

For Kubernetes-based executors, use `kubectl` to monitor pods and jobs.

## How It Works

1. The `automodel` CLI detects the `nemo_run:` key and imports `NemoRunLauncher`.
2. The `nemo_run:` section is popped from the config. The remaining training config is written to `nemo_run_jobs/<timestamp>/job_config.yaml` for record-keeping.
3. The launcher loads a pre-configured executor from `$NEMORUN_HOME/executors.py` by name (or creates a `LocalExecutor` for `executor: local`). Override fields are applied on top of the executor defaults.
4. The training config YAML is embedded in a self-contained inline bash script via a heredoc, so no separate file transfer is needed.
5. A `torchrun` command is built with `--nproc-per-node` and (for multi-node) distributed rendezvous arguments.
6. The script is submitted via `nemo_run.Experiment`. By default the call returns immediately (`detach=True`).

## Customize Configuration

Override any training parameter from the command line, same as with local runs:

```bash
automodel config_with_nemo_run.yaml \
  --model.pretrained_model_name_or_path meta-llama/Llama-3.2-3B
```

## When to Use NeMo-Run vs. SkyPilot vs. Slurm

| | NeMo-Run | SkyPilot | Slurm (sbatch) |
|---|---|---|---|
| **Infrastructure** | Slurm, Kubernetes, Docker, local | Public cloud (AWS, GCP, Azure) | On-prem HPC |
| **Container support** | Yes (Pyxis/Enroot, Docker, K8s pods) | N/A (cloud VMs) | Manual (in sbatch script) |
| **Setup required** | `nemo-run` + `$NEMORUN_HOME/executors.py` | Cloud credentials + `sky check` | Cluster access + sbatch script |
| **Job submission** | `automodel config.yaml` | `automodel config.yaml` | `sbatch slurm.sub` |
| **Good for** | Managed multi-backend execution, reusable executor configs | Cloud burst, cost optimization, spot instances | Direct Slurm scripts, full control over sbatch |
