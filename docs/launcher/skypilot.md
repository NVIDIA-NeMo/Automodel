# Run on Any Cloud with SkyPilot

In this guide, you will learn how to launch NeMo AutoModel training jobs on any major cloud provider (AWS, GCP, Azure, Lambda, Kubernetes) using [SkyPilot](https://skypilot.readthedocs.io). For on-premises cluster usage, see [Run on a Cluster (Slurm)](./slurm.md). For single-node workstation usage, see [Run on Your Local Workstation](./local-workstation.md).

SkyPilot is an open-source framework that abstracts cloud infrastructure so you can train on whichever cloud is cheapest or most available at launch time — including automatic spot-instance handling for significant cost savings.

## Before You Begin

Complete the following setup steps before launching your first AutoModel job on a cloud provider.

1. **Install SkyPilot** with the connector for your target cloud:

```bash
pip install "skypilot[gcp]"      # Google Cloud
pip install "skypilot[aws]"      # Amazon Web Services
pip install "skypilot[azure]"    # Microsoft Azure
pip install "skypilot[lambda]"   # Lambda Cloud
pip install "skypilot[kubernetes]"  # Any Kubernetes cluster
```

2. **Configure your cloud credentials** by following the SkyPilot credential setup guide for your cloud, then verify:

```bash
sky check
```

You should see at least one cloud listed as **OK**.

3. **Set required environment variables:**

```bash
export HF_TOKEN=hf_...          # Required for gated models (e.g. Llama)
export WANDB_API_KEY=...        # Optional: Weights & Biases logging
```

## Quickstart

Add a `skypilot:` section to any existing config YAML, then run the same `automodel` command you already know:

```bash
automodel finetune llm -c your_config_with_skypilot.yaml
```

The CLI detects the `skypilot:` key, strips it from the training config, uploads the code and config to a cloud VM, and launches training — all in one command.

## Configuration Reference

Below is an annotated example for fine-tuning Llama-3.2-1B on SQuAD on a GCP spot T4. A ready-to-run copy lives at [`examples/llm_finetune/llama3_2/llama3_2_1b_squad_skypilot.yaml`](../../examples/llm_finetune/llama3_2/llama3_2_1b_squad_skypilot.yaml).

```yaml
# ── SkyPilot launcher section ─────────────────────────────────────────────
# Removed before the training config reaches the remote VM.
skypilot:
  cloud: gcp                  # aws | gcp | azure | lambda | kubernetes
  accelerators: T4:1          # GPU type:count per node, e.g. A100:8
  use_spot: true              # ~80 % cost reduction vs on-demand
  disk_size: 100              # Remote VM disk size in GB
  num_nodes: 1                # Increase for multi-node distributed training
  region: us-central1         # Optional — SkyPilot picks cheapest if omitted
  job_name: llama3_2_finetune # Also used as the SkyPilot cluster name

  # Use env-var placeholders so secrets are never stored in YAML
  hf_token: ${HF_TOKEN}
  # wandb_key: ${WANDB_API_KEY}

  # Optional: extra shell commands run on the VM after `pip install -e .`
  # setup: |
  #   pip install some-extra-dependency

  # Optional: override the default output directory (default: ./skypilot_jobs)
  # job_dir: /path/to/skypilot/jobs

# ── Training config (forwarded to the VM unchanged) ───────────────────────
step_scheduler:
  global_batch_size: 64
  local_batch_size: 8
  num_epochs: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# ... rest of your training config ...
```

### All `skypilot:` Fields

| Field | Default | Description |
|---|---|---|
| `cloud` | *(required)* | Cloud provider: `aws`, `gcp`, `azure`, `lambda`, `kubernetes` |
| `accelerators` | `T4:1` | GPU type and count per node, e.g. `A100:8`, `V100:4` |
| `num_nodes` | `1` | Number of VMs for distributed training |
| `use_spot` | `true` | Use spot/preemptible instances |
| `disk_size` | `100` | Remote VM disk size in GB |
| `region` | *(auto)* | Cloud region; SkyPilot selects cheapest if omitted |
| `zone` | *(auto)* | Availability zone within the region |
| `instance_type` | *(auto)* | Specific instance type; auto-selected if omitted |
| `job_name` | `<domain>_<command>` | Job and SkyPilot cluster name |
| `setup` | *(auto)* | Extra setup commands run after `pip install -e .` |
| `hf_home` | `~/.cache/huggingface` | Hugging Face cache directory on the remote VM |
| `hf_token` | `$HF_TOKEN` env | Hugging Face token for gated model access |
| `wandb_key` | `$WANDB_API_KEY` env | Weights & Biases API key |
| `env_vars` | `{}` | Additional environment variables for the remote VM |
| `job_dir` | `./skypilot_jobs` | Local directory for job artifacts (config snapshot, logs) |
| `gpus_per_node` | *(parsed from `accelerators`)* | Override GPU count per node passed to `torchrun` |

## Cloud Examples

### AWS — On-Demand A10G

```yaml
skypilot:
  cloud: aws
  accelerators: A10G:1
  use_spot: false
  region: us-east-1
  job_name: llm_aws_finetune
  hf_token: ${HF_TOKEN}
```

### GCP — spot V100, 8 GPUs (single node)

```yaml
skypilot:
  cloud: gcp
  accelerators: V100:8
  use_spot: true
  region: us-west1
  job_name: llm_gcp_v100_8gpu
  hf_token: ${HF_TOKEN}
```

### Multi-node distributed training (2 × 8 × A100)

```yaml
skypilot:
  cloud: gcp
  accelerators: A100:8
  num_nodes: 2
  use_spot: false
  job_name: llm_multinode_a100
  hf_token: ${HF_TOKEN}
```

For multi-node jobs the launcher automatically adds the SkyPilot rendezvous environment variables (`$SKYPILOT_NODE_RANK`, `$SKYPILOT_NUM_NODES`, `$SKYPILOT_NODE_IPS`) to the `torchrun` command.

## Monitor and Manage Jobs

After submitting, use standard SkyPilot commands:

```bash
sky status                    # List running clusters and their status
sky logs <cluster_name>       # Stream training logs
sky ssh <cluster_name>        # SSH into the VM for debugging
sky cancel <cluster_name> <job_id>  # Cancel a running job
sky down <cluster_name>       # Terminate the cluster and stop billing
```

## How It Works

1. The `automodel` CLI detects the `skypilot:` key in the YAML and calls `launch_with_skypilot()`.
2. The training config (with `skypilot:` removed) is written to a local `skypilot_jobs/<timestamp>/job_config.yaml`.
3. A `sky.Task` is created with:
   - **workdir** — the current directory synced to `~/sky_workdir` on the remote VM.
   - **file_mounts** — the job config uploaded to `/tmp/automodel_job_config.yaml`.
   - **setup** — `pip install -e .` (plus any custom `setup:` commands).
   - **run** — a `torchrun` command pointing at the recipe script and config.
4. `sky.launch()` provisions the VM, runs setup, then executes training. The call returns immediately (`detach_run=True`); use `sky logs` to follow progress.

## Customize Configuration

Override any training parameter from the command line, same as with local runs:

```bash
automodel finetune llm -c config_with_skypilot.yaml \
  --model.pretrained_model_name_or_path meta-llama/Llama-3.2-3B
```

## When to Use SkyPilot vs. Slurm

| | SkyPilot | Slurm |
|---|---|---|
| **Infrastructure** | Any public cloud | On-premises HPC cluster |
| **Spot instances** | Yes (automatic) | Depends on cluster config |
| **Setup required** | Cloud credentials + `sky check` | Cluster access |
| **Good for** | Flexible cloud burst, cost optimization | Fixed on-prem GPU clusters |
