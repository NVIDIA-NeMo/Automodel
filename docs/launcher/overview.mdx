# Job Launchers

NeMo AutoModel provides several ways to launch training. The right choice depends on your hardware and environment.

## Which Launcher Should I Use?

| Launcher | Best for | GPUs | Guide |
|---|---|---|---|
| **Local Workstation** | Getting started, debugging, single-node training | 1-8 on one machine | [Local Workstation](./local-workstation.md) |
| **NeMo-Run** | Managed execution on Slurm, Kubernetes, Docker, local | 1+ | [NeMo-Run](./nemo-run.md) |
| **SkyPilot** | Cloud training or Kubernetes clusters | Any | [SkyPilot](./skypilot.md) |
| **Slurm** | Multi-node batch jobs on HPC clusters | 8+ across nodes | [Slurm](./slurm.md) |

### I Have 1–2 GPUs on My Workstation

Use the **interactive** launcher. No scheduler or cluster software is needed:

```bash
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

See the [Local Workstation](./local-workstation.md) guide.

### I Have Access to a Slurm Cluster

Add a `slurm:` section to your YAML config and submit with the same `automodel` command. The CLI generates the `torchrun` invocation and calls `sbatch` for you:

```bash
automodel config_with_slurm.yaml
```

See the [Slurm](./slurm.md) guide.

### I Want Managed Job Submission (Slurm, Kubernetes, Docker)

Add a `nemo_run:` section to your YAML config. NeMo-Run loads a pre-configured executor for your compute target and submits the job:

```bash
automodel config_with_nemo_run.yaml
```

See the [NeMo-Run](./nemo-run.md) guide.

### I Want to Train on the Cloud

Add a `skypilot:` section to your YAML config. SkyPilot provisions VMs on any major cloud and handles spot-instance preemption automatically:

```bash
automodel config_with_skypilot.yaml
```

See the [SkyPilot](./skypilot.md) guide.

### I Want to Train on Kubernetes with SkyPilot

Use the same `skypilot:` launcher, but set `cloud: kubernetes`. This is a good fit when your team already has a GPU-backed Kubernetes cluster and you want SkyPilot to handle job submission and multi-node orchestration:

```bash
automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad_skypilot_kubernetes.yaml
```

See the [SkyPilot + Kubernetes tutorial](./skypilot-kubernetes.md).

## All Launchers Use the Same Config

Every launcher shares the same YAML recipe format. The only difference is an optional launcher section (`slurm:`, `nemo_run:`, or `skypilot:`) that tells the CLI where to run. Without a launcher section, training runs interactively on the current machine.
