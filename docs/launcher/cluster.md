# Run on a Cluster (Slurm / Multi-node)

Use this guide for submitting distributed training jobs on Slurm clusters (single- or multi-node). For single-node workstation usage, see [Run on Your Local Workstation](./local-workstation.md). For setup details, refer to our [Installation Guide](../guides/installation.md).

## Run with Automodel CLI (Slurm)

The AutoModel CLI is the preferred method for most users. It provides a unified interface to submit Slurm batch jobs without deep knowledge of cluster specifics.

### Basic Usage

The CLI follows this format:
```bash
automodel <command> <domain> -c <config_file> [options]
```

Where:
- `<command>`: The operation to perform (`finetune`)
- `<domain>`: The model domain (`llm` or `vlm`)
- `<config_file>`: Path to your YAML configuration file

For workstation single-node instructions, see [Run on Your Local Workstation](./local-workstation.md).

### Submit a Batch Job with Slurm

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
  gpus_per_node: 8 # This adds "#SBATCH --gpus-per-node=8" to the script
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

The CLI will automatically submit the job to Slurm and handle the distributed setup.

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
- You don't need to modify the underlying code
- You prefer a higher-level abstraction

**Use uv when:**
- You're developing or debugging the codebase
- You want automatic dependency management
- You need maximum control over the execution
- You want to avoid manual environment setup
- You're experimenting with custom modifications

**Use Torchrun when:**
- You have a stable, pre-configured environment
- You prefer explicit control over Python execution
- You're working in environments where uv is not available
- You're integrating with existing PyTorch workflows

All approaches use the same configuration files. For single-node workflows, see [Run on Your Local Workstation](./local-workstation.md).
