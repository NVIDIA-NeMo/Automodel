# Run on Your Local Workstation

NeMo AutoModel supports various methods for launching jobs, allowing you to choose the approach that best fits your workflow and development needs. For setup details, refer to our [Installation Guide](../guides/installation.md).

## Run with Automodel CLI

The AutoModel CLI is the preferred method for most users. It offers a unified interface to launch training jobs locally or across distributed systems such as Slurm clusters, without requiring deep knowledge of the underlying infrastructure.

### Basic Usage

The CLI follows this format:
```bash
automodel <command> <domain> -c <config_file> [options]
```

Where:
- `<command>`: The operation to perform (`finetune`)
- `<domain>`: The model domain (`llm` or `vlm`)
- `<config_file>`: Path to your YAML configuration file

### Train on a Single GPU

For simple fine-tuning on a single GPU:

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml
```

### Train on Multiple GPUs

For interactive single-node jobs, the CLI automatically detects the number of available GPUs and
uses `torchrun` for multi-GPU training. You can specify manually the number of GPUs using the `--nproc-per-node` option, as follows:

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

If you don't specify `--nproc-per-node`, it will use all available GPUs on your system.

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
  gpus_per_node: 8
```

Then submit the job:
```bash
automodel finetune llm -c your_config_with_slurm.yaml
```

The CLI will automatically submit the job to Slurm and handle the distributed setup.

## Run with uv (Development Mode)

When you need more control over the environment or are actively developing with the codebase, you can use `uv` to run training scripts directly. This approach gives you direct access to the underlying Python scripts and is ideal for debugging or customization.

### Train on a Single GPU

```bash
uv run nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

### Train on Multiple GPUs with Torchrun

For multi-GPU training, use `torchrun` directly:

```bash
uv run torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

### Why use uv?

uv provides several advantages for development and experimentation:

- **Automatic environment management**: uv automatically creates and manages virtual environments, ensuring consistent dependencies without manual setup.
- **Lock file synchronization**: Keeps your local environment perfectly synchronized with the project's `uv.lock` file.
- **No installation required**: Run scripts directly from the repository without installing packages system-wide.
- **Development flexibility**: Direct access to Python scripts for debugging, profiling, and customization.
- **Dependency isolation**: Each project gets its own isolated environment, preventing conflicts.

## Run with Torchrun

If you have NeMo Automodel installed in your environment and prefer to run recipes directly without uv, you can use `torchrun` directly:

### Train on a Single GPU

```bash
python nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

### Train on Multiple GPUs

```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

This approach requires that you have already installed NeMo Automodel and its dependencies in your Python environment (see the [installation guide](../guides/installation.md) for details).

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
- You are running on production clusters (Slurm)
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

All approaches use the same configuration files and provide the same training capabilities - choose based on your workflow preferences and requirements.