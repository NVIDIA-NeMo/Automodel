# Run on Your Local Workstation

There are multiple ways to launch jobs with NeMo Automodel, depending on your workflow and development needs. For installation instructions, see our [installation guide](../guides/installation.md).

## Automodel CLI Application

The automodel CLI application is the recommended approach for most users. It provides a unified interface for running training jobs locally or on distributed environments like Slurm clusters, without needing to worry about the underlying infrastructure details.

### Basic Usage

The CLI follows this format:
```bash
automodel <command> <domain> -c <config_file> [options]
```

Where:
- `<command>`: The operation to perform (`finetune`)
- `<domain>`: The model domain (`llm` or `vlm`)
- `<config_file>`: Path to your YAML configuration file

### Single GPU Training

For simple fine-tuning on a single GPU:

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml
```

### Multi-GPU Training

The CLI automatically detects available GPUs and uses `torchrun` for multi-GPU training. To specify the number of GPUs:

```bash
automodel finetune llm -c examples/llm/llama_3_2_1b_squad.yaml --nproc-per-node=2
```

If you don't specify `--nproc-per-node`, it will use all available GPUs on your system.

### Batch Job Submission (Slurm)

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
automodel llm finetune -c your_config_with_slurm.yaml
```

The CLI will automatically submit the job to Slurm and handle the distributed setup.

## Run with UV (Development Mode)

When you need more control over the environment or are actively developing with the codebase, you can use `uv` to run training scripts directly. This approach gives you direct access to the underlying Python scripts and is ideal for debugging or customization.

### Single GPU Training

```bash
uv run nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

### Multi-GPU Training with Torchrun

For multi-GPU training, use `torchrun` directly:

```bash
uv run torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

### Why Use UV?

UV provides several advantages for development and experimentation:

- **Automatic environment management**: UV automatically creates and manages virtual environments, ensuring consistent dependencies without manual setup
- **Lock file synchronization**: Keeps your local environment perfectly synchronized with the project's `uv.lock` file
- **No installation required**: Run scripts directly from the repository without installing packages system-wide
- **Development flexibility**: Direct access to Python scripts for debugging, profiling, and customization
- **Dependency isolation**: Each project gets its own isolated environment, preventing conflicts
- **Fast and reliable**: UV is written in Rust and provides faster dependency resolution than traditional tools

## Direct Torchrun Execution

If you have NeMo Automodel installed in your environment and prefer to run recipes directly without UV, you can use `torchrun` directly:

### Single GPU Training

```bash
python nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

### Multi-GPU Training

```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm/finetune.py -c examples/llm/llama_3_2_1b_squad.yaml
```

This approach requires that you have already installed NeMo Automodel and its dependencies in your Python environment (see the [installation guide](../guides/installation.md) for details).

## Configuration

All approaches use the same YAML configuration files. You can easily customize training by:

1. **Command-line arguments**: Override config values directly.
For example, if you want to finetune `Qwen/Qwen3-0.6B` instead of `meta-llama/Llama-3.2-1B`, you can use:
   ```bash
   automodel llm finetune -c config.yaml --model.pretrained_model_name_or_path Qwen/Qwen3-0.6B
   ```

2. **Editing the config file**: Modify the YAML directly for persistent changes

3. **Creating custom configs**: Copy and modify existing configurations from the `examples/` directory

## When to Use Which Approach

**Use the Automodel CLI when:**
- You want a simple, unified interface
- Running on production clusters (Slurm)
- You don't need to modify the underlying code
- You prefer a higher-level abstraction

**Use UV when:**
- You're developing or debugging the codebase
- You want automatic dependency management
- You need maximum control over the execution
- You want to avoid manual environment setup
- You're experimenting with custom modifications

**Use Torchrun when:**
- You have a stable, pre-configured environment
- You prefer explicit control over Python execution
- You're working in environments where UV is not available
- You're integrating with existing PyTorch workflows

All approaches use the same configuration files and provide the same training capabilities - choose based on your workflow preferences and requirements.