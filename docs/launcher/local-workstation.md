# Run on Your Local Workstation

NeMo AutoModel supports running training/finetuning job from a single node with a single GPU to multiple multi-GPU nodes.
Use this guide for local, single-node workflows. For setup details, refer to our [Installation Guide](../guides/installation.md).
For executing distributed multi-node jobs, please refer to our [Run on a Cluster](./cluster.md) guide.

## Quick start: Choose your job launch option

- **CLI (recommended)**
  ```bash
  automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
  ```

- **Direct recipe script**
  - Single GPU
    ```bash
    python nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
    ```
  - Multi-GPU (single node)
    ```bash
    torchrun --nproc-per-node=2 nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
    ```

## Run with Automodel CLI (Single Node)

The AutoModel CLI is the preferred method for most users. It offers a unified interface to launch training jobs on your workstation without requiring deep knowledge of the underlying distributed setup.

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
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

### Train on Multiple GPUs (Single Node)

For interactive single-node jobs, the CLI automatically detects the number of available GPUs and
uses `torchrun` for multi-GPU training. You can manually specify the number of GPUs using the `--nproc-per-node` option:

```bash
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node=2
```

If you don't specify `--nproc-per-node`, it will use all available GPUs on your system.

Looking for Slurm or multi-node? See [Run on a Cluster](./cluster.md).

## Run with uv (Development Mode)

When you need more control over the environment or are actively developing with the codebase, you can use `uv` to run training scripts directly. This approach gives you direct access to the underlying Python scripts and is ideal for debugging or customization.

### Train on a Single GPU

```bash
uv run nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

### Train on Multiple GPUs with Torchrun (Single Node)

For multi-GPU single-node training, use `torchrun` directly:

```bash
uv run torchrun --nproc-per-node=2 nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
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
python nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

### Train on Multiple GPUs (Single Node)

```bash
torchrun --nproc-per-node=2 nemo_automodel/recipes/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
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
- You are running locally on a single machine
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

All approaches use the same configuration files and provide the same training capabilities on a single node. For Slurm-based multi-node training, see [Run on a Cluster](./cluster.md).
