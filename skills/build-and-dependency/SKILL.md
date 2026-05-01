---
name: build-and-dependency
description: Dev environment setup for NeMo AutoModel — container-based development, uv package management, installation options, environment variables, and common build pitfalls.
when_to_use: Setting up a dev environment, adding or removing dependencies, switching container images, configuring environment variables, 'uv sync fails', 'ModuleNotFoundError', 'TransformerEngine version mismatch', stale .venv issues.
---

# Build and Dependency

## Quick Start

Clone and install:

```bash
git clone <repo-url> && cd nemo-automodel
uv sync --locked --extra cuda --extra extra
```

Or use the NeMo-AutoModel container from NVIDIA NGC:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:latest
docker run --gpus all -it nvcr.io/nvidia/nemo-automodel:latest
```

## Installation Options

### Option 1: NeMo-AutoModel Container (NGC)

The container ships with all dependencies pre-installed. Mount your data and code:

```bash
docker run --gpus all -v $(pwd):/workspace -it nvcr.io/nvidia/nemo-automodel:latest
```

### Option 2: uv (Recommended for Local Development)

```bash
uv sync --locked                          # base install
uv sync --locked --extra cuda             # CUDA support
uv sync --locked --extra fa               # flash-attention
uv sync --locked --extra moe              # mixture-of-experts
uv sync --locked --extra vlm              # vision-language models
uv sync --locked --extra diffusion        # diffusion models
uv sync --locked --extra delta-databricks # Delta Lake / Databricks
uv sync --locked --extra all              # everything
```

### Option 3: pip

```bash
pip install -e ".[all]"
```

## Package Management

Always use `uv`. Do not introduce `pip install` commands in scripts or docs.

| Task | Command |
|---|---|
| Install from lockfile | `uv sync --locked` |
| Add a new dependency | `uv add <package>` |
| Add an optional dependency | `uv add --optional --extra <group> <package>` |
| Regenerate the lockfile | `uv lock` |

## Environment Variables

```bash
export HF_TOKEN="hf_..."           # Hugging Face token for gated models
export WANDB_API_KEY="..."         # Weights & Biases logging
export HF_HOME="/path/to/hf_cache" # Hugging Face cache directory
```

## CLI Usage

The entry point is `automodel` (defined at `nemo_automodel._cli.app:main`).

Pattern: `automodel <command> <domain> -c <config.yaml>`

```bash
# LLM
automodel finetune llm -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
automodel pretrain llm -c config.yaml
automodel kd llm -c config.yaml
automodel benchmark llm -c config.yaml

# VLM
automodel finetune vlm -c config.yaml

# Diffusion
automodel finetune diffusion -c config.yaml

# Retrieval
automodel finetune retrieval -c config.yaml
```

Override any config value from the CLI:

```bash
automodel finetune llm -c config.yaml --model.name_or_path meta-llama/Llama-3.2-1B
```

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Stale `.venv` after switching branches | Cached environment out of sync | Delete `.venv` and re-run `uv sync --locked` |
| Import errors for optional features (TE, flash-attn, MoE) | Missing extras | Install the matching `uv` extra (`--extra fa`, `--extra moe`, etc.) |
| TransformerEngine version mismatch | TE pinned to container version | Pin TE version to what the container ships, or rebuild from source |
