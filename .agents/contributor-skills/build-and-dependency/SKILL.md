---
name: build-and-dependency
description: Dev environment setup for NeMo AutoModel — container-based development, uv package management, installation options, environment variables, and common build pitfalls.
when_to_use: Setting up a dev environment, adding or removing dependencies, switching container images, configuring environment variables, 'uv sync fails', 'ModuleNotFoundError', 'TransformerEngine version mismatch', stale .venv issues.
---

# Build and Dependency

## Quick Start

Clone and install:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel
uv sync --locked --all-groups --extra all
```

Or use the NeMo-AutoModel container from NVIDIA NGC (pick a published tag from
[the NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel) —
e.g. `26.04`):

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.04
docker run --gpus all -it nvcr.io/nvidia/nemo-automodel:26.04
```

## Installation Options

### Option 1: NeMo-AutoModel Container (NGC)

The container ships with all dependencies pre-installed at `/opt/Automodel`
(WORKDIR) with the venv at `/opt/venv`. Run as-is:

```bash
docker run --gpus all --network=host -it --rm --shm-size=32g \
    nvcr.io/nvidia/nemo-automodel:26.04 /bin/bash
```

#### Mounting your local checkout into the container

To develop against your host checkout, bind-mount it over `/opt/Automodel` to
override the installed source:

```bash
docker run --gpus all --network=host -it --rm --shm-size=32g \
    -v <local-Automodel-path>:/opt/Automodel \
    nvcr.io/nvidia/nemo-automodel:26.04 /bin/bash
```

Inside the container, patch `pyproject.toml` / `uv.lock` for the PyTorch base
image, then re-sync:

```bash
cd /opt/Automodel
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --all-groups --extra all
```

> **Warning:** the `update_pyproject_pytorch.sh` step is required. Without it,
> `uv sync` will try to reinstall `torch`, which leads to CUDA version
> mismatches and TE import failures — uv cannot recognize the torch baked into
> the PyTorch base container.

### Option 2: uv (Recommended for Local Development)

`--all-groups` pulls the `build`, `docs`, and `test` dev groups (defined in
`pyproject.toml`); drop it for a runtime-only install.

```bash
uv sync --locked --all-groups                          # base + dev groups
uv sync --locked --all-groups --extra cuda             # CUDA support
uv sync --locked --all-groups --extra fa               # flash-attention
uv sync --locked --all-groups --extra moe              # mixture-of-experts
uv sync --locked --all-groups --extra vlm              # vision-language models
uv sync --locked --all-groups --extra diffusion        # diffusion models
uv sync --locked --all-groups --extra delta-databricks # Delta Lake / Databricks
uv sync --locked --all-groups --extra all              # everything
```

### Option 3: pip

Full install (matches `uv sync --extra all`):

```bash
pip install -e ".[all]"
```

Login-node / submitter-only install — lightweight package for SLURM, k8s, or
NeMo-Run job submission without local CUDA deps:

```bash
pip install nemo-automodel[cli]
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
| TransformerEngine version mismatch | The TE installed by `uv sync` takes precedence over the version baked into the container | Set the desired TE version in `pyproject.toml` / `uv.lock` and re-run `uv sync` — the venv's TE wins, not the container's |
