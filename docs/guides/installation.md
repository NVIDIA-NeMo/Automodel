# Install NeMo AutoModel

This guide explains how to install NeMo AutoModel for LLM, VLM, and OMNI models on various platforms and environments. Depending on your use case, there are several ways to install it:

| Method                  | Dev Mode | Use Case                                                          | Recommended For             |
| ----------------------- | ---------|----------------------------------------------------------------- | ---------------------------- |
| 📦 **PyPI**             | - | Install stable release with minimal setup                         | Most users, production usage |
| 🐳 **Docker**           | - | Use in isolated GPU environments, e.g., with NeMo container       | Multi-node deployments     |
| 🐍 **Git Repo**         | ✅ | Use the latest code without cloning or installing extras manually | Power users, testers         |
| 🧪 **Editable Install** | ✅ | Contribute to the codebase or make local modifications            | Contributors, researchers    |
| 🐳 **Docker + Mount**   | ✅ | Use in isolated GPU environments, e.g., with NeMo container       | Multi-node deployments     |

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **Memory**: Minimum 16GB RAM, 32GB+ recommended
- **Storage**: At least 50GB free space for models and datasets

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **CPU**: Multi-core processor (8+ cores recommended)
- **Network**: Stable internet connection for downloading models

---
## Installation Options for Non-Developers
This section explains the easiest installation options for non-developers, including using pip3 via PyPI or leveraging a preconfigured NVIDIA NeMo Docker container. Both methods offer quick access to the latest stable release of NeMo Automodel with all required dependencies.
### Install via PyPI (Recommended)

For most users, the easiest way to get started is using `pip3`.

```bash
pip3 install nemo-automodel
```
:::{tip}
This installs the latest stable release of NeMo Automodel from PyPI.

To verify the install, run `python -c "import nemo_automodel; print(nemo_automodel.__version__)"`. See [nemo-automodel on PyPI](https://pypi.org/project/nemo-automodel/).
:::

### Install via NeMo Docker Container
You can use NeMo Automodel with the NeMo Docker container. Pull the container by running:
```bash
docker pull nvcr.io/nvidia/nemo-automodel:25.11.00
```
:::{note}
The above `docker` command uses the [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) container. Use the [most recent container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel) version to ensure you get the latest version of Automodel and its dependencies like torch, transformers, etc.
:::

Then you can enter the container using:
```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  nvcr.io/nvidia/nemo-automodel:25.11.00
```

---
## Installation Options for Developers
This section provides installation options for developers, including pulling the latest source from GitHub, using editable mode, or mounting the repo inside a NeMo Docker container.
### Install via GitHub (Source)

If you want the **latest features** from the `main` branch or want to contribute:

#### Option A - Use `pip` with git repo:
```bash
pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git
```
:::{note}
This installs the repo as a standard Python package (not editable).
:::

#### Option B - Use `uv` with git repo:
```bash
uv pip install git+https://github.com/NVIDIA-NeMo/Automodel.git
```
:::{note}
`uv` handles virtual environment transparently and enables more reproducible installs.
:::

### Install in Developer Mode (Editable Install)
To contribute or modify the code:
```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
pip3 install -e .
```

:::{note}
This installs Automodel in editable mode, so changes to the code are immediately reflected in Python.
:::

### Mount the Repo into a NeMo Docker Container
To run NeMo AutoModel inside a NeMo container while **mounting your local repo**, follow these steps:

```bash
# Step 1: Clone the AutoModel repository
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel

# Step 2: Pull a compatible container image (replace the tag as needed)
docker pull nvcr.io/nvidia/nemo-automodel:25.11.00

# Step 3: Run the container, mount the repo, and run a quick sanity check
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v "$PWD":/workspace/Automodel \
  nvcr.io/nvidia/nemo-automodel:25.11.00 \
  /bin/bash -lc "\
    cd /workspace/Automodel && \
    pip install -e . && \
    python3 examples/llm_finetune/finetune.py -c examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml"
```
:::{note}
The above `docker` command mounts your local `Automodel` directory into the container at `/workspace/Automodel`.
:::

## Optional dependencies (extras)
Some functionality (notably VLM recipes and CUDA-accelerated libraries) is gated behind optional extras.

Common extras:
- `vlm`: required for `examples/vlm_*`
- `cuda`: CUDA-adjacent dependencies (e.g., Transformer Engine, bitsandbytes, mamba)
- `fa`: flash-attn
- `moe`: MoE dependencies (e.g., DeepEP)
- `all`: most optional dependencies (includes `vlm` and `cuda`)

Install extras with pip:

```bash
pip3 install "nemo-automodel[vlm]"
pip3 install "nemo-automodel[all]"

# Add flash-attn (optional)
pip3 install "nemo-automodel[all,fa]"
```

## Summary
| Goal                        | Command or Method                                               |
| --------------------------- | --------------------------------------------------------------- |
| Stable install (PyPI)       | `pip3 install nemo-automodel`                                   |
| Latest from GitHub          | `pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git` |
| Editable install (dev mode) | `pip install -e .` after cloning                                |
| Run without installing      | Use `PYTHONPATH=$(pwd)` to run scripts                          |
| Use in Docker container     | Mount repo and `pip install -e .` inside container              |
| Fast install (via `uv`)     | `uv pip install ...`                                            |
