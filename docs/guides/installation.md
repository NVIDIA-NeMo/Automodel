# Install NeMo AutoModel

This guide explains how to install NeMo AutoModel for LLM, VLM, and OMNI models on various platforms and environments. Depending on your use case, there are several ways to install it:

| Method                  | Dev Mode | Use Case                                                          | Recommended For             |
| ----------------------- | ---------|----------------------------------------------------------------- | ---------------------------- |
| 📦 **PyPI**             | - | Install stable release with minimal setup                         | Most users, production usage |
| 🐳 **Docker**           | - | Use in isolated GPU environments, e.g., with NeMo container       | Multi-node deployments     |
| 🐍 **Git Repo**         | ✅ | Use the latest code without cloning or installing extras manually | Power users, testers         |
| 🧪 **Editable Install** | ✅ | Contribute to the codebase or make local modifications            | Contributors, researchers    |
| 🐳 **Docker + Mount**   | ✅ | Use in isolated GPU environments, e.g., with NeMo container       | Multi-node deployments     |

## Choose Your Installation Method

Pick the installation method that matches your needs and platform.

### Decision Criteria

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Docker Container** | Production, multi-node, Debian-based systems | Reproducible environment, pre-configured dependencies, GPU driver isolation | Larger download size, container overhead |
| **virtualenv (PyPI/Git)** | Local development, quick prototyping, macOS | Fast setup, lightweight, direct code access | Manual dependency management, platform-specific issues |

### When to Use Docker Containers

Use Docker containers when you need:

- **Multi-node deployments**: Containers ensure consistency across cluster nodes
- **Production environments**: Reproducible builds with tested dependency versions
- **GPU driver compatibility**: Isolates CUDA/driver versions from host system
- **Debian-based systems**: Recommended for Ubuntu, Debian, and derivatives due to dependency complexity
- **Complex dependencies**: Pre-configured environment with all optimizations (TransformerEngine, DeepEP, etc.)
- **Team consistency**: Same environment across development, testing, and production

### When to Use virtualenv

Use virtualenv (PyPI, Git, or editable install) when you need:

- **Local development**: Fast iteration on code changes
- **Quick prototyping**: Minimal setup for experimentation
- **macOS systems**: Better native support without container overhead
- **Frequent code changes**: Contributors working on the codebase (use editable install)
- **Compatible GPU drivers**: System has correct CUDA toolkit and drivers installed
- **Lightweight setup**: Minimal disk space and memory footprint

### Platform-Specific Recommendations

#### Linux (Debian-based: Ubuntu, Debian)

**Recommended: Docker Container**

Debian-based systems can have dependency conflicts with system packages. Containers provide isolation and consistency.

```bash
docker pull nvcr.io/nvidia/nemo-automodel:25.11.00
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:25.11.00
```

**Alternative: virtualenv** (if Docker is not available)

Ensure CUDA 11.8+ and compatible drivers are installed:

```bash
# Check CUDA version
nvidia-smi

# Install via PyPI
pip3 install nemo-automodel
```

#### Linux (RHEL, CentOS, Fedora)

**Recommended: Docker Container**

Containers avoid enterprise Linux package management complexity.

Follow the same Docker commands as Debian-based systems above.

#### macOS

**Recommended: virtualenv**

Docker on macOS has GPU limitations. Use native Python installation:

```bash
# Using PyPI
pip3 install nemo-automodel

# Or using uv for reproducible environments
uv pip install nemo-automodel
```

:::{note}
GPU training on macOS is not supported. Use macOS for CPU-based experimentation or remote cluster submission.
:::

#### Windows

**Recommended: WSL2 + Docker**

Run NeMo AutoModel in WSL2 with Docker Desktop:

1. Install WSL2 and Docker Desktop.
2. Use Docker container within WSL2 (follow Linux instructions).

**Alternative: WSL2 and virtualenv**

Install directly in WSL2 Ubuntu environment (follow Debian instructions).

### Common Issues and Solutions

**GPU driver compatibility errors**
- **Problem**: CUDA version mismatch between host and application
- **Solution**: Use Docker container to isolate driver versions

**Dependency conflicts on Debian/Ubuntu**
- **Problem**: System packages conflict with Python packages
- **Solution**: Use Docker container or create isolated virtualenv with `uv`

**Out of memory during container startup**
- **Problem**: Insufficient shared memory for PyTorch data loading
- **Solution**: Increase `--shm-size` parameter (e.g., `--shm-size=16g`)

**TransformerEngine import failures**
- **Problem**: Incorrect CUDA toolkit or missing dependencies
- **Solution**: Use pre-configured Docker container

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **CUDA**: 11.8 or higher (for GPU support)
- **Memory**: Minimum 16GB RAM, 32GB+ recommended
- **Storage**: At least 50GB free space for models and datasets

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **CPU**: Multi-core processor (8+ cores recommended)
- **Network**: Stable internet connection for downloading models

---
## Installation Options for Non-Developers
This section explains the easiest installation options for non-developers, including using pip3 via PyPI or leveraging a preconfigured NVIDIA NeMo Docker container. Both methods offer quick access to the latest stable release of NeMo AutoModel with all required dependencies.

### Install via PyPI (Recommended)

For most users, the easiest way to get started is using `pip3`.

```bash
pip3 install nemo-automodel
```
:::{tip}
This installs the latest stable release of NeMo AutoModel from PyPI.

To verify the install, run `python -c "import nemo_automodel; print(nemo_automodel.__version__)"`. See [nemo-automodel on PyPI](https://pypi.org/project/nemo-automodel/).
:::

### Install with NeMo Docker Container
You can use NeMo AutoModel with the NeMo Docker container. Pull the container by running:
```bash
docker pull nvcr.io/nvidia/nemo-automodel:25.11.00
```
:::{note}
The above `docker` command uses the [`25.11.00`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) container. Use the [most recent container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel) version to ensure you get the latest version of AutoModel and its dependencies like PyTorch, Transformers, etc.
:::

Then you can enter the container using:
```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  nvcr.io/nvidia/nemo-automodel:25.11.00
```

---
## Installation Options for Developers
## Installation Options for Developers

This section provides installation options for developers, including pulling the latest source from GitHub, using editable mode, or mounting the repo inside a NeMo Docker container.
### Install from GitHub (Source)


If you want the **latest features** from the `main` branch or want to contribute:

#### Option A – Use `pip` With Git Repo
```bash
pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git
```
:::{note}
This installs the repo as a standard Python package (not editable).
:::

#### Option B – Use `uv` With Git Repo
```bash
uv pip install git+https://github.com/NVIDIA-NeMo/Automodel.git
```
:::{note}
`uv` handles virtual environment transparently and enables more reproducible installs.
:::

### Install in Developer Mode (Editable Install)
### Install in Developer Mode (Editable Install)

To contribute or modify the code:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
pip3 install -e .
```

:::{note}
This installs AutoModel in editable mode, so changes to the code are immediately reflected in Python.
:::

### Mount the Repo into a NeMo Docker Container
### Mount the Repo into a NeMo Docker Container

To run `Automodel` inside a NeMo container while **mounting your local repo**, follow these steps:

```bash
# Step 1: Clone the Automodel repository.
git clone https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel && \

# Step 2: Pull the latest compatible NeMo container (replace `25.11.00` with [latest](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel?version=25.11.00) if needed).
docker pull nvcr.io/nvidia/nemo-automodel:25.11.00 && \

# Step 3: Run the NeMo container with GPU support, shared memory, and mount the repo.
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/Automodel \         # Mount repo into container workspace
  -v $(pwd)/Automodel:/opt/Automodel \     # Optional: Mount Automodel under /opt for flexibility
  --shm-size=8g \                           # Increase shared memory for PyTorch/data loading
  nvcr.io/nvidia/nemo-automodel:25.11.00 /bin/bash -c "\
    cd /workspace/Automodel && \           # Enter the mounted repo
    pip install -e . && \                  # Install Automodel in editable mode
     python3 examples/llm_finetune/finetune.py" # Run a usage example
```
:::{note}
The above `docker` command uses the volume `-v` option to mount the local `Automodel` directory
under `/opt/Automodel`.
:::

## Bonus: Install Extras
Some functionality may require optional extras. You can install them like this:
```bash
pip3 install nemo-automodel[cli]    # Installs only the Automodel CLI
pip3 install nemo-automodel         # Installs the CLI and all LLM dependencies.
pip3 install nemo-automodel[vlm]    # Install all VLM-related dependencies.
```

## Summary
| Goal                        | Command or Method                                               |
| --------------------------- | --------------------------------------------------------------- |
| Stable install (PyPI)       | `pip3 install nemo-automodel`                                   |
| Latest from GitHub          | `pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git` |
| Editable install (dev mode) | `pip install -e .` after cloning                                |
| Run without installing      | Use `PYTHONPATH=$(pwd)` to run scripts                          |
| Use in Docker container     | Mount repo and `pip install -e .` inside container              |
| Fast install (using `uv`)     | `uv pip install ...`                                            |
