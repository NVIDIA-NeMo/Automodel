# 🤖 Install NeMo Automodel

This guide explains how to install NeMo AutoModel for LLM, VLM, and OMNI models on various platforms and environments. Depending on your use case, there are several ways to install it:

| Method                  | Dev Mode | Use Case                                                          | Recommended For             |
| ----------------------- | ---------|----------------------------------------------------------------- | ---------------------------- |
| 📦 **PyPI**             | - | Install stable release with minimal setup                         | Most users, production usage |
| 🐳 **Docker**           | - | Use in isolated GPU environments, e.g., with NeMo container       | Multinode deployments     |
| 🐍 **Git Repo**         | ✅ | Use the latest code without cloning or installing extras manually | Power users, testers         |
| 🧪 **Editable Install** | ✅ | Contribute to the codebase or make local modifications            | Contributors, researchers    |
| 🐳 **Docker + Mount**   | ✅ | Use in isolated GPU environments, e.g., with NeMo container       | Multinode deployments     |

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
This section explains the easiest installation options for non-developers, including using pip3 via PyPI or leveraging a preconfigured NVIDIA NeMo Docker container. Both methods offer quick access to the latest stable release of NeMo Automodel with all required dependencies.
### 📦 Install via PyPI (Recommended)

For most users, the easiest way to get started is using `pip3`.

```bash
pip3 install nemo-Automodel
```
> [!TIP]
> This installs the latest stable release of NeMo Automodel from PyPI, along with all of its required dependencies.

### Install via NeMo Docker Container
You can use NeMo Automodel with the NeMo Docker container. You can pull the container by running:
```bash
docker pull nvcr.io/nvidia/nemo:25.07
```
> [!NOTE]
> The above `docker` command uses the `25.07` container. Use the most recent container version to ensure you get the latest version of AutoModel and its dependencies like torch, transformers, etc.

Then you can enter the container using:
```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  nvcr.io/nvidia/nemo:25.07
```

---
## Installation Options for Developers
This section provides installation options for developers, including pulling the latest source from GitHub, using editable mode, or mounting the repo inside a NeMo Docker container.
### 🐍 Install via GitHub (Source)

If you want the **latest features** from the `main` branch or want to contribute:

#### Option A - Use `pip` with git repo:
```bash
pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git
```
> [!NOTE]
> This installs the repo as a standard Python package (not editable).


#### Option B - Use `uv` with git repo:
```bash
uv pip install git+https://github.com/NVIDIA-NeMo/Automodel.git
```
> [!NOTE]
> `uv` handles virtual environment transparently and enables more reproducible installs.


### 🧪 Install in Developer Mode (Editable Install)
To contribute or modify the code:
```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
pip3 install -e .
```

> [!NOTE]
> 🛠️ This installs Automodel in editable mode, so changes to the code are immediately reflected in Python.


### 🐳 Mount the Repo into a NeMo Docker Container
To run `Automodel` inside a NeMo container while **mounting your local repo**, follow these steps:

```
# Step 1: Clone the Automodel repository.
git clone https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel && \

# Step 2: Pull the latest compatible NeMo container (replace 25.07 with latest if needed).
docker pull nvcr.io/nvidia/nemo:25.07 && \

# Step 3: Run the NeMo container with GPU support, shared memory, and mount the repo.
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/Automodel \         # Mount repo into container workspace
  -v $(pwd)/Automodel:/opt/Automodel \     # Optional: Mount Automodel under /opt for flexibility
  --shm-size=8g \                           # Increase shared memory for PyTorch/data loading
  nvcr.io/nvidia/nemo:25.07 /bin/bash -c "\
    cd /workspace/Automodel && \           # Enter the mounted repo
    pip install -e . && \                  # Install Automodel in editable mode
    python3 examples/llm/finetune.py" # Run a usage example
```
> [!NOTE]
> The above `docker` command uses the volume `-v` option to mount the local `Automodel` directory
> under `/opt/Automodel`.

## 🧪 Bonus: Install Extras
Some functionality may require optional extras. You can install them like this:
```bash
pip3 install nemo-Automodel[cli]    # Installs only the Automodel CLI
pip3 install nemo-Automodel         # Installs the CLI and all LLM dependencies.
pip3 install nemo-Automodel[vlm]    # Install all VLM-related dependencies.
```

## 📌 Summary
| Goal                        | Command or Method                                               |
| --------------------------- | --------------------------------------------------------------- |
| Stable install (PyPI)       | `pip3 install nemo-Automodel`                                   |
| Latest from GitHub          | `pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git` |
| Editable install (dev mode) | `pip install -e .` after cloning                                |
| Run without installing      | Use `PYTHONPATH=$(pwd)` to run scripts                          |
| Use in Docker container     | Mount repo and `pip install -e .` inside container              |
| Fast install (via `uv`)     | `uv pip install ...`                                            |
