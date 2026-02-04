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
This installs the latest stable release of NeMo Automodel from PyPI, along with all of its required dependencies.
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

## Troubleshooting

### FlashAttention (FA) import fails with `undefined symbol`: ABI mismatch (prebuilt wheel vs your PyTorch)
If `flash-attn` installs very quickly and then fails to import with an error mentioning an **undefined symbol**, you likely installed a prebuilt wheel that doesn’t match your local PyTorch/CUDA ABI.

Example symptom (C++ symbols are often shown *mangled*; you may also see a partially demangled name):

```text
ImportError: .../flash_attn_2_cuda*.so: undefined symbol: _ZN3c1011SymInt6sym_neERKS0_
# sometimes shown as: c10::SymInt::sym_ne(...)
```

To demangle a symbol:

```bash
echo '_ZN3c1011SymInt6sym_neERKS0_' | c++filt
```

**Confirm `flash-attn` is installed and working**

Quick import check:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  uv run --extra fa python -c "import torch, flash_attn; print('flash-attn import OK', torch.__version__)"
```

Kernel sanity check (requires a CUDA GPU):

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  uv run --extra fa python - <<'PY'
import torch
from flash_attn.flash_attn_interface import flash_attn_func

assert torch.cuda.is_available(), "CUDA not available"
q = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 8, 2, 64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
print("flash-attn kernel OK:", out.shape, out.dtype)
PY
```

**Detect / confirm which `.so` is being loaded**

```bash
python - <<'PY'
import sysconfig, pathlib
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
cands = sorted(purelib.glob("flash_attn_2_cuda*.so"))
print("Found:", cands)
PY
```

If you see a candidate `.so`, you can inspect its RPATH/RUNPATH:

```bash
readelf -d /path/to/flash_attn_2_cuda*.so | grep -E 'RPATH|RUNPATH' || true
```

**Fix: force a real local build of `flash-attn` (recommended for `uv`)**

```bash
# Keep the environment clean so extensions load against the venv's torch.
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

uv pip uninstall flash-attn

# Optional but recommended: avoid reusing cached incompatible wheels.
export UV_CACHE_DIR="$(mktemp -d)"

CUDA_HOME=/usr/local/cuda FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  uv pip install --no-build-isolation --no-binary flash-attn "flash-attn==2.8.3"
```

::::{warning}
Avoid setting `LD_LIBRARY_PATH` to another environment’s `torch/lib` (for example from your shell startup files). That can cause `undefined symbol` errors or crashes when loading CUDA extensions in this environment.
::::

### CUDA extensions fail to build with missing headers (`cudnn.h`, `nccl.h`)
When installing `nemo-automodel[cuda]` (or running `uv run --extra cuda`) you may hit build failures while compiling CUDA/C++ extensions (e.g., `transformer-engine-torch`).

Typical errors look like:

```text
fatal error: cudnn.h: No such file or directory
```

or:

```text
fatal error: nccl.h: No such file or directory
```

**What’s happening**: the compiler can’t find cuDNN / NCCL header files. With `uv`, those headers are often present inside the environment under `site-packages/nvidia/...`, but not on the default include path.

**Detect (are the headers available in this environment?)**

```bash
python - <<'PY'
import sysconfig, pathlib
purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
for name, header in [("cudnn", "include/cudnn.h"), ("nccl", "include/nccl.h")]:
    root = purelib / "nvidia" / name
    print(f"{name.upper()}_HOME candidate: {root}")
    print(f"  exists: {root.exists()}")
    print(f"  header: {root / header} (exists: {(root / header).exists()})")
PY
```

**Fix (venv-only; points builds at the `nvidia-*` wheels)**

```bash
export CUDNN_HOME="$(python - <<'PY'
import sysconfig, pathlib
print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cudnn")
PY
)"
export NCCL_HOME="$(python - <<'PY'
import sysconfig, pathlib
print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "nccl")
PY
)"

# headers
export CPATH="$CUDNN_HOME/include:$NCCL_HOME/include${CPATH:+:$CPATH}"

# libs (often needed later for link/runtime)
export LIBRARY_PATH="$CUDNN_HOME/lib:$NCCL_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# optional: some builds look for lib64
[ -e "$CUDNN_HOME/lib64" ] || ln -s lib "$CUDNN_HOME/lib64"
[ -e "$NCCL_HOME/lib64" ]  || ln -s lib "$NCCL_HOME/lib64"
```

Then retry the install/build (e.g., `uv run --extra cuda ...` or your `uv pip install ...`).

::::{note}
If you have cuDNN/NCCL installed system-wide, you can instead set `CUDNN_HOME` / `NCCL_HOME` to those install prefixes and skip the `site-packages/nvidia/...` paths.
::::

## Summary
| Goal                        | Command or Method                                               |
| --------------------------- | --------------------------------------------------------------- |
| Stable install (PyPI)       | `pip3 install nemo-automodel`                                   |
| Latest from GitHub          | `pip3 install git+https://github.com/NVIDIA-NeMo/Automodel.git` |
| Editable install (dev mode) | `pip install -e .` after cloning                                |
| Run without installing      | Use `PYTHONPATH=$(pwd)` to run scripts                          |
| Use in Docker container     | Mount repo and `pip install -e .` inside container              |
| Fast install (via `uv`)     | `uv pip install ...`                                            |
