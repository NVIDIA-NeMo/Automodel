# 🤖 Install NeMo Automodel

NeMo Automodel support LLM, VLM, and OMNI models. Depending on your use case, there are several ways to install it.

Automodel can be installed with one of the following ways:
- 📦 Install via PyPI
- 🐍 Install via GitHub (pip or uv)
- 🚀 Run Directly from Git Repo
- 🐳 Mount Repo into NeMo Docker Container
- 🧪 (Optional) Advanced: Editable Install or Dev Mode

---

## 📦 Install via PyPI (Recommended)

For most users, the easiest way to get started is using `pip3`.

```bash
pip3 install nemo-automodel
```
> [!TIP]
> This installs the latest stable release of nemo-automodel from PyPI along with all required dependencies.


## 🐍 Install via GitHub (Source)

If you want the **latest features** from the `main` branch or want to contribute:

### Option A - Using `pip` with git repo:
```bash
pip3 install git+https://github.com/NVIDIA-NeMo/AutoModel.git
```
> [!NOTE]
> This installs the repo as a standard Python package (not editable).


### Option B - Using `uv` with git repo:
```bash
uv pip install git+https://github.com/NVIDIA-NeMo/AutoModel.git
```
> [!NOTE]
> `uv` handles virtual environment transparently and enables more reproducible installs.


## 🧪 Developer Mode (Editable Install)
To contribute or modify the code:
```bash
git clone https://github.com/NVIDIA-NeMo/AutoModel.git
cd AutoModel
pip3 install -e .
```

> [!NOTE]
> 🛠️ This installs AutoModel in editable mode, so changes to the code are immediately reflected in Python.



## Run Without Installing
You can also use the repo directly **without** installing it:
```bash

git clone https://github.com/NVIDIA-NeMo/AutoModel.git
cd AutoModel

# Run your Python script by setting PYTHONPATH
PYTHONPATH=$(pwd) python3 examples/example_load_model.py
```
> [!WARNING]
> This does not installa any dependencies and is only recommended for advanced users.
> Useful for quick testing, but dependency resolution is left to the user.


## 🐳 Mount the Repo into a NeMo Docker Container
To run `AutoModel` inside a NeMo container while **mounting your local repo**, follow these steps:

### Step 1 - Clone the repo locally
```bash
git clone https://github.com/NVIDIA-NeMo/AutoModel.git
cd AutoModel
```

### Step 2 - Pull a compatible NeMo Container
```bash
docker pull nvcr.io/nvidia/nemo:25.07
```
> [!NOTE]
> Use a recent NeMo container that includes the latest dependencies like torch, transformers, etc.

### Step 3 - Mount and run:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/automodel \
  --shm-size=8g \
  -v Automodel:/opt/Automodel \
  nvcr.io/nvidia/nemo:25.07
```

You now have full access to `nemo-automodel` inside the NeMo docker container!

## 🧪 Bonus: Install Extras
Some functionality may require optional extras. You can install them like this:
```bash
pip3 install nemo-automodel[cli]    # Installs only the automodel CLI
pip3 install nemo-automodel         # Installs the CLI and all LLM dependencies.
pip3 install nemo-automodel[vlm]    # Install all VLM-related dependencies.
```

## 📌 Summary
| Goal                        | Command or Method                                               |
| --------------------------- | --------------------------------------------------------------- |
| Stable install (PyPI)       | `pip3 install nemo-automodel`                                   |
| Latest from GitHub          | `pip3 install git+https://github.com/NVIDIA-NeMo/AutoModel.git` |
| Editable install (dev mode) | `pip install -e .` after cloning                                |
| Run without installing      | Use `PYTHONPATH=$(pwd)` to run scripts                          |
| Use in Docker container     | Mount repo and `pip install -e .` inside container              |
| Fast install (via `uv`)     | `uv pip install ...`                                            |
