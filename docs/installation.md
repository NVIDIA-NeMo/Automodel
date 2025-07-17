# 🤖 Install NeMo Automodel

NeMo Automodel support LLM, VLM, and OMNI models. Depending on your use case, there are several ways to install it:

| Method                             | Dev Mode | Use Case                                                          | Recommended For             |
| ---------------------------------- | ---------|----------------------------------------------------------------- | ---------------------------- |
| 📦 **PyPI**                        | - | Install stable release with minimal setup                         | Most users, production usage |
| 🐳 **Docker**                      | - | Use in isolated GPU environments, e.g., with NeMo container       | Multinode deployments     |
| 🐍 **GitHub (pip or uv)**          | ✅ | Use the latest code without cloning or installing extras manually | Power users, testers         |
| 🧪 **Editable Install** | ✅ | Contribute to the codebase or make local modifications            | Contributors, researchers    |
| 🐳 **Docker + Mount**     | ✅ | Use in isolated GPU environments, e.g., with NeMo container       | Multinode deployments     |


---
# Installation options for non-developers

## 📦 Install via PyPI (Recommended)

For most users, the easiest way to get started is using `pip3`.

```bash
pip3 install nemo-automodel
```
> [!TIP]
> This installs the latest stable release of nemo-automodel from PyPI along with all required dependencies.

## NeMo docker container
You can use NeMo Automodel with NeMo docker container. You can pull the container by running:
```bash
docker pull nvcr.io/nvidia/nemo:25.07
```
> [!NOTE]
> The above `docker` command uses the `25.07` container, but it's recommended to use the most recent,
> to ensure you get the latest version of Automodel and its dependencies like torch, transformers, etc.

Then you can enter the container using:
```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  nvcr.io/nvidia/nemo:25.07
```

---
# Installation options for developers

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
> The above `docker` command uses the `25.07` container, but it's recommended to use the most recent,
> to ensure you get the latest version of Automodel and its dependencies like torch, transformers, etc.

### Step 3 - run the container:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/automodel \
  --shm-size=8g \
  nvcr.io/nvidia/nemo:25.07
```

You now have full access to `nemo-automodel` contained inside the NeMo docker container!


### Optional: Mount a local Automodel repo in docker container:
```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/automodel \
  --shm-size=8g \
  -v $(pwd)/Automodel:/opt/Automodel \
  nvcr.io/nvidia/nemo:25.07
```
> [!NOTE]
> The above `docker` command uses the volume `-v` option to mount the local `Automodel` directory
> under `/opt/Automodel`. This works for most cases, where dependencies stay the same.

### Optional: Install a local Automodel repo inside a docker container:
```bash
cd /workspace/automodel
pip install -e .
python3 examples/example_load_model.py
```


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
