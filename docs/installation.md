# Install NeMo Automodel

NeMo Automodel can be installed in the following ways, depending on your needs:


# 🧠 Install NeMo Framework

The NeMo Framework supports LLM, Multimodal, ASR, and TTS models. Depending on your use case, there are several ways to install it.

---

## 🎯 Choose Your Installation Path

| User Type         | Goal                                                   | Recommended Method                |
|------------------|--------------------------------------------------------|-----------------------------------|
| **Non-developer** | Use NeMo models/tools with minimal setup               | 🐳 Prebuilt Docker Container      |
| **HPC User**      | SLURM cluster usage with isolated environments         | 📦 Enroot Container               |
| **Developer**     | Modify NeMo code, use dev branch, MM/LLM work          | 🔧 Source Install (Docker or Conda) |
| **Lightweight Dev** | Use NeMo without containers or with specific domains | 🐍 Conda + Pip (binary)           |

---

## 🙋 Non-Developers: Quick and Easy Install

### 🐳 Prebuilt Docker Container (Recommended for most)

1. Create an NGC account: [NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com/signin)
2. Browse containers: [NGC NeMo Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
3. Pull the container:

    ```bash
    docker pull nvcr.io/nvidia/nemo:24.07.0  # Use latest stable tag (yy.mm.patch)
    ```

4. (Optional) SLURM Credential Setup:
   - Login:
     - Username: `$oauthtoken`
     - Password: NGC API Key
   - See [Docker Login to NGC](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#docker-login-to-ngc)

---

### 📦 Enroot Container (For SLURM/HPC users)

1. Sign in to [NGC](https://ngc.nvidia.com/signin)
2. View containers: [NGC NeMo Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
3. Pull and convert:

    ```bash
    enroot import docker://nvcr.io/nvidia/nemo:24.07.0
    ```

4. Create and start:

    ```bash
    enroot create -n nemo nvcr.io+nvidia+nemo+24.07.0.squashfs
    enroot start nemo
    ```

5. Set SLURM Credentials:

    Create `~/.config/enroot/.credentials`:

    ```text
    machine [NGC-REGISTRY-URL] login $oauthtoken password [NGC-API-KEY]
    ...
    ```

    Set permissions:

    ```bash
    chmod 0600 ~/.config/enroot/.credentials
    ```

---

## 👩‍💻 Developers: Source or Custom Setup

### 🐳 Docker + Source Install (Recommended Dev Path)

Use this if you want to:

- Use the latest GitHub `main` branch
- Work on LLM/MM models
- Customize builds

#### 1. Clone NeMo:

```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo```

---

## 🐍 Conda + Pip

This method gives you local control and flexibility. Ideal for minimal setups or when building into existing environments.

> Important: Start with NVIDIA PyTorch container: nvcr.io/nvidia/pytorch:24.07-py3

## 🔧 Install from Source (For LLM/MM Dev)
1. Install NeMo from GitHub:

```
export BRANCH="main"

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
```
2. Install additional dependencies for LLM/MM:

LLM/MM Dependency Instructions

> [!CAUTION]
> ⚠️ Apex & Transformer Engine are:
> Optional for LLM
> Required for MM
> RMSNorm currently requires Apex

## 📌 Summary
| Installation Type	| Pros               | Use Case
|-------------------|--------------------|--------------------------------------
| Docker (prebuilt) | Easy, stable, reproducible | Non-developers, fast start
| Docker (source)	| Latest features, dev-ready | LLM/MM developers
| Enroot	        | SLURM, HPC compatible | Cluster environments
| Conda + Pip       | Custom local installs | Minimalist setups or devs

> [!TIP]
> Use pip install nemo_toolkit['all'] only if you want all domains including LLM/MM.
> or speech-only projects, use ['asr'] and ['tts'] for a lighter install.

