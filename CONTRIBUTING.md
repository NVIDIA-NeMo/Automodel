# Contributing To NeMo-Automodel

## Environment setup for Automodel

Common workflows used for setting up Automodel environment:

1. [Developing with Automodel container](#1-developing-with-automodel-container)
2. [Developing with UV sync/pip install]($2-developing-with-uv-syncpip-install)
3. [Developing with custom docker build](#3-developing-with-custom-docker-build)

### 1. Developing with Automodel container

The latest Automodel container can be found: [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel)

The container can be run with the following docker command:

```bash
docker run --gpus all --network=host -it --rm --shm-size=32g nvcr.io/nvidia/nemo-automodel:25.11.00 /bin/bash
```

#### Mounting local Automodel directory into the container

To sync local Automodel directory into the container, mount the local directory into `/opt/Automodel` and override the installed Automodel repository.
Example docker command:

```bash
docker run --gpus all --network=host -it --rm -v <local-Automodel-path>:/opt/Automodel --shm-size=32g nvcr.io/nvidia/nemo-automodel:25.11.00 /bin/bash
```

Within the container, cd into `/opt/Automodel/` and update the pyproject.toml and uv.lock file by running the following command:

```bash
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
```

Finally, run uv sync to sync the container with the updated repository:

```bash
uv sync --locked --extra all --all-groups
```

> [!WARNING]
> Ensure `bash docker/common/update_pyproject_pytorch.sh /opt/Automodel` is executed. Without this command, uv sync will attempt to reinstall `torch`. This leads to errors relating to CUDA version mismatch, TE import failures, etc. This work around is required as uv cannot recognize the torch installed in the PyTorch base container.

### 2. Developing with uv sync/pip install

Uv sync and pip install are both supported in Automodel. Uv sync is the recommened path.

From the local Automodel directory run the following command:

```bash
uv sync --locked --extra all
```

The following optional dependencies are available, please see `[project.optional-dependencies]` section in [pyproject.toml](./pyproject.toml):

- cuda (all dependencies that require cuda)
- extra (additional dependencies for model coverage)
- fa (flash attention)
- delta-databricks
- moe
- vlm
- all (installs cuda, delta-databricks, extra and vlm)

Example, installing vlm dependencies:

```bash
uv sync --locked --extra vlm
```

### 3. Developing with custom docker build

For developers building a custom docker container with Automodel, please refer to Automodel's [Dockerfile](./docker/Dockerfile).

If [Nvidia PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) container is used as a base image, please review [Developing with Automodel container](#1-developing-with-automodel-container).

Command to build Automodel's Dockerfile:

```bash
# Set build arguments
export AUTOMODEL_INSTALL=all #(cuda, moe, vlm, ...)
export BASE_IMAGE=pytorch #(cuda, pytorch)

# Dependency install options [True, False]
export INSTALL_DEEPEP=True
export INSTALL_TE=True

docker build -f docker/Dockerfile \
    --build-arg AUTOMODEL_INSTALL=$AUTOMODEL_INSTALL \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    --build-arg INSTALL_DEEPEP=$INSTALL_DEEPEP \
    --build-arg INSTALL_MAMBA=$INSTALL_MAMBA \
    -t automodel --target=automodel_final .
```

## MoE Dependency

* Requires [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cudnn-cuda-12
pip install transformer-engine[pytorch]==2.8.0
# Flash-attn version should be selected to satisfy TE requirements
# https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/attention/dot_product_attention/utils.py#L108
pip install flash-attn==2.7.4.post1
pip install grouped_gemm
```

## Development Dependencies

We use [uv](https://docs.astral.sh/uv/) for managing dependencies.

New required dependencies can be added by `uv add $DEPENDENCY`.

Adding a new dependency will update UV's lock-file. Please check this into your branch:

```bash
git add uv.lock pyproject.toml
git commit -s -m "build: Adding dependencies"
git push
```

## Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

Installation:

```bash
pip install ruff
```

Format:

```bash
ruff check --fix .
ruff format .
```

## Pre-commit

We recommand to use [perk](https://github.com/j178/prek) to ensure code quality. It is a faster and more modern alternative to [pre-commit](https://github.com/pre-commit/pre-commit).

Installation:

```bash
uv tool install perk
```

Usage:

```bash
# Install git hooks
perk install

# Run manually on all files
perk run --all-files
```

After installing the git hooks, `git commit` will automatically run incremental checks.

## Adding Documentation

If your contribution involves documentation changes, please refer to the [Documentation Development](docs/documentation.md) guide for detailed instructions on building and serving the documentation.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
