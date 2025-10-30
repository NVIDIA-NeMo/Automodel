# Contributing To NeMo-Automodel

## Building NeMo-Automodel container

* Testing and validation occurs through the Automodel container. This is the primary and recommended path for development. Container can be built using the following command:

```bash
# Set build arguments
export AUTOMODEL_INSTALL=vlm #[fa, moe, vlm]
export BASE_IMAGE=pytorch #[cuda, pytorch]
export INSTALL_DEEPEP=True #[True, False]

docker build -f docker/Dockerfile \
--build-arg AUTOMODEL_INSTALL=$AUTOMODEL_INSTALL \
--build-arg BASE_IMAGE=$BASE_IMAGE \
--build-arg INSTALL_DEEPEP=$INSTALL_DEEPEP \
-t automodel --target=automodel_final .
```

* All testing is currently executed with PyTorch base image. This is the recommended installation path.

* Run the following command to start your container:

```bash
docker run --rm -it --entrypoint bash --runtime nvidia --gpus all automodel
```

## Installing dependencies from source

* Install Transformer Engine from source:

```bash
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout <commit_hash>
git submodule init && git submodule update
pip install nvidia-mathdx==25.1.1
env NVTE_CUDA_ARCHS="80;90;100;120" NVTE_BUILD_THREADS_PER_JOB=8 pip install --no-cache-dir --no-build-isolation -v .
```

* Install Grouped Gemm:

```bash
RUN pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

* Install DeepEP from source:

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
git checkout <commit_hash>
patch -p1 < <path_to_Automodel>/docker/common/deepep.patch
pip install --no-cache-dir nvidia-nvshmem-cu13
TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install --no-cache-dir --no-build-isolation -v .
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
