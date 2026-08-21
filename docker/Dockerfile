# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.04-cuda13.2-devel-ubuntu24.04
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3
ARG BASE_IMAGE=cuda

FROM ${CUDA_IMAGE} AS cuda
# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-dev \
    python3 \
    python3-dev \
    python3-venv \
    python-is-python3 \
    curl \
    git \
    libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

FROM ${PYTORCH_IMAGE} AS pytorch

FROM ${BASE_IMAGE} AS update_base_container

ENV PIP_NO_CACHE_DIR=1
WORKDIR /opt

ENV DEBIAN_FRONTEND=noninteractive
# Address CVE-2025-68973
RUN apt-get update && apt-get install -y --only-upgrade gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the pinned uv and uvx on the system path so runtime user overrides can
# execute them. This also overwrites the base image's vulnerable copies.
ENV UV_VERSION="0.11.24"
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
        amd64) \
            uv_arch="x86_64"; \
            uv_sha256="5ce1ad074a78f96c5c8122088bb85a12eb282195bc1453151a48762e4fc31fed"; \
            ;; \
        arm64) \
            uv_arch="aarch64"; \
            uv_sha256="e22c66d36a0098b17cff80a8647e0b8c58202af899d4e9eb820fc7ad126435a1"; \
            ;; \
        *) \
            echo "Unsupported TARGETARCH for uv: ${TARGETARCH}" >&2; \
            exit 1; \
            ;; \
    esac && \
    uv_archive="uv-${uv_arch}-unknown-linux-gnu.tar.gz" && \
    curl -fLSs -o "/tmp/${uv_archive}" \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${uv_archive}" && \
    echo "${uv_sha256}  /tmp/${uv_archive}" | sha256sum -c - && \
    tar -xzf "/tmp/${uv_archive}" -C /tmp && \
    install -m 0755 "/tmp/uv-${uv_arch}-unknown-linux-gnu/uv" /usr/local/bin/uv && \
    install -m 0755 "/tmp/uv-${uv_arch}-unknown-linux-gnu/uvx" /usr/local/bin/uvx && \
    rm -rf "/tmp/${uv_archive}" "/tmp/uv-${uv_arch}-unknown-linux-gnu"
ENV PATH="/usr/local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532
RUN if ! getent group "${RUNTIME_GID}" >/dev/null; then \
        groupadd --gid "${RUNTIME_GID}" nemo-runtime; \
    fi && \
    if ! getent passwd "${RUNTIME_UID}" >/dev/null; then \
        useradd --no-log-init --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
            --create-home --home-dir /home/nemo-runtime --shell /bin/bash nemo-runtime; \
    fi && \
    install -d -o "${RUNTIME_UID}" -g "${RUNTIME_GID}" \
        /home/nemo-runtime \
        /opt/Automodel \
        /opt/uv_cache \
        /opt/uv_cache/xdg && \
    uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages && \
    chown -R "${RUNTIME_UID}:${RUNTIME_GID}" ${UV_PROJECT_ENVIRONMENT}

# Torchrun uses uv venv
RUN if [ -f /usr/local/bin/torchrun ]; then \
        sed -i '1c\#!/opt/venv/bin/python3' /usr/local/bin/torchrun; \
    fi

FROM update_base_container AS automodel_dep

# Install TE
ARG INSTALL_TE=True
ARG TE_COMMIT=release_v2.14
RUN if [ "$INSTALL_TE" = "True" ]; then \
    git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    git fetch origin $TE_COMMIT && \
    git checkout FETCH_HEAD && \
    git submodule init && git submodule update && \
    pip install nvidia-mathdx==25.1.1 && \
    env NVTE_CUDA_ARCHS="80;90;100;120" NVTE_BUILD_THREADS_PER_JOB=8 pip install --no-cache-dir --no-build-isolation -v . && \
    cd ../ && rm -rf TransformerEngine; \
    fi

# Install HybridEP / DeepEP: apt rdma-core v60 (build==runtime libibverbs), nvshmem wheel 3.6.5.
# DEEPEP_COMMIT = 17cfb817 + DeepEP #638, padding max_num_of_tokens_per_rank to the combine-kernel chunk size.
COPY docker/common/deepep.patch /opt/deepep.patch
ARG DEEPEP_COMMIT=42144303752422ade37f24bca9e2dde12df70e09
ENV HYBRID_EP_MULTINODE=1
ENV RDMA_CORE_HOME=/opt/rdma-core/build
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN apt-get update && apt-get install -y --allow-change-held-packages \
        rdma-core libibverbs-dev && \
    apt-get clean && \
    ARCH_LIB=$(dpkg-architecture -qDEB_HOST_MULTIARCH) && \
    test -f /usr/lib/${ARCH_LIB}/libmlx5.so || \
        ln -sf /usr/lib/${ARCH_LIB}/libmlx5.so.1 /usr/lib/${ARCH_LIB}/libmlx5.so && \
    mkdir -p ${RDMA_CORE_HOME} && \
    ln -sfn /usr/include ${RDMA_CORE_HOME}/include && \
    ln -sfn /usr/lib/${ARCH_LIB} ${RDMA_CORE_HOME}/lib && \
    git clone https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && \
    git fetch origin $DEEPEP_COMMIT && \
    git checkout FETCH_HEAD && \
    patch -p1 < /opt/deepep.patch && \
    pip install --no-cache-dir nvidia-nvshmem-cu13==3.6.5 && \
    NVSHMEM_LIB_PATH=$(pip show nvidia-nvshmem-cu13 | grep "Location:" | cut -d' ' -f2)/nvidia/nvshmem/lib && \
    ln -sf ${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3 ${NVSHMEM_LIB_PATH}/libnvshmem_host.so && \
    apt-get update && apt-get install -y --no-install-recommends libnvidia-ml-dev && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install --no-cache-dir --no-build-isolation -v . && \
    apt-get purge -y libnvidia-ml-dev && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /opt/deepep.patch && \
    cd / && rm -rf DeepEP

# Install Bitsandbytes
ARG INSTALL_BITSANDBYTES=True
ARG BITSANDBYTES_COMMIT=0.49.2
# Include native Blackwell targets, especially SM121 for DGX Spark, instead of relying on forward-compatible PTX.
RUN if [ $INSTALL_BITSANDBYTES = "True" ]; then \
    git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && \
    cd bitsandbytes && \
    git pull && \
    git fetch origin $BITSANDBYTES_COMMIT && \
    git checkout FETCH_HEAD && \
    cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="80;86;90;100;110;120;121" -S . && \
    make && \
    cmake -DCOMPUTE_BACKEND=cpu -S . && \
    make && \
    pip install . && \
    cd ../ && rm -rf bitsandbytes; \
    fi

# Install UCCL-EP (Azure-compatible RDMA for expert parallelism)
COPY scripts/setup_uccl_ep.sh /opt/setup_uccl_ep.sh
ARG INSTALL_UCCL_EP=False
RUN if [ "$INSTALL_UCCL_EP" = "True" ]; then \
    bash /opt/setup_uccl_ep.sh --no-efa && \
    rm -f /opt/setup_uccl_ep.sh; \
    fi

# ---- Wheel builder: compile the heavy AOT source packages off the uv-sync path (cached across uv.lock bumps).
# Build torchao then FA3/FA4 sequentially and cap compiler parallelism to stay within CI memory limits.
FROM automodel_dep AS wheel_builder

# torchao MXFP8: build w/ submodules (PyPI wheels omit mxfp8 CUDA kernels on aarch64); pin v0.17.0 (v0.14.1 ~2.5x slower; main needs nvidia-cutlass-dsl).
ARG TORCHAO_REF=v0.17.0
RUN mkdir -p /wheels && \
    git clone --depth 1 --branch ${TORCHAO_REF} --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/ao.git /tmp/torchao && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" MAX_JOBS=8 \
        pip wheel --no-cache-dir --no-build-isolation --no-deps -w /wheels /tmp/torchao && \
    rm -rf /tmp/torchao

# FA3 (Hopper-only, SM90a-locked, ~25 min nvcc) + FA4 cute (Blackwell, lightweight). arm64 -> skip FA3: GB200 target, QEMU cross-build impractical.
ARG FLASH_ATTN_REF=002cce0a1
ARG INSTALL_FA3=true
ARG INSTALL_FA4=false
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then INSTALL_FA3=false; fi && \
    if [ "$INSTALL_FA3" = "true" ] || [ "$INSTALL_FA4" = "true" ]; then \
        git clone --filter=blob:none https://github.com/Dao-AILab/flash-attention.git /tmp/fa && \
        cd /tmp/fa && git checkout ${FLASH_ATTN_REF} && \
        git submodule update --init csrc/cutlass; \
    fi && \
    if [ "$INSTALL_FA3" = "true" ]; then \
        cd /tmp/fa/hopper && \
        FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=16 NVCC_THREADS=2 \
        pip wheel --no-cache-dir --no-build-isolation --no-deps -w /wheels . ; \
    fi && \
    if [ "$INSTALL_FA4" = "true" ]; then \
        pip wheel --no-cache-dir --no-deps -w /wheels /tmp/fa/flash_attn/cute ; \
    fi && \
    rm -rf /tmp/fa

FROM automodel_dep AS automodel_final

WORKDIR /opt/Automodel

# torchao MXFP8 + FA3/FA4: prebuilt wheels from the wheel_builder stage, installed into system site-packages
# before uv sync — pip targets system deps, uv the venv (torchao/FA gated `never`), so uv sync won't prune or shadow them.
COPY --from=wheel_builder /wheels /tmp/wheels/
ARG INSTALL_FA4=false
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && \
    if [ "$INSTALL_FA4" = "true" ]; then \
        pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.6.0.dev0" && \
        FA2_DIR=$(python -c "import flash_attn, os; print(os.path.dirname(flash_attn.__file__))") && \
        VENV_CUTE=/opt/venv/lib/python3.12/site-packages/flash_attn/cute && \
        { [ -e "$FA2_DIR/cute" ] || ln -s "$VENV_CUTE" "$FA2_DIR/cute"; }; \
    fi && \
    rm -rf /tmp/wheels

# Address base image CVE
RUN pip install "aiohttp>=3.14.3" \
        "black>=26.3.1" \
        "GitPython>=3.1.59" \
        "jaraco-context>=6.1.0" \
        "jupyter-server>=2.20.0" \
        "jupyterlab>=4.5.7" \
        "mistune>=3.3.0" \
        "nbconvert>=7.17.0" \
        "notebook>=7.5.6" \
        "onnx>=1.21.0" \
        "pillow>=12.3.0" \
        "protobuf>=6.33.5" \
        "setuptools>=80.10.2" \
        "tornado>=6.5.5" \
        "urllib3>=2.7.0" && \
    pip uninstall -y nvidia-dali-cuda130 wandb && \
    rm -rf /opt/pytorch/pytorch/third_party/onnx

ARG RUNTIME_UID=65532
ARG RUNTIME_GID=65532
COPY --chown=${RUNTIME_UID}:${RUNTIME_GID} pyproject.toml uv.lock /opt/Automodel/
COPY --chown=${RUNTIME_UID}:${RUNTIME_GID} nemo_automodel/__init__.py nemo_automodel/package_info.py /opt/Automodel/nemo_automodel/
COPY --chown=${RUNTIME_UID}:${RUNTIME_GID} docker/common/uv-pytorch.toml docker/common/uv-pytorch.lock /opt/Automodel/docker/common/
COPY --chown=${RUNTIME_UID}:${RUNTIME_GID} docker/common/update_pyproject_pytorch.sh /opt/Automodel/docker/common/

# Install Automodel
ARG BASE_IMAGE=cuda
ARG AUTOMODEL_INSTALL=all
ARG UV_SYNC_ARGS="--locked"
# Preserve locally built wheels for source-only CUDA packages such as flash-attn,
# causal-conv1d, mamba-ssm, and nv-grouped-gemm across BuildKit builds.
# Ray's wheel bundles Java worker jars (ray/jars/ray_dist.jar) that vendor Jackson
# releases flagged by the container scan (GHSA-r7wm-3cxj-wff9 and the earlier
# jackson-databind findings); upstream Ray still ships jackson 2.18.6. Ray reaches
# the venv only as a transitive Python dependency and Automodel never starts Java
# workers, so drop the jars.
RUN --mount=type=cache,target=/opt/uv_cache,sharing=locked \
    if [ "$BASE_IMAGE" = "pytorch" ]; then \
        bash docker/common/update_pyproject_pytorch.sh /opt/Automodel; \
    fi && \
    MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY="90,100" \
        MAGI_ATTENTION_PREBUILD_FFA=0 \
        MAX_JOBS=8 \
        UV_CONCURRENT_INSTALLS=8 \
        uv sync --extra $AUTOMODEL_INSTALL --all-groups $UV_SYNC_ARGS && \
    python -c "import importlib.metadata as m; import magi_attention; print('magi-attention', m.version('magi-attention'))" && \
    rm -rf ${UV_PROJECT_ENVIRONMENT}/lib/python*/site-packages/ray/jars && \
    chown -R "${RUNTIME_UID}:${RUNTIME_GID}" ${UV_PROJECT_ENVIRONMENT}

# Persistent disk cache for FA4's CuTe-DSL JIT (amortizes kernel compiles across restarts)
ENV FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

COPY --chown=${RUNTIME_UID}:${RUNTIME_GID} . /opt/Automodel

# The checkout is owned by RUNTIME_UID but the image still runs as root by default,
# so Git would reject /opt/Automodel for dubious ownership. Trust it system-wide to
# keep both identities able to read the repository history.
RUN git config --system --add safe.directory /opt/Automodel

# Re-apply PyTorch overrides after full COPY (which overwrites the modified pyproject.toml/uv.lock)
RUN if [ "$BASE_IMAGE" = "pytorch" ]; then \
        bash docker/common/update_pyproject_pytorch.sh /opt/Automodel; \
    fi

WORKDIR /opt/Automodel

COPY <<EOF /opt/venv/env.sh
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/usr/local/bin:$PATH"
EOF

RUN chmod +x /opt/venv/env.sh

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ARG RC_DATE=00.00
ARG TARGETARCH
# NOTICES.txt file points to where the OSS source code is archived
RUN echo "This distribution includes open source which is archived at the following URL: https://opensource.nvidia.com/oss/teams/nvidia/nemo-automodel/${RC_DATE}:linux-${TARGETARCH}/index.html" > NOTICES.txt && \
    echo "For further inquiries or assistance, contact us at oss-requests@nvidia.com" >> NOTICES.txt

# Runtime callers select this UID and set HOME/cache variables explicitly; the
# published image remains root by default. Prove that the non-root contract
# works before restoring that default.
USER ${RUNTIME_UID}:${RUNTIME_GID}
RUN export HOME=/home/nemo-runtime \
        XDG_CACHE_HOME=/opt/uv_cache/xdg && \
    test "$(id -u)" = "${RUNTIME_UID}" && \
    test -w "${HOME}" && \
    test -w "${XDG_CACHE_HOME}" && \
    test -w "${UV_CACHE_DIR}" && \
    test -x /usr/local/bin/uv && \
    test -x /usr/local/bin/uvx && \
    test -x /opt/venv/bin/python && \
    venv_site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')" && \
    touch /opt/Automodel/.nonroot-write-probe \
        /opt/venv/.nonroot-write-probe \
        /opt/uv_cache/.nonroot-write-probe \
        "${venv_site_packages}/.nonroot-write-probe" && \
    rm /opt/Automodel/.nonroot-write-probe \
        /opt/venv/.nonroot-write-probe \
        /opt/uv_cache/.nonroot-write-probe \
        "${venv_site_packages}/.nonroot-write-probe"

USER root
