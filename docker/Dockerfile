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

# Install uv; drop the base image's own /usr/local/bin copy (vulnerable rustls-webpki)
ENV UV_VERSION="0.11.24"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    rm -f /usr/local/bin/uv /usr/local/bin/uvx
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

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

# Install HybridEP / DeepEP. Build with apt rdma-core v60 (so build==runtime libibverbs)
# and RDMA_CORE_HOME symlinked to the system install; create the libnvshmem_host.so->.so.3
# symlink required for DeepEP fabric handle ops; use the nvshmem wheel 3.6.5 and pin
# Ablation: PR-673 DeepEP backend with AutoModel's dense routing path unchanged.
ARG DEEPEP_COMMIT=7e880e13c9a2ba80bc9bf89ca811f083987867eb
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
    git clone https://github.com/akoumpa/DeepEP.git && \
    cd DeepEP && \
    git fetch origin $DEEPEP_COMMIT && \
    git checkout FETCH_HEAD && \
    pip install --no-cache-dir nvidia-nvshmem-cu13==3.6.5 && \
    NVSHMEM_LIB_PATH=$(pip show nvidia-nvshmem-cu13 | grep "Location:" | cut -d' ' -f2)/nvidia/nvshmem/lib && \
    ln -sf ${NVSHMEM_LIB_PATH}/libnvshmem_host.so.3 ${NVSHMEM_LIB_PATH}/libnvshmem_host.so && \
    apt-get update && apt-get install -y --no-install-recommends libnvidia-ml-dev && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install --no-cache-dir --no-build-isolation -v . && \
    apt-get purge -y libnvidia-ml-dev && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    cd / && rm -rf DeepEP

# Install Bitsandbytes
ARG INSTALL_BITSANDBYTES=True
ARG BITSANDBYTES_COMMIT=0.49.2
RUN if [ $INSTALL_BITSANDBYTES = "True" ]; then \
    git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && \
    cd bitsandbytes && \
    git pull && \
    git fetch origin $BITSANDBYTES_COMMIT && \
    git checkout FETCH_HEAD && \
    cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY="80;86;90;100;110" -S . && \
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
# torchao then FA3/FA4 as sequential steps so each compiles alone at full MAX_JOBS — no cross-compile OOM. FA3 is the long pole.
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
        FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=$(nproc) NVCC_THREADS=2 \
        pip wheel --no-cache-dir --no-build-isolation --no-deps -w /wheels . ; \
    fi && \
    if [ "$INSTALL_FA4" = "true" ]; then \
        pip wheel --no-cache-dir --no-deps -w /wheels /tmp/fa/flash_attn/cute ; \
    fi && \
    rm -rf /tmp/fa

FROM automodel_dep as automodel_final

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
RUN pip install "aiohttp>=3.13.3" \
        "black>=26.3.1" \
        "GitPython>=3.1.50" \
        "jaraco-context>=6.1.0" \
        "jupyter-server>=2.20.0" \
        "jupyterlab>=4.5.7" \
        "mistune>=3.2.1" \
        "nbconvert>=7.17.0" \
        "notebook>=7.5.6" \
        "onnx>=1.21.0" \
        "pillow>=12.2.0" \
        "protobuf>=6.33.5" \
        "setuptools>=80.10.2" \
        "tornado>=6.5.5" \
        "urllib3>=2.7.0" && \
    pip uninstall -y nvidia-dali-cuda130 wandb && \
    rm -rf /opt/pytorch/pytorch/third_party/onnx

COPY pyproject.toml uv.lock /opt/Automodel/
COPY nemo_automodel/__init__.py nemo_automodel/package_info.py /opt/Automodel/nemo_automodel/
COPY docker/common/uv-pytorch.toml docker/common/uv-pytorch.lock /opt/Automodel/docker/common/
COPY docker/common/update_pyproject_pytorch.sh /opt/Automodel/docker/common/

# Install Automodel
ARG BASE_IMAGE=cuda
ARG AUTOMODEL_INSTALL=all
ARG UV_SYNC_ARGS="--locked"
# Preserve locally built wheels for source-only CUDA packages such as flash-attn,
# causal-conv1d, mamba-ssm, and nv-grouped-gemm across BuildKit builds.
# DeepEP is already compiled above against the PyTorch image's Torch. The final
# PyTorch sync must not rebuild it after the lock intentionally removes runtime Torch.
RUN --mount=type=cache,target=/opt/uv_cache,sharing=locked \
    if [ "$BASE_IMAGE" = "pytorch" ]; then \
        bash docker/common/update_pyproject_pytorch.sh /opt/Automodel && \
        DEEPEP_SYNC_ARGS="--no-install-package deep-ep"; \
    else \
        DEEPEP_SYNC_ARGS=""; \
    fi && \
    MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY="90,100" \
        MAGI_ATTENTION_PREBUILD_FFA=0 \
        MAX_JOBS=8 \
        UV_CONCURRENT_INSTALLS=8 \
        uv sync --extra $AUTOMODEL_INSTALL --all-groups $UV_SYNC_ARGS $DEEPEP_SYNC_ARGS && \
    python -c "import importlib.metadata as m; import magi_attention; print('magi-attention', m.version('magi-attention')); assert m.version('deep-ep') == '1.2.1+7e880e1'"

# Persistent disk cache for FA4's CuTe-DSL JIT (amortizes kernel compiles across restarts)
ENV FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

COPY . /opt/Automodel

# Re-apply PyTorch overrides after full COPY (which overwrites the modified pyproject.toml/uv.lock)
RUN if [ "$BASE_IMAGE" = "pytorch" ]; then \
        bash docker/common/update_pyproject_pytorch.sh /opt/Automodel; \
    fi

WORKDIR /opt/Automodel

COPY <<EOF /opt/venv/env.sh
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/root/.local/bin:$PATH"
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
