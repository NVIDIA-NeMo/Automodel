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
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
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
# Address CVE-2025-68973; install ffmpeg dev libs for torchcodec source build
RUN apt-get update && apt-get install -y --only-upgrade gnupg && \
    apt-get install -y --no-install-recommends ffmpeg libavdevice-dev pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ENV UV_VERSION="0.11.24"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
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
# DEEPEP_COMMIT to 42144303 (== 17cfb817 + DeepEP #638), which pads the HybridEP
# max_num_of_tokens_per_rank up to the combine-kernel chunk size so a per-rank token
# count that is not a multiple of 64 (e.g. 672) no longer fails the combine-kernel
# static_assert (MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0).
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

# Address base image CVE
RUN pip install "aiohttp>=3.13.3" \
        "black>=26.3.1" \
        "GitPython>=3.1.50" \
        "jaraco-context>=6.1.0" \
        "jupyter-server>=2.18.0" \
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
    pip uninstall -y nvidia-dali-cuda130 && \
    rm -rf /opt/pytorch/pytorch/third_party/onnx

FROM automodel_dep as automodel_final

WORKDIR /opt/Automodel

COPY pyproject.toml uv.lock /opt/Automodel/
COPY nemo_automodel/__init__.py nemo_automodel/package_info.py /opt/Automodel/nemo_automodel/
COPY docker/common/uv-pytorch.toml docker/common/uv-pytorch.lock /opt/Automodel/docker/common/
COPY docker/common/update_pyproject_pytorch.sh /opt/Automodel/docker/common/

# Install Automodel
ARG BASE_IMAGE=cuda
ARG AUTOMODEL_INSTALL=all
ARG UV_SYNC_ARGS="--locked"
RUN if [ "$BASE_IMAGE" = "pytorch" ]; then \
        bash docker/common/update_pyproject_pytorch.sh /opt/Automodel; \
    fi && \
    uv sync --extra $AUTOMODEL_INSTALL --all-groups $UV_SYNC_ARGS --no-cache

# Rebuild torchcodec from source to match the container's PyTorch ABI
RUN I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
    pip install --no-build-isolation --force-reinstall --no-deps \
        git+https://github.com/pytorch/torchcodec.git@v0.8.0

# MXFP8 MoE training (experts="torch_mm_mxfp8"): the NGC PyTorch base does not bundle
# torchao and uv-pytorch.toml gates it off (sys_platform=='never'), so `uv sync` skips
# it. Install explicitly AFTER the sync so it is not pruned.
#
# Build FROM SOURCE (not `pip install torchao==0.14.0`): the MXFP8 grouped-GEMM path
# needs the compiled mxfp8 CUDA kernels (torchao/csrc/cuda/mx_kernels), but PyPI ships
# only a kernel-less `py3-none-any` wheel and an x86_64-only compiled wheel — no
# aarch64 wheel and no sdist. On the sbsa/aarch64 build the pip wheel would silently
# omit the kernels and MXFP8 would fall back to bf16 at runtime. setup.py has no
# x86-only gate and cross-compiles the sm_100a kernels from the CUDA toolkit version
# (build_for_sm100a = CUDA>=12.8; base is CUDA 13.x), so source builds on both arches.
# --no-build-isolation reuses the container's torch ABI (same as the torchcodec rebuild).
# Pin v0.17.0 (NOT v0.14.1): v0.14.1's forward quantizes via EAGER unfused to_mx (no CUDA
# kernel, ~49ms/iter) — a ~2.5x regression vs bf16. v0.17.0's forward is fully kernelized
# (triton_to_mxfp8_dim0 for both activation and weight + device-side scale swizzle); it is
# the torchao blog version (~1.43x MoE-layer speedup). NOT main: main's forward additionally
# needs the nvidia-cutlass-dsl runtime for its CuTeDSL activation kernel, which this image
# does not install (and _SM100_KERNELS_AVAILABLE doesn't gate it -> runtime crash); v0.17.0
# is triton-only with no extra dep. CUTLASS is a git submodule the mxfp8 kernels need, so
# clone with --recurse-submodules (pip's git+ submodule handling is unreliable). setup.py
# self-gates the mxfp8/cutlass sm_100a kernels on CUDA>=12.8 and hardcodes their gencode
# flags; TORCH_CUDA_ARCH_LIST + MAX_JOBS mirror the in-image DeepEP source build (above) to
# cover torchao's other CUDAExtensions and bound the heavy CUTLASS sm_100 compile.
RUN git clone --depth 1 --branch v0.17.0 --recurse-submodules --shallow-submodules \
        https://github.com/pytorch/ao.git /tmp/torchao && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" MAX_JOBS=8 \
        pip install --no-cache-dir --no-build-isolation /tmp/torchao && \
    rm -rf /tmp/torchao

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
