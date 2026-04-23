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

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3
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
ENV UV_VERSION="0.10.11"
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
ARG INSTALL_TE=False
ARG TE_COMMIT=release_v2.11
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

# Install HybridEP
## Dependency: RDMA Core
RUN git clone https://github.com/linux-rdma/rdma-core.git && \
    cd rdma-core && git checkout tags/v60.0 && sh build.sh
ENV RDMA_CORE_HOME=/opt/rdma-core/build
## Use stub of libnvidia-ml-dev during build only
RUN apt-get update && \
    apt-get install -y --no-install-recommends libnvidia-ml-dev
COPY docker/common/deepep.patch /opt/deepep.patch
ARG DEEPEP_COMMIT=7febc6e25660af0f54d95dd781ecdcd62265ecca
RUN git clone -b hybrid-ep https://github.com/deepseek-ai/DeepEP.git
ENV HYBRID_EP_MULTINODE=1
RUN cd DeepEP && \
    git fetch origin $DEEPEP_COMMIT && \
    git checkout FETCH_HEAD && \
    patch -p1 < /opt/deepep.patch && \
    pip install --no-cache-dir nvidia-nvshmem-cu13==3.4.5 && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" MAX_JOBS=8 pip install --no-build-isolation . && \
    apt-get purge -y libnvidia-ml-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /opt/deepep.patch && \
    cd / && rm -rf DeepEP rdma-core

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
    apt-get update -qq && \
    apt-get install -y --no-install-recommends libibverbs-dev librdmacm-dev && \
    rm -rf /var/lib/apt/lists/* && \
    bash /opt/setup_uccl_ep.sh --no-efa --skip-apt && \
    rm -f /opt/setup_uccl_ep.sh; \
    fi

# Address base image CVE
RUN pip install "aiohttp>=3.13.3" \
        "black>=26.3.1" \
        "jaraco-context>=6.1.0" \
        "nbconvert>=7.17.0" \
        "onnx>=1.21.0" \
        "pillow>=12.1.1" \
        "protobuf>=6.33.5" \
        "setuptools>=80.10.2" \
        "tornado>=6.5.5" \
        "urllib3>=2.6.0" && \
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

# Patch wandb-core: bump vulnerable Go deps (CVE fixes)
ARG TARGETARCH
RUN if python3 -c "import wandb" 2>/dev/null; then \
        GRPC_VERSION=1.79.3 && \
        OTEL_VERSION=1.43.0 && \
        GO_VERSION=1.26.1 && \
        WANDB_VERSION=$(python3 -c "import wandb; print(wandb.__version__)") && \
        WANDB_CORE_BIN=/opt/venv/lib/python3.12/site-packages/wandb/bin/wandb-core && \
        curl -fsSL "https://dl.google.com/go/go${GO_VERSION}.linux-${TARGETARCH}.tar.gz" | tar -C /tmp -xz && \
        export PATH="/tmp/go/bin:$PATH" && \
        export GOPATH=/tmp/gopath && \
        git clone --depth 1 --branch "v${WANDB_VERSION}" https://github.com/wandb/wandb.git /tmp/wandb-src && \
        cd /tmp/wandb-src/core && \
        go get google.golang.org/grpc@v${GRPC_VERSION} && \
        go get go.opentelemetry.io/otel/sdk@v${OTEL_VERSION} && \
        go get github.com/go-jose/go-jose/v4@v4.1.4 && \
        go mod tidy && \
        go mod vendor && \
        CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -trimpath -ldflags="-s -w" \
            -o "${WANDB_CORE_BIN}" ./cmd/wandb-core/ && \
        rm -rf /tmp/wandb-src /tmp/go /tmp/gopath; \
    else \
        echo "wandb not installed, skipping CVE patch"; \
    fi

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
