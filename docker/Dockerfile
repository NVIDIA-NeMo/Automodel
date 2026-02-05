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
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
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

# Install uv
ENV UV_VERSION="0.8.22"
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

# Install Deepep
COPY docker/common/deepep.patch /opt/deepep.patch
ARG INSTALL_DEEPEP=False
ARG DEEPEP_COMMIT=1dddd194c26911c35b4f53a148617dd73de0ffc9
RUN if [ "$INSTALL_DEEPEP" = "True" ]; then \
    git clone https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && \
    git pull && \
    git fetch origin $DEEPEP_COMMIT && \
    git checkout FETCH_HEAD && \
    patch -p1 < /opt/deepep.patch && \
    pip install --no-cache-dir nvidia-nvshmem-cu13==3.4.5 && \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install --no-cache-dir --no-build-isolation -v . && \
    rm -rf /opt/deepep.patch && \
    rm -rf DeepEP; \
    fi

# Address base image CVE
RUN pip install "aiohttp>=3.13.3" \
        "jaraco-context>=6.1.0" \
        "protobuf>=6.3.5" \
        "urllib3>=2.6.0" \
        "wheel>=0.46.2"

FROM automodel_dep as automodel_final

WORKDIR /opt/Automodel

COPY pyproject.toml uv.lock /opt/Automodel/
COPY nemo_automodel/__init__.py nemo_automodel/package_info.py /opt/Automodel/nemo_automodel/
COPY docker/common/uv-pytorch.toml docker/common/uv-pytorch.lock /opt/Automodel/docker/common/
COPY docker/common/update_pyproject_pytorch.sh /opt/Automodel/docker/common/

# Install Automodel
ARG BASE_IMAGE=cuda
ARG AUTOMODEL_INSTALL=all
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$BASE_IMAGE" = "pytorch" ]; then \
        bash docker/common/update_pyproject_pytorch.sh /opt/Automodel; \
    fi && \
    uv sync --locked --extra $AUTOMODEL_INSTALL --all-groups && \
    uv cache prune --ci

COPY . /opt/Automodel

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
