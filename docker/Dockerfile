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

ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
ARG AUTOMODEL_BASE_IMAGE=cuda

FROM ${CUDA_BASE_IMAGE} AS cuda
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

FROM ${PYTORCH_BASE_IMAGE} AS pytorch

FROM ${AUTOMODEL_BASE_IMAGE} AS automodel_base

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

# Copy Automodel dependency files
RUN mkdir /opt/Automodel
COPY pyproject.toml uv.lock /opt/Automodel

# Install Deepep
COPY docker/common/deepep.patch /opt/deepep.patch
ARG INSTALL_DEEPEP=False
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
RUN if [ "$INSTALL_DEEPEP" = "True" ]; then \
    git clone https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && \
    git pull && \
    git fetch origin $DEEPEP_COMMIT && \
    git checkout FETCH_HEAD && \
    patch -p1 < /opt/deepep.patch && \
    rm -rf /opt/deepep.patch; fi

# Install Automodel
ARG AUTOMODEL_BASE_IMAGE=cuda
COPY . /opt/Automodel
RUN cd /opt/Automodel && \
    NO_INSTALL_UV_ARGS=$(cat /opt/Automodel/docker/common/pytorch_uv_args.txt) && \
    if [ "$INSTALL_DEEPEP" = "True" ]; then \
        TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" \
        uv add --no-build-isolation -v /opt/DeepEP --no-sync; \
    fi && \
    if [ "$AUTOMODEL_BASE_IMAGE" = "pytorch" ]; then \
        uv sync --locked --extra all --all-groups $NO_INSTALL_UV_ARGS; \
    else \
        uv sync --locked --extra all --all-groups; \
    fi

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
