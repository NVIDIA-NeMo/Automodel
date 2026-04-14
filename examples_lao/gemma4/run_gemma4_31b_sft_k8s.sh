#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Launch Gemma4 31B SFT in a Kubernetes multi-node environment.
#
# Expected env vars injected by the platform:
#   MASTER_ADDR
#   MASTER_PORT
#   RANK
#   WORLD_SIZE
#
# Run this script inside every worker pod.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/llm-align/liuchonghan/Automodel_lao}
CONFIG=${CONFIG:-examples_lao/gemma4/gemma4_31b_sft.yaml}
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.venv}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export TMPDIR=${TMPDIR:-/llm-align/liuchonghan/tmp}
export TEMP=${TEMP:-/llm-align/liuchonghan/tmp}
export TMP=${TMP:-/llm-align/liuchonghan/tmp}
export HF_HOME=${HF_HOME:-/llm-align/liuchonghan/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/llm-align/liuchonghan/hf_cache/hub}
export TORCH_HOME=${TORCH_HOME:-/llm-align/liuchonghan/torch_cache}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/llm-align/liuchonghan/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/llm-align/liuchonghan/triton_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/llm-align/liuchonghan/xdg_cache}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DISABLED=${WANDB_DISABLED:-false}
export WANDB_DIR=${WANDB_DIR:-/llm-align/liuchonghan/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/llm-align/liuchonghan/wandb/cache}
export HF_TOKEN=${HF_TOKEN:-}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTHONFAULTHANDLER=1

# Avoid CUDA allocator fragmentation (fixes "reserved but unallocated" OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for name in MASTER_ADDR MASTER_PORT RANK WORLD_SIZE; do
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 1
    fi
done

if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual environment not found: ${VENV_DIR}" >&2
    exit 1
fi

if [ ! -f "${REPO_ROOT}/${CONFIG}" ] && [ ! -f "${CONFIG}" ]; then
    echo "Config file not found: ${CONFIG}" >&2
    exit 1
fi

mkdir -p \
    "${TMPDIR}" \
    "${HF_HOME}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    /llm-align/liuchonghan/ckpt_automodel

echo "HOSTNAME=$(hostname)"
echo "RANK=${RANK}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG}"

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"

torchrun \
    --nnodes="${WORLD_SIZE}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m nemo_automodel.cli.app \
    "${CONFIG}"
