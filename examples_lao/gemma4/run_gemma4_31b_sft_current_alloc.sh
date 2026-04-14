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

# Launch Gemma4 31B SFT inside an already-allocated multi-node Slurm job.
#
# Usage:
#   bash examples_lao/gemma4/run_gemma4_31b_sft_current_alloc.sh
#
# Requirements:
# - You already have a Slurm allocation covering all training nodes.
# - The repo path is visible from every node.

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/llm-align/liuchonghan/Automodel_lao}
CONFIG=${CONFIG:-examples_lao/gemma4/gemma4_31b_sft.yaml}
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.venv}

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

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
    echo "SLURM_JOB_NODELIST is not set. Run this inside an existing Slurm allocation." >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual environment not found: ${VENV_DIR}" >&2
    exit 1
fi

if [ ! -f "${REPO_ROOT}/${CONFIG}" ] && [ ! -f "${CONFIG}" ]; then
    echo "Config file not found: ${CONFIG}" >&2
    exit 1
fi

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG}"

mkdir -p \
    "${TMPDIR}" \
    "${HF_HOME}" \
    "${TRANSFORMERS_CACHE}" \
    "${TORCH_HOME}" \
    "${TORCH_EXTENSIONS_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}"

cd "${REPO_ROOT}"

srun bash -lc "\
    source ${VENV_DIR}/bin/activate && \
    cd ${REPO_ROOT} && \
    mkdir -p \
        ${TMPDIR} \
        ${HF_HOME} \
        ${TRANSFORMERS_CACHE} \
        ${TORCH_HOME} \
        ${TORCH_EXTENSIONS_DIR} \
        ${TRITON_CACHE_DIR} \
        ${XDG_CACHE_HOME} \
        ${WANDB_DIR} \
        ${WANDB_CACHE_DIR} && \
    torchrun \
        --nproc-per-node=\${SLURM_GPUS_PER_NODE:-8} \
        --nnodes=\${SLURM_NNODES:-8} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
        -m nemo_automodel.cli.app ${CONFIG}"
