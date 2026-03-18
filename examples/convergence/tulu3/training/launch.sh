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

# Tulu-3 convergence training launch script.
#
# Usage:
#   bash examples/convergence/tulu3/training/launch.sh \
#       --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml \
#       --nproc 8 \
#       --wandb-project my-project \
#       --wandb-entity my-entity \
#       --wandb-name my-run

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=0
export NVTE_FLASH_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG=""
NPROC=8
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --nproc)
            NPROC="$2"; shift 2 ;;
        --wandb-project)
            WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity)
            WANDB_ENTITY="$2"; shift 2 ;;
        --wandb-name)
            WANDB_NAME="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required." >&2
    exit 1
fi

if [[ -z "${HF_HOME:-}" ]]; then
    echo "Error: HF_HOME must be set." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build W&B arguments
# ---------------------------------------------------------------------------
WANDB_ARGS=""
if [[ -n "$WANDB_PROJECT" ]]; then
    WANDB_ARGS+=" --wandb.project ${WANDB_PROJECT}"
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=" --wandb.entity ${WANDB_ENTITY}"
fi
if [[ -n "$WANDB_NAME" ]]; then
    WANDB_ARGS+=" --wandb.name ${WANDB_NAME}"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
torchrun --nproc-per-node "$NPROC" --tee 3 \
    examples/llm_finetune/finetune.py \
    --config "$CONFIG" \
    $WANDB_ARGS \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
