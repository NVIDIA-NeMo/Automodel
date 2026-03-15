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

# Tulu-3 convergence SLURM launch script.
#
# Generates and submits an sbatch job that invokes launch.sh internally.
#
# Usage:
#   bash examples/convergence/tulu3/training/launch_slurm.sh \
#       --config examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml \
#       --nodes 2 \
#       --partition batch \
#       --job-name tulu3-convergence \
#       --wandb-project my-project \
#       --wandb-entity my-entity \
#       --wandb-name my-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG=""
NPROC=8
NODES=1
PARTITION=""
JOB_NAME="tulu3-convergence"
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
        --nodes)
            NODES="$2"; shift 2 ;;
        --partition)
            PARTITION="$2"; shift 2 ;;
        --job-name)
            JOB_NAME="$2"; shift 2 ;;
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

if [[ -z "$PARTITION" ]]; then
    echo "Error: --partition is required." >&2
    exit 1
fi

if [[ -z "${HF_HOME:-}" ]]; then
    echo "Error: HF_HOME must be set." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build launch.sh arguments
# ---------------------------------------------------------------------------
LAUNCH_ARGS="--config ${CONFIG} --nproc ${NPROC}"
if [[ -n "$WANDB_PROJECT" ]]; then
    LAUNCH_ARGS+=" --wandb-project ${WANDB_PROJECT}"
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    LAUNCH_ARGS+=" --wandb-entity ${WANDB_ENTITY}"
fi
if [[ -n "$WANDB_NAME" ]]; then
    LAUNCH_ARGS+=" --wandb-name ${WANDB_NAME}"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    LAUNCH_ARGS+=" ${EXTRA_ARGS[*]}"
fi

# ---------------------------------------------------------------------------
# Build and submit sbatch script
# ---------------------------------------------------------------------------
LOG_DIR="$(pwd)/logs/${JOB_NAME}"
mkdir -p "$LOG_DIR"

SBATCH_SCRIPT=$(mktemp /tmp/tulu3_sbatch_XXXXXX.sh)

cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=${PARTITION}
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=${LOG_DIR}/%j.out
#SBATCH --error=${LOG_DIR}/%j.err

set -eux

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME}"

# Resolve head node for multi-node torchrun
nodes=( \$( scontrol show hostnames \$SLURM_JOB_NODELIST ) )
head_node=\${nodes[0]}
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)

srun --kill-on-bad-exit=1 \
    bash ${SCRIPT_DIR}/launch.sh ${LAUNCH_ARGS}
EOF

echo "Submitting SLURM job from: ${SBATCH_SCRIPT}"
echo "Logs will be written to: ${LOG_DIR}/"
sbatch "$SBATCH_SCRIPT"
