#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Submit a Slurm CPU job to build/warm Hugging Face dataset caches for a
# retrieval training config. This should be run before GPU training when
# datasets.load_dataset(...) startup is expensive.
#
# Required:
#   CONFIG=/path/to/original_retrieval_config.yaml
#
# Common overrides:
#   CACHE_DIR=/path/to/shared/hf_cache
#   PARTITION=cpu_short
#   TIME=02:00:00
#   CPUS_PER_TASK=32
#   TOUCH_SAMPLES=128
#   EXTRA_CONTAINER_MOUNTS=/source/data:/source/data,/models:/models

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)}"

CONFIG="${CONFIG:-}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-cpu_short}"
TIME="${TIME:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
RUN_NAME="${RUN_NAME:-warm-retrieval-hf-cache}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.06.00}"
USE_CONTAINER="${USE_CONTAINER:-1}"
EXTRA_CONTAINER_MOUNTS="${EXTRA_CONTAINER_MOUNTS:-}"

CACHE_DIR="${CACHE_DIR:-${PWD}/hf_cache}"
LOG_DIR="${LOG_DIR:-${CACHE_DIR}/slurm_logs}"
TOUCH_SAMPLES="${TOUCH_SAMPLES:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
EXTRA_WARM_ARGS="${EXTRA_WARM_ARGS:-}"

if [[ -z "${CONFIG}" ]]; then
    cat >&2 <<EOF
Usage:
  CONFIG=/path/to/original_config.yaml CACHE_DIR=/path/to/shared/hf_cache $0

Example:
  CONFIG=/path/to/eagle_llama_1b_gmoreira_8_nodes_image.yaml \\
  CACHE_DIR=/path/to/shared/hf_cache \\
  PARTITION=cpu_short TIME=02:00:00 CPUS_PER_TASK=32 TOUCH_SAMPLES=128 \\
  EXTRA_CONTAINER_MOUNTS=/path/to/source_data:/path/to/source_data \\
  $0
EOF
    exit 2
fi

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "AutoModel repo not found: ${REPO_DIR}" >&2
    exit 3
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 4
fi

mkdir -p "${CACHE_DIR}" "${LOG_DIR}"

CONFIG_DIR="$(cd -- "$(dirname -- "${CONFIG}")" && pwd)"
CACHE_PARENT="$(mkdir -p "${CACHE_DIR}" && cd -- "$(dirname -- "${CACHE_DIR}")" && pwd)"

CONTAINER_MOUNTS="${REPO_DIR}:/opt/Automodel,${CONFIG_DIR}:${CONFIG_DIR},${CACHE_PARENT}:${CACHE_PARENT}"
if [[ -n "${EXTRA_CONTAINER_MOUNTS}" ]]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${EXTRA_CONTAINER_MOUNTS}"
fi

max_train_args=()
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    max_train_args=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi

account_args=()
if [[ -n "${ACCOUNT}" ]]; then
    account_args=(--account="${ACCOUNT}")
fi

exclude_args=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
    exclude_args=(--exclude="${EXCLUDE_NODES}")
fi

REPORT_PATH="${CACHE_DIR}/warm-retrieval-hf-cache-${RUN_NAME}.json"
METADATA_FILE="${CACHE_DIR}/warm-retrieval-hf-cache-${RUN_NAME}.yaml"
cat > "${METADATA_FILE}" <<EOF
run_name: ${RUN_NAME}
repo_dir: ${REPO_DIR}
config: ${CONFIG}
cache_dir: ${CACHE_DIR}
partition: ${PARTITION}
time: ${TIME}
cpus_per_task: ${CPUS_PER_TASK}
touch_samples: ${TOUCH_SAMPLES}
max_train_samples: ${MAX_TRAIN_SAMPLES}
container_image: ${CONTAINER_IMAGE}
extra_container_mounts: ${EXTRA_CONTAINER_MOUNTS}
report_path: ${REPORT_PATH}
EOF

sbatch \
    "${account_args[@]}" \
    --partition="${PARTITION}" \
    --time="${TIME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    "${exclude_args[@]}" \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" <<EOF
#!/bin/bash
set -euo pipefail

echo "Running on host: \$(hostname)"
echo "Config: ${CONFIG}"
echo "Cache dir: ${CACHE_DIR}"
echo "SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-unset}"
ln -sfn "${LOG_DIR}/${RUN_NAME}-\${SLURM_JOB_ID}.out" "${CACHE_DIR}/warm-retrieval-hf-cache-latest.out"
ln -sfn "${LOG_DIR}/${RUN_NAME}-\${SLURM_JOB_ID}.err" "${CACHE_DIR}/warm-retrieval-hf-cache-latest.err"

srun_args=()
if [[ "${USE_CONTAINER}" == "1" ]]; then
    srun_args+=(
        --container-image="${CONTAINER_IMAGE}"
        --container-mounts="${CONTAINER_MOUNTS}"
        --container-entrypoint
        --no-container-mount-home
    )
fi

srun "\${srun_args[@]}" bash -lc 'set -euo pipefail
    if [[ "${USE_CONTAINER}" == "1" ]]; then
        cd /opt/Automodel
    else
        cd "${REPO_DIR}"
    fi
    export PYTHONPATH="\$(pwd):\${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    python --version
    python tools/retrieval/warm_retrieval_hf_cache.py \
        --config "${CONFIG}" \
        --cache-dir "${CACHE_DIR}" \
        --touch-samples "${TOUCH_SAMPLES}" \
        --report-path "${REPORT_PATH}" \
        ${max_train_args[*]} \
        ${EXTRA_WARM_ARGS}
'
EOF
