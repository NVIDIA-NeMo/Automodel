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

# Submit a Slurm CPU array to build resolved VL retrieval data.
#
# Required:
#   CONFIG=/path/to/original_retrieval_config.yaml
#   OUT_DIR=/path/to/resolved_vl_retrieval
#
# Common overrides:
#   NUM_BUILD_SHARDS=32
#   ARRAY_SPEC=0-31
#   PARTITION=cpu_short
#   TIME=04:00:00
#   CPUS_PER_TASK=32
#   EXTRA_CONTAINER_MOUNTS=/source/data:/source/data,/models:/models

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)}"

CONFIG="${CONFIG:-}"
OUT_DIR="${OUT_DIR:-}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-cpu_short}"
TIME="${TIME:-04:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
NUM_BUILD_SHARDS="${NUM_BUILD_SHARDS:-32}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
RUN_NAME="${RUN_NAME:-resolved-vl-retrieval}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/slurm_logs}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidian/nemo-automodel:26.04.rc10}"
USE_CONTAINER="${USE_CONTAINER:-1}"
EXTRA_CONTAINER_MOUNTS="${EXTRA_CONTAINER_MOUNTS:-}"

CACHE_DIR="${CACHE_DIR:-${OUT_DIR}/../resolved_vl_retrieval_cache}"
HF_CACHE="${HF_CACHE:-${CACHE_DIR}/hf}"
TRITON_CACHE="${TRITON_CACHE:-${CACHE_DIR}/triton}"

MAX_SAMPLES="${MAX_SAMPLES:-}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-10000}"
SEED="${SEED:-42}"
JPEG_QUALITY="${JPEG_QUALITY:-90}"
USE_DATASET_INSTRUCTION="${USE_DATASET_INSTRUCTION:-0}"
EXTRA_PREP_ARGS="${EXTRA_PREP_ARGS:-}"

if [[ -z "${CONFIG}" || -z "${OUT_DIR}" ]]; then
    cat >&2 <<EOF
Usage:
  CONFIG=/path/to/config.yaml OUT_DIR=/path/to/resolved_data $0

Example:
  CONFIG=/path/to/eagle_llama_1b_gmoreira_8_nodes_image.yaml \\
  OUT_DIR=/path/to/resolved_vl_retrieval_full \\
  PARTITION=cpu_short TIME=04:00:00 CPUS_PER_TASK=32 \\
  NUM_BUILD_SHARDS=32 ARRAY_SPEC=0-31 \\
  EXTRA_CONTAINER_MOUNTS=/path/to/source_data:/path/to/source_data \\
  $0
EOF
    exit 2
fi

if [[ "${NUM_BUILD_SHARDS}" -lt 1 ]]; then
    echo "NUM_BUILD_SHARDS must be >= 1, got ${NUM_BUILD_SHARDS}" >&2
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

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${HF_CACHE}" "${TRITON_CACHE}"

array_last=$((NUM_BUILD_SHARDS - 1))
if [[ -z "${ARRAY_SPEC}" ]]; then
    ARRAY_SPEC="0-${array_last}"
fi

CONFIG_DIR="$(cd -- "$(dirname -- "${CONFIG}")" && pwd)"
OUT_PARENT="$(mkdir -p "${OUT_DIR}" && cd -- "$(dirname -- "${OUT_DIR}")" && pwd)"
CACHE_PARENT="$(mkdir -p "${CACHE_DIR}" && cd -- "$(dirname -- "${CACHE_DIR}")" && pwd)"

CONTAINER_MOUNTS="${REPO_DIR}:/opt/Automodel,${CONFIG_DIR}:${CONFIG_DIR},${OUT_PARENT}:${OUT_PARENT},${CACHE_PARENT}:${CACHE_PARENT}"
if [[ -n "${EXTRA_CONTAINER_MOUNTS}" ]]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${EXTRA_CONTAINER_MOUNTS}"
fi

max_samples_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then
    max_samples_args=(--max-samples "${MAX_SAMPLES}")
fi

instruction_args=()
if [[ "${USE_DATASET_INSTRUCTION}" == "1" ]]; then
    instruction_args=(--use-dataset-instruction)
fi

account_args=()
if [[ -n "${ACCOUNT}" ]]; then
    account_args=(--account="${ACCOUNT}")
fi

exclude_args=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
    exclude_args=(--exclude="${EXCLUDE_NODES}")
fi

METADATA_FILE="${OUT_DIR}/submission-metadata-${RUN_NAME}.yaml"
cat > "${METADATA_FILE}" <<EOF
run_name: ${RUN_NAME}
repo_dir: ${REPO_DIR}
config: ${CONFIG}
partition: ${PARTITION}
time: ${TIME}
cpus_per_task: ${CPUS_PER_TASK}
num_build_shards: ${NUM_BUILD_SHARDS}
array_spec: ${ARRAY_SPEC}
max_samples: ${MAX_SAMPLES}
samples_per_shard: ${SAMPLES_PER_SHARD}
seed: ${SEED}
jpeg_quality: ${JPEG_QUALITY}
output_dir: ${OUT_DIR}
container_image: ${CONTAINER_IMAGE}
extra_container_mounts: ${EXTRA_CONTAINER_MOUNTS}
EOF
if [[ ! -e "${OUT_DIR}/submission-metadata.yaml" ]]; then
    ln -s "$(basename "${METADATA_FILE}")" "${OUT_DIR}/submission-metadata.yaml"
fi

sbatch \
    "${account_args[@]}" \
    --partition="${PARTITION}" \
    --time="${TIME}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    "${exclude_args[@]}" \
    --array="${ARRAY_SPEC}" \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/%x-%A_%a.out" \
    --error="${LOG_DIR}/%x-%A_%a.err" <<EOF
#!/bin/bash
set -euo pipefail

echo "Running on host: \$(hostname)"
echo "Output dir: ${OUT_DIR}"
echo "Array task: \${SLURM_ARRAY_TASK_ID}"
echo "SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-unset}"
ln -sfn "${LOG_DIR}/${RUN_NAME}-\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.out" "${OUT_DIR}/slurm-\${SLURM_ARRAY_TASK_ID}.out"
ln -sfn "${LOG_DIR}/${RUN_NAME}-\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.err" "${OUT_DIR}/slurm-\${SLURM_ARRAY_TASK_ID}.err"

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
    export HF_HOME="${HF_CACHE}"
    export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
    export HF_DATASETS_CACHE="${HF_CACHE}"
    export TRANSFORMERS_CACHE="${HF_CACHE}"
    export TRITON_CACHE_DIR="${TRITON_CACHE}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    python --version
    python tools/retrieval/prepare_resolved_vl_retrieval_data.py \
        --config "${CONFIG}" \
        --output-dir "${OUT_DIR}" \
        --samples-per-shard "${SAMPLES_PER_SHARD}" \
        --seed "${SEED}" \
        --jpeg-quality "${JPEG_QUALITY}" \
        --num-build-shards "${NUM_BUILD_SHARDS}" \
        --build-shard-index "\${SLURM_ARRAY_TASK_ID}" \
        ${max_samples_args[*]} \
        ${instruction_args[*]} \
        ${EXTRA_PREP_ARGS}
'
EOF
