#!/usr/bin/env bash
set -euo pipefail

# Run from the AutoModel repo root on a node with enough CPU RAM.
# The merge itself is CPU/safetensors based; a GPU node is useful here mainly
# because those nodes usually have much more host memory than login nodes.

PYTHON="${PYTHON:-.venv/bin/python}"
MAX_SHARD_SIZE_GB="${MAX_SHARD_SIZE_GB:-5}"
MAX_SHARD_SIZE_GIB_FOR_TOOL="$("${PYTHON}" -c 'import sys; print(float(sys.argv[1]) * (1000**3) / (1024**3))' "${MAX_SHARD_SIZE_GB}")"
KARCHER_MAX_ITERATIONS="${KARCHER_MAX_ITERATIONS:-10}"
KARCHER_TOLERANCE="${KARCHER_TOLERANCE:-1e-5}"
STAMP="${STAMP:-$(date -u +%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v13_${STAMP}}"

CHECKPOINT_1="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_slim_v5_07-11-183840/LATEST/model/consolidated"
CHECKPOINT_2="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_final_from_base_code_v2_07-11-222130/LATEST/model/consolidated"
CHECKPOINT_3="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_final_FT_from_17M_v2_07-11-221426/LATEST/model/consolidated"
CHECKPOINT_4="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_6_HNs_FT_17M_07-10-230406/LATEST/model/consolidated"

# CHECKPOINT_1="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_slim_v5_07-11-183840/LATEST/model/consolidated"
# CHECKPOINT_2="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_final_from_base_code_v2_07-11-222130/LATEST/model/consolidated"
# CHECKPOINT_3="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_final_FT_from_17M_v2_07-11-221426/LATEST/model/consolidated"
# CHECKPOINT_4="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated"
# CHECKPOINT_5="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_07-03-195935/LATEST/model/consolidated"
# CHECKPOINT_6="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_FT_from_17M_07-07-143433/LATEST/model/consolidated"
# CHECKPOINT_7="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_6_HNs_FT_17M_07-10-230406/LATEST/model/consolidated"
# CHECKPOINT_8="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_slim_v3_07-10-223809/LATEST/model/consolidated"


export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

echo "Writing ensemble to: ${OUTPUT_DIR}"
echo "Max shard size: ${MAX_SHARD_SIZE_GB} GB decimal, matching Transformers' default"
echo "Passing ${MAX_SHARD_SIZE_GIB_FOR_TOOL} GiB to the merge tool"

"${PYTHON}" scripts/yauhen/multislerp_merge_checkpoints.py \
  --layout experiment \
  --merge-method karcher \
  --copy-non-floating \
  --max-shard-size-gb "${MAX_SHARD_SIZE_GIB_FOR_TOOL}" \
  --karcher-max-iterations "${KARCHER_MAX_ITERATIONS}" \
  --karcher-tolerance "${KARCHER_TOLERANCE}" \
  --checkpoint "${CHECKPOINT_1}" \
  --checkpoint "${CHECKPOINT_2}" \
  --checkpoint "${CHECKPOINT_3}" \
  --checkpoint "${CHECKPOINT_4}" \
  --weight 1 \
  --weight 1 \
  --weight 1 \
  --weight 1 \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

if [[ -d "${OUTPUT_DIR}" ]]; then
  chmod -R a+rX "${OUTPUT_DIR}"
  echo "Readable model path: ${OUTPUT_DIR}/LATEST/model/consolidated"
else
  echo "No output directory found at ${OUTPUT_DIR}; this is expected for --dry-run."
fi
