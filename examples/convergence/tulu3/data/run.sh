#!/usr/bin/env bash
# Stage 1: Data pre-filtering and validation.
#
# Runs truncation analysis, pre-filters the dataset, and validates
# token-level correctness. Exits non-zero if validation fails.
#
# Usage:
#   bash examples/convergence/tulu3/data/run.sh
#   DATASET=my/dataset SEQ_LENGTH=2048 bash examples/convergence/tulu3/data/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/venv/bin/activate

DATASET="${DATASET:-allenai/tulu-3-sft-mixture}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
SPLIT="${SPLIT:-train}"
NUM_VALIDATE="${NUM_VALIDATE:-500}"

echo "=== Stage 1: Data Pre-filtering and Validation ==="
echo "  Dataset:    ${DATASET}"
echo "  Model:      ${MODEL}"
echo "  Seq length: ${SEQ_LENGTH}"
echo ""

# Step 1: Truncation analysis
echo "--- Step 1: Truncation analysis ---"
python "${SCRIPT_DIR}/check_truncation.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seq_length ${SEQ_LENGTH} 2048 4096 8192 \
    --split "${SPLIT}" \
    --num_samples 1000

echo ""

# Step 2: Pre-filter and cache
echo "--- Step 2: Pre-filter and cache as Parquet ---"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cached}"
python "${SCRIPT_DIR}/prefilter_dataset.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seq_length "${SEQ_LENGTH}" \
    --split "${SPLIT}" \
    --strategies text_only \
    --cache_dir "${CACHE_DIR}"

echo ""

# Step 3: Validate data correctness
echo "--- Step 3: Token-level validation ---"
python "${SCRIPT_DIR}/validate_data.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seq_length "${SEQ_LENGTH}" \
    --split "${SPLIT}" \
    --num-samples "${NUM_VALIDATE}"

echo ""
echo "=== Stage 1 complete ==="
