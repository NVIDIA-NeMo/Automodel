#!/usr/bin/env bash
# Stage 2: Model verification — layer-by-layer activation comparison
# between NeMo AutoModel and HF Transformers.
#
# Compares per-layer hidden states (cosine similarity) and final logits
# to verify our implementation matches the reference HF model.
#
# Usage:
#   bash examples/convergence/tulu3/model-verification/run.sh
#   CONFIG=path/to/config.yaml NPROC=4 bash examples/convergence/tulu3/model-verification/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/opt/venv/bin/python}"

CONFIG="${CONFIG:-examples/llm_finetune/qwen/qwen3_moe_30b_te_chat_thd.yaml}"
NPROC="${NPROC:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
THRESHOLD="${THRESHOLD:-0.99}"
GATE_PRECISION="${GATE_PRECISION:-fp32}"
LM_HEAD_PRECISION="${LM_HEAD_PRECISION:-fp32}"

export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export NVTE_UNFUSED_ATTN="${NVTE_UNFUSED_ATTN:-0}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Stage 2: Model Verification (Layer-by-Layer) ==="
echo "  Config:      ${CONFIG}"
echo "  GPUs:        ${NPROC}"
echo "  Prompts:     ${NUM_PROMPTS}"
echo "  Threshold:   ${THRESHOLD} (cosine similarity)"
echo "  Precision:   gate=${GATE_PRECISION}, lm_head=${LM_HEAD_PRECISION}"
echo ""

source /opt/venv/bin/activate

python "${SCRIPT_DIR}/compare_activations.py" \
    --config "${CONFIG}" \
    --nproc "${NPROC}" \
    --num-prompts "${NUM_PROMPTS}" \
    --threshold "${THRESHOLD}" \
    --gate-precision "${GATE_PRECISION}" \
    --lm-head-precision "${LM_HEAD_PRECISION}" \
    --model.backend.rope_fusion False

echo ""
echo "=== Stage 2 complete ==="
