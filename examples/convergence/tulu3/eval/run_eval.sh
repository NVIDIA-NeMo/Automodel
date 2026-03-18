#!/bin/bash
# Run lm-evaluation-harness with vLLM backend on a finetuned checkpoint.
#
# Usage:
#   bash run_eval.sh --model-path /path/to/checkpoint [OPTIONS] [-- EXTRA_LM_EVAL_ARGS]
#
# Required:
#   --model-path PATH     Path to the model checkpoint directory
#
# Options:
#   --tokenizer NAME      HF tokenizer name (default: Qwen/Qwen3-30B-A3B)
#   --tasks TASKS         Comma-separated eval tasks (default: ifeval)
#   --output-path PATH    Where to write results (default: auto-derived from model path)
#   --thinking            Enable thinking mode for chat template
#   --tp-size N           Tensor parallel size (default: 2)
#   --dp-size N           Data parallel size (default: 4)
#
# Everything after "--" is forwarded to lm_eval verbatim.

set -euo pipefail

LM_EVAL_DIR="${LM_EVAL_DIR:-/opt/lm-evaluation-harness}"
ENABLE_THINKING="False"
TP_SIZE=2
DP_SIZE=4
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)   MODEL_PATH="$2";   shift 2 ;;
    --tokenizer)    TOKENIZER="$2";    shift 2 ;;
    --tasks)        TASKS="$2";        shift 2 ;;
    --output-path)  OUTPUT_PATH="$2";  shift 2 ;;
    --thinking)     ENABLE_THINKING="True"; shift ;;
    --tp-size)      TP_SIZE="$2";      shift 2 ;;
    --dp-size)      DP_SIZE="$2";      shift 2 ;;
    --)             shift; EXTRA_ARGS+=("$@"); break ;;
    *)              echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "${MODEL_PATH:-}" ]; then
  echo "ERROR: --model-path is required"
  exit 1
fi

: "${TOKENIZER:=Qwen/Qwen3-30B-A3B}"
: "${TASKS:=ifeval}"
: "${OUTPUT_PATH:=results/$(echo "$MODEL_PATH" | rev | cut -d'/' -f1-4 | rev)}"

# ── Patch config.json for vLLM rope_theta compatibility ─────────────────────
CONFIG="$MODEL_PATH/config.json"
if [ -f "$CONFIG" ]; then
  python3 -c "
import json
with open('$CONFIG') as f:
    cfg = json.load(f)
if 'rope_parameters' in cfg and 'rope_theta' in cfg['rope_parameters']:
    cfg['rope_theta'] = cfg['rope_parameters']['rope_theta']
    with open('$CONFIG', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'Patched rope_theta={cfg[\"rope_theta\"]} in $CONFIG')
else:
    print('No rope_parameters.rope_theta found, skipping patch')
"; else
  echo "WARNING: $CONFIG not found, skipping rope_theta patch"
fi

# ── Activate lm_eval venv ───────────────────────────────────────────────────
VENV_DIR="$LM_EVAL_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: lm_eval venv not found at $VENV_DIR"
  echo "Run setup_lm_eval.sh first."
  exit 1
fi
source "$VENV_DIR/bin/activate"

# ── Run evaluation ──────────────────────────────────────────────────────────
cd "$LM_EVAL_DIR"

lm_eval --model vllm \
    --model_args "pretrained=$MODEL_PATH,tokenizer=$TOKENIZER,tensor_parallel_size=$TP_SIZE,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=$DP_SIZE,enable_thinking=$ENABLE_THINKING" \
    --tasks "$TASKS" \
    --batch_size auto \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
