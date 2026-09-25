#!/bin/bash
# Run one resumable standard-LM evaluation range, or merge completed ranges.
#
# Examples:
#   BENCHMARK=wikitext103 START=0 END=4358 CHECKPOINT=/path/to/consolidated sbatch "$0"
#   BENCHMARK=lambada START=0 END=5153 CHECKPOINT=/path/to/consolidated sbatch "$0"
#   BENCHMARK=fineweb_edu START=0 END=1000 CHECKPOINT=/path/to/consolidated sbatch "$0"
#   MODE=merge BENCHMARK=wikitext103 sbatch "$0"
# Set INPUT_FILE to a held-out JSONL/text fixture to avoid any dataset download.

#SBATCH --job-name=titans-lm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-$(cd "$AUTOMODEL_CHECKOUT/../.." && pwd)}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
TOKENIZER=${TITANS_TOKENIZER:-NousResearch/Llama-2-7b-hf}
MODE=${MODE:-evaluate}
BENCHMARK=${BENCHMARK:?Set BENCHMARK to wikitext103, lambada, or fineweb_edu}
RESULT_ROOT=${RESULT_ROOT:-"$TITANS_WORK_ROOT/results/eval/lm"}

if [[ ${TITANS_EVAL_IN_CONTAINER:-0} != 1 ]]; then
  mounts=/scratch:/scratch
  if [[ -n ${INPUT_FILE:-} ]]; then
    mounts="$mounts,$(dirname "$INPUT_FILE"):$(dirname "$INPUT_FILE")"
  fi
  exec srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$TITANS_IMAGE" \
    --container-mounts="$mounts" \
    --container-workdir="$AUTOMODEL_CHECKOUT" \
    env \
      TITANS_EVAL_IN_CONTAINER=1 \
      AUTOMODEL_CHECKOUT="$AUTOMODEL_CHECKOUT" \
      TITANS_WORK_ROOT="$TITANS_WORK_ROOT" \
      TITANS_TOKENIZER="$TOKENIZER" \
      MODE="$MODE" \
      BENCHMARK="$BENCHMARK" \
      RESULT_ROOT="$RESULT_ROOT" \
      CHECKPOINT="${CHECKPOINT:-}" \
      START="${START:-0}" \
      END="${END:-}" \
      MAX_TOKENS="${MAX_TOKENS:-}" \
      INPUT_FILE="${INPUT_FILE:-}" \
      INPUT_BIN="${INPUT_BIN:-}" \
      ENABLE_TTT_UPDATES="${ENABLE_TTT_UPDATES:-1}" \
      TEXT_FIELD="${TEXT_FIELD:-text}" \
      PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" \
      HF_HOME="$TITANS_WORK_ROOT/cache/huggingface" \
      bash "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/titans_blackwell_lm_eval.sbatch"
fi

cd "$AUTOMODEL_CHECKOUT"
export PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-"$TITANS_WORK_ROOT/cache/huggingface"}
mkdir -p "$RESULT_ROOT"

if [[ "$MODE" == merge ]]; then
  python examples/llm_pretrain/titans_lm_eval.py merge \
    --benchmark "$BENCHMARK" \
    --output-dir "$RESULT_ROOT"
  exit 0
fi

: "${CHECKPOINT:?Set CHECKPOINT to a consolidated safetensors directory}"
args=(
  evaluate
  --benchmark "$BENCHMARK"
  --checkpoint "$CHECKPOINT"
  --tokenizer "$TOKENIZER"
  --output-dir "$RESULT_ROOT"
  --start "${START:-0}"
)
if [[ -n ${END:-} ]]; then
  args+=(--end "$END")
fi
if [[ -n ${MAX_TOKENS:-} ]]; then
  args+=(--max-tokens "$MAX_TOKENS")
fi
if [[ -n ${INPUT_FILE:-} ]]; then
  args+=(--input-file "$INPUT_FILE" --text-field "${TEXT_FIELD:-text}")
fi
if [[ -n ${INPUT_BIN:-} ]]; then
  args+=(--input-bin "$INPUT_BIN")
fi
if [[ ${ENABLE_TTT_UPDATES:-1} != 1 ]]; then
  args+=(--disable-ttt-updates)
fi

python examples/llm_pretrain/titans_lm_eval.py "${args[@]}"
