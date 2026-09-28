#!/bin/bash
# Run all four Titans architectures on one official RULER S-NIAH context length.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-$(cd "$AUTOMODEL_CHECKOUT/../.." && pwd)}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
RULER_REPO=${RULER_REPO:-"$TITANS_WORK_ROOT/third_party/RULER"}
TOKENIZER=${TITANS_TOKENIZER:-NousResearch/Llama-2-7b-hf}
NUM_SAMPLES=${TITANS_RULER_NUM_SAMPLES:-100}
RULER_SEEDS=${TITANS_RULER_SEEDS:-"42 43 44"}
SAMPLE_START=${TITANS_RULER_SAMPLE_START:-0}
SAMPLE_END=${TITANS_RULER_SAMPLE_END:-$NUM_SAMPLES}
RULER_MODE=${TITANS_RULER_MODE:-all}
RULER_TASKS=${TITANS_RULER_TASKS:-"niah_single_1 niah_single_2"}
MAX_NEW_TOKENS=${TITANS_RULER_MAX_NEW_TOKENS:-128}
TTT_MODE=${TITANS_TTT_MODE:-both}
CONTEXT_LENGTH=${TITANS_RULER_CONTEXT_LENGTH:-${SLURM_JOB_NAME##*_}}
RESULT_NAMESPACE=${TITANS_RULER_RESULT_NAMESPACE:-ruler}
MODEL_REGISTRY=${TITANS_MODEL_REGISTRY:-"$AUTOMODEL_CHECKOUT/examples/llm_pretrain/titans_eval_registry.json"}
MODEL_NAME=${TITANS_MODEL_NAME:-}
SHARED_DATA_DIR="$TITANS_WORK_ROOT/results/eval/$RESULT_NAMESPACE/prepared-data"

if [[ ! $CONTEXT_LENGTH =~ ^[0-9]+$ ]]; then
  echo "Could not derive numeric context length from job name $SLURM_JOB_NAME" >&2
  exit 2
fi

if [[ ${TITANS_EVAL_IN_CONTAINER:-0} != 1 ]]; then
  exec srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$TITANS_IMAGE" \
    --container-mounts=/scratch:/scratch \
    --container-workdir="$AUTOMODEL_CHECKOUT" \
    env \
      TITANS_EVAL_IN_CONTAINER=1 \
      AUTOMODEL_CHECKOUT="$AUTOMODEL_CHECKOUT" \
      TITANS_WORK_ROOT="$TITANS_WORK_ROOT" \
      RULER_REPO="$RULER_REPO" \
      TITANS_TOKENIZER="$TOKENIZER" \
      TITANS_RULER_NUM_SAMPLES="$NUM_SAMPLES" \
      TITANS_RULER_SEEDS="$RULER_SEEDS" \
      TITANS_RULER_SAMPLE_START="$SAMPLE_START" \
      TITANS_RULER_SAMPLE_END="$SAMPLE_END" \
      TITANS_RULER_MODE="$RULER_MODE" \
      TITANS_RULER_TASKS="$RULER_TASKS" \
      TITANS_RULER_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      TITANS_TTT_MODE="$TTT_MODE" \
      TITANS_RULER_CONTEXT_LENGTH="$CONTEXT_LENGTH" \
      TITANS_RULER_RESULT_NAMESPACE="$RESULT_NAMESPACE" \
      TITANS_MODEL_REGISTRY="$MODEL_REGISTRY" \
      TITANS_MODEL_NAME="$MODEL_NAME" \
      PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" \
      HF_HOME="$TITANS_WORK_ROOT/cache/huggingface" \
      bash "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/titans_blackwell_ruler.sbatch"
fi

cd "$AUTOMODEL_CHECKOUT"
export PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-"$TITANS_WORK_ROOT/cache/huggingface"}
export NLTK_DATA=${NLTK_DATA:-"$TITANS_WORK_ROOT/cache/nltk"}
mkdir -p "$NLTK_DATA"
python -c "import bs4, html2text, nltk, tenacity, wonderwords" 2>/dev/null || \
  python -m pip install --quiet beautifulsoup4 html2text nltk tenacity wonderwords
python -m nltk.downloader -q -d "$NLTK_DATA" punkt punkt_tab

RULER_ASSET_DIR="$RULER_REPO/scripts/data/synthetic/json"
RULER_ESSAYS="$RULER_ASSET_DIR/PaulGrahamEssays.json"
if [[ ! -s "$RULER_ESSAYS" ]]; then
  (
    flock 9
    if [[ ! -s "$RULER_ESSAYS" ]]; then
      cd "$RULER_ASSET_DIR"
      python download_paulgraham_essay.py
      test -s "$RULER_ESSAYS"
    fi
  ) 9>"$RULER_ASSET_DIR/.PaulGrahamEssays.lock"
fi

read -r -a task_args <<<"$RULER_TASKS"
case "$TTT_MODE" in both|on|off) ;; *) echo "TITANS_TTT_MODE must be both, on, or off" >&2; exit 2 ;; esac
model_args=()
[[ -n $MODEL_NAME ]] && model_args=(--model "$MODEL_NAME")
mapfile -t model_rows < <(
  python examples/llm_pretrain/titans_eval_registry.py \
    "$MODEL_REGISTRY" --work-root "$TITANS_WORK_ROOT" "${model_args[@]}"
)
(( ${#model_rows[@]} )) || { echo "Model registry has no enabled checkpoints" >&2; exit 1; }
read -r -a seeds <<<"$RULER_SEEDS"
(( ${#seeds[@]} == 3 )) || { echo "TITANS_RULER_SEEDS must contain exactly three seeds" >&2; exit 2; }
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
(( GPU_COUNT > 0 )) || { echo "No evaluation GPUs are visible" >&2; exit 1; }

for seed in "${seeds[@]}"; do
  [[ $seed =~ ^[0-9]+$ ]] || { echo "invalid RULER seed: $seed" >&2; exit 2; }
  pids=()
  gpu=0
  for row in "${model_rows[@]}"; do
    IFS=$'\t' read -r architecture checkpoint enable_ttt <<<"$row"
    [[ $TTT_MODE == both || ( $TTT_MODE == on && $enable_ttt == 1 ) || \
       ( $TTT_MODE == off && $enable_ttt == 0 ) ]] || continue
    ttt_label=off
    ttt_args=(--disable-ttt-updates)
    if [[ $enable_ttt == 1 ]]; then
      ttt_label=on
      ttt_args=()
    fi
    result_root="$TITANS_WORK_ROOT/results/eval/$RESULT_NAMESPACE/$architecture/ttt-$ttt_label"
    log_namespace=${RESULT_NAMESPACE//\//_}
    log="$TITANS_WORK_ROOT/logs/titans-ruler-${log_namespace}-${architecture}-${ttt_label}-${CONTEXT_LENGTH}-seed${seed}.log"

    CUDA_VISIBLE_DEVICES=$gpu python examples/llm_pretrain/titans_ruler.py \
      --ruler-repo "$RULER_REPO" \
      --checkpoint "$checkpoint" \
      --tokenizer "$TOKENIZER" \
      --output-dir "$result_root" \
      --data-dir "$SHARED_DATA_DIR" \
      --context-lengths "$CONTEXT_LENGTH" \
      --tasks "${task_args[@]}" \
      --num-samples "$NUM_SAMPLES" \
      --seed "$seed" \
      --sample-start "$SAMPLE_START" \
      --sample-end "$SAMPLE_END" \
      --mode "$RULER_MODE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      "${ttt_args[@]}" \
      >"$log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
    if (( gpu == GPU_COUNT )); then
      failed=0
      for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
      (( failed == 0 )) || { echo "RULER batch failed; inspect $TITANS_WORK_ROOT/logs" >&2; exit 1; }
      pids=()
      gpu=0
    fi
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "RULER batch failed; inspect $TITANS_WORK_ROOT/logs" >&2; exit 1; }
done

echo "Completed official RULER S-NIAH at context length $CONTEXT_LENGTH."
