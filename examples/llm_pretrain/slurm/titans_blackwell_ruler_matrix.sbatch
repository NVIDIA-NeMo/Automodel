#!/bin/bash
# Evaluate one model/TTT-mode/seed across the matched 512--4096 RULER matrix.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:?set TITANS_WORK_ROOT}
TITANS_MODEL=${TITANS_MODEL:?set TITANS_MODEL}
TITANS_TTT_MODE=${TITANS_TTT_MODE:?set TITANS_TTT_MODE to on or off}
TITANS_SEED=${TITANS_SEED:?set TITANS_SEED}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
MODEL_REGISTRY=${TITANS_MODEL_REGISTRY:-"$AUTOMODEL_CHECKOUT/examples/llm_pretrain/titans_eval_registry.json"}
RULER_REPO=${RULER_REPO:-"$TITANS_WORK_ROOT/third_party/RULER"}
RESULT_ROOT=${RESULT_ROOT:-"$TITANS_WORK_ROOT/results/eval/matched-controls-v1"}
DATA_ROOT=${DATA_ROOT:-"$RESULT_ROOT/prepared-data"}
TOKENIZER=${TITANS_TOKENIZER:-NousResearch/Llama-2-7b-hf}

[[ $TITANS_TTT_MODE == on || $TITANS_TTT_MODE == off ]] || {
  echo "TITANS_TTT_MODE must be on or off" >&2
  exit 2
}

srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="$TITANS_IMAGE" \
  --container-mounts=/scratch:/scratch \
  --container-workdir="$AUTOMODEL_CHECKOUT" \
  --export=ALL \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH="$AUTOMODEL_CHECKOUT"
    export HF_HOME="$TITANS_WORK_ROOT/cache/huggingface"
    enable_ttt=$([[ "$TITANS_TTT_MODE" == on ]] && echo 1 || echo 0)
    checkpoint=$(
      python examples/llm_pretrain/titans_eval_registry.py \
        "$MODEL_REGISTRY" --work-root "$TITANS_WORK_ROOT" --model "$TITANS_MODEL" |
      awk -F "\t" -v expected="$enable_ttt" "\$3 == expected {print \$2; exit}"
    )
    [[ -n $checkpoint ]] || {
      echo "Could not resolve $TITANS_MODEL / TTT $TITANS_TTT_MODE" >&2
      exit 1
    }

    mkdir -p "$TITANS_WORK_ROOT/logs"
    pids=()
    gpu=0
    for length in 512 1024 2048 4096; do
      tasks=(niah_single_1)
      [[ $length == 512 ]] || tasks+=(niah_single_2)
      disable=()
      [[ $TITANS_TTT_MODE == on ]] || disable=(--disable-ttt-updates)
      log="$TITANS_WORK_ROOT/logs/titans-ruler-matched-controls-v1-${TITANS_MODEL}-${TITANS_TTT_MODE}-${length}-seed${TITANS_SEED}.log"
      (
        export CUDA_VISIBLE_DEVICES=$gpu
        python examples/llm_pretrain/titans_ruler.py \
          --ruler-repo "$RULER_REPO" \
          --checkpoint "$checkpoint" \
          --tokenizer "$TOKENIZER" \
          --output-dir "$RESULT_ROOT/$TITANS_MODEL/ttt-$TITANS_TTT_MODE" \
          --data-dir "$DATA_ROOT" \
          --context-lengths "$length" \
          --tasks "${tasks[@]}" \
          --num-samples 100 \
          --seed "$TITANS_SEED" \
          --max-new-tokens 128 \
          --model-template-type base \
          --mode all \
          "${disable[@]}"
      ) >"$log" 2>&1 &
      pids+=($!)
      gpu=$((gpu + 1))
    done

    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    exit "$status"
  '
