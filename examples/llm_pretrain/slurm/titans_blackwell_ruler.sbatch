#!/bin/bash
# Run all four Titans architectures on one official RULER S-NIAH context length.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-$(cd "$AUTOMODEL_CHECKOUT/../.." && pwd)}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
RULER_REPO=${RULER_REPO:-"$TITANS_WORK_ROOT/third_party/RULER"}
TOKENIZER=${TITANS_TOKENIZER:-NousResearch/Llama-2-7b-hf}
NUM_SAMPLES=${TITANS_RULER_NUM_SAMPLES:-500}
CONTEXT_LENGTH=${TITANS_RULER_CONTEXT_LENGTH:-${SLURM_JOB_NAME##*_}}

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
      TITANS_RULER_CONTEXT_LENGTH="$CONTEXT_LENGTH" \
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

declare -A RUNS=(
  [lmm]=titans-170m-baseline-15B-blackwell-optimized
  [mac]=titans-170m-mac-15B-blackwell-mac-v1
  [mag]=titans-170m-mag-15B-blackwell-mag-v1
  [mal]=titans-170m-mal-15B-blackwell-mal-v1
)

pids=()
gpu=0
for architecture in lmm mac mag mal; do
  root="$TITANS_WORK_ROOT/outputs/${RUNS[$architecture]}"
  latest=$(readlink "$root/LATEST")
  checkpoint="$root/$latest/model/consolidated"
  result_root="$TITANS_WORK_ROOT/results/eval/ruler/$architecture"
  log="$TITANS_WORK_ROOT/logs/titans-ruler-${architecture}-${CONTEXT_LENGTH}.log"

  CUDA_VISIBLE_DEVICES=$gpu python examples/llm_pretrain/titans_ruler.py \
    --ruler-repo "$RULER_REPO" \
    --checkpoint "$checkpoint" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$result_root" \
    --context-lengths "$CONTEXT_LENGTH" \
    --tasks niah_single_1 niah_single_2 niah_single_3 \
    --num-samples "$NUM_SAMPLES" \
    --max-new-tokens 128 \
    >"$log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one architecture failed; inspect $TITANS_WORK_ROOT/logs/titans-ruler-*-${CONTEXT_LENGTH}.log" >&2
  exit 1
fi

echo "Completed official RULER S-NIAH at context length $CONTEXT_LENGTH."
