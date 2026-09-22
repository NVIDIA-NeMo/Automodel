#!/bin/bash
# Consolidate completed 170M Titans checkpoints for Hugging Face evaluation.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-$(cd "$AUTOMODEL_CHECKOUT/../.." && pwd)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_THREADS=${NUM_THREADS:-8}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}

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
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      NUM_THREADS="$NUM_THREADS" \
      PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" \
      bash "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/titans_blackwell_eval_prepare.sbatch"
fi

cd "$AUTOMODEL_CHECKOUT"
export PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"

RUNS=(
  titans-170m-baseline-15B-blackwell-optimized
  titans-170m-mac-15B-blackwell-mac-v1
  titans-170m-mag-15B-blackwell-mag-v1
  titans-170m-mal-15B-blackwell-mal-v1
)

for run in "${RUNS[@]}"; do
  root="$TITANS_WORK_ROOT/outputs/$run"
  if [[ ! -L "$root/LATEST" ]]; then
    echo "Missing LATEST checkpoint pointer: $root/LATEST" >&2
    exit 1
  fi
  latest=$(readlink "$root/LATEST")
  model_dir="$root/$latest/model"
  output_dir="$model_dir/consolidated"

  echo "=== $run / $latest ==="
  if [[ -f "$output_dir/model.safetensors.index.json" || -f "$output_dir/model.safetensors" ]]; then
    echo "Already consolidated: $output_dir"
    continue
  fi
  if [[ ! -x "$model_dir/consolidate.sh" ]]; then
    echo "Missing generated consolidation helper: $model_dir/consolidate.sh" >&2
    exit 1
  fi

  NPROC_PER_NODE="$NPROC_PER_NODE" \
  NUM_THREADS="$NUM_THREADS" \
  CONSOLIDATION_TOOL="$AUTOMODEL_CHECKOUT/tools/offline_hf_consolidation.py" \
    bash "$model_dir/consolidate.sh"

  if [[ ! -f "$output_dir/model.safetensors.index.json" && ! -f "$output_dir/model.safetensors" ]]; then
    echo "Consolidation did not produce Hugging Face safetensors in $output_dir" >&2
    exit 1
  fi
  du -sh "$output_dir"
done

echo "All completed Titans checkpoints are consolidated."
