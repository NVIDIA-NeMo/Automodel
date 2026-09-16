#!/bin/bash
# Prepare cluster-local FineWeb-Edu data, then submit the full training chain.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:?set AUTOMODEL_CHECKOUT}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:?set TITANS_WORK_ROOT}
TITANS_MOUNT_ROOT=${TITANS_MOUNT_ROOT:?set TITANS_MOUNT_ROOT}
TITANS_SCALE=${TITANS_SCALE:?set TITANS_SCALE}
TITANS_EXPERIMENT=${TITANS_EXPERIMENT:-baseline}
TITANS_TRAIN_NODES=${TITANS_TRAIN_NODES:?set TITANS_TRAIN_NODES}
TITANS_GPUS_PER_NODE=${TITANS_GPUS_PER_NODE:?set TITANS_GPUS_PER_NODE}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}

case "$TITANS_SCALE" in
  170m|340m)
    MAX_TOKENS=15B
    TOKEN_BUDGET=15B
    MIN_TRAIN_SHARDS=15
    ;;
  760m)
    MAX_TOKENS=30B
    TOKEN_BUDGET=30B
    MIN_TRAIN_SHARDS=30
    ;;
  *) echo "unsupported Titans scale: $TITANS_SCALE" >&2; exit 2 ;;
esac

DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-$TOKEN_BUDGET
STAGING_ROOT=$TITANS_WORK_ROOT/data/.fineweb-edu-llama2-$TOKEN_BUDGET-$SLURM_JOB_ID
READY_MARKER=$DATA_ROOT/READY

mkdir -p "$TITANS_WORK_ROOT"/{cache,data,logs,outputs} "$AUTOMODEL_CHECKOUT/slurm_jobs"
export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache

if [[ ! -s "$READY_MARKER" ]]; then
  mkdir -p "$STAGING_ROOT"/{train,validation}
  srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$TITANS_IMAGE" \
    --container-mounts="$TITANS_MOUNT_ROOT:$TITANS_MOUNT_ROOT" \
    --container-workdir="$AUTOMODEL_CHECKOUT" \
    --export=ALL \
    bash -lc "
      set -euo pipefail
      export PYTHONPATH='$AUTOMODEL_CHECKOUT'
      python tools/nanogpt_data_processor.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --set-name sample-100BT \
        --tokenizer NousResearch/Llama-2-7b-hf \
        --output-dir '$STAGING_ROOT/train' \
        --validation-dir '$STAGING_ROOT/validation' \
        --validation-tokens 4194304 \
        --max-tokens '$MAX_TOKENS' \
        --tokens-per-shard 1B \
        --output-prefix fineweb_edu \
        --data-cache-dir '$TITANS_WORK_ROOT/cache/fineweb-edu' \
        --num-workers 32 \
        --chunk-size 128 \
        --prefetch 2
    "

  shopt -s nullglob
  TRAIN_FILES=("$STAGING_ROOT"/train/fineweb_edu_*.bin)
  VALIDATION_FILES=("$STAGING_ROOT"/validation/fineweb_edu_*.bin)
  if (( ${#TRAIN_FILES[@]} < MIN_TRAIN_SHARDS || ${#VALIDATION_FILES[@]} < 1 )); then
    echo "incomplete dataset: train=${#TRAIN_FILES[@]} validation=${#VALIDATION_FILES[@]}" >&2
    exit 1
  fi
  printf 'git_sha=%s\ndataset=HuggingFaceFW/fineweb-edu\nconfig=sample-100BT\ntokenizer=NousResearch/Llama-2-7b-hf\ntrain_tokens=%s\nvalidation_tokens=4194304\n' \
    "$(git -C "$AUTOMODEL_CHECKOUT" rev-parse HEAD)" "$MAX_TOKENS" >"$STAGING_ROOT/READY"
  if [[ -e "$DATA_ROOT" ]]; then
    echo "refusing to replace incomplete data root: $DATA_ROOT" >&2
    exit 1
  fi
  mv "$STAGING_ROOT" "$DATA_ROOT"
fi

TRAIN_JOB_ID=$(env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK \
  sbatch --parsable \
    --partition="$SLURM_JOB_PARTITION" \
    --account="$SLURM_JOB_ACCOUNT" \
    --nodes="$TITANS_TRAIN_NODES" \
    --ntasks="$TITANS_TRAIN_NODES" \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --gpus-per-node="$TITANS_GPUS_PER_NODE" \
    --mem=0 \
    --time=4:00:00 \
    --dependency=singleton \
    --job-name="titans-${TITANS_SCALE}-${TITANS_EXPERIMENT}-${TOKEN_BUDGET}-blackwell" \
    --chdir="$AUTOMODEL_CHECKOUT" \
    --output="$TITANS_WORK_ROOT/logs/titans-${TITANS_SCALE}-${TITANS_EXPERIMENT}-${TOKEN_BUDGET}-blackwell_%j.out" \
    "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/titans_blackwell_lmm_full.sbatch")
echo "Dataset ready at $DATA_ROOT"
echo "Submitted full Titans training job $TRAIN_JOB_ID"
