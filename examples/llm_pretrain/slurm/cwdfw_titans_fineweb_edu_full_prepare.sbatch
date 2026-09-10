#!/bin/bash
#SBATCH --job-name=titans-full-data
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_compeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel/logs/%x_%j.out

# Prepare 15B non-repeated Llama-2 tokens plus a held-out validation shard
# through AutoModel's Hugging Face -> NanoGPT processor, then launch training.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-15B
STAGING_ROOT=$TITANS_WORK_ROOT/data/.fineweb-edu-llama2-15B-$SLURM_JOB_ID
READY_MARKER=$DATA_ROOT/READY

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$AUTOMODEL_CHECKOUT/slurm_jobs"

export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache

if [[ ! -s "$READY_MARKER" ]]; then
  mkdir -p "$STAGING_ROOT"/{train,validation}
  srun \
    --container-image="$TITANS_IMAGE" \
    --container-mounts=/lustre:/lustre \
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
        --max-tokens 15B \
        --tokens-per-shard 1B \
        --output-prefix fineweb_edu \
        --data-cache-dir '$TITANS_WORK_ROOT/cache/fineweb-edu' \
        --num-workers 112 \
        --chunk-size 128 \
        --prefetch 2
    "

  shopt -s nullglob
  TRAIN_FILES=("$STAGING_ROOT"/train/fineweb_edu_*.bin)
  VALIDATION_FILES=("$STAGING_ROOT"/validation/fineweb_edu_*.bin)
  TRAIN_SHARDS=${#TRAIN_FILES[@]}
  VALIDATION_SHARDS=${#VALIDATION_FILES[@]}
  if (( TRAIN_SHARDS < 15 || VALIDATION_SHARDS < 1 )); then
    echo "incomplete dataset: train_shards=$TRAIN_SHARDS validation_shards=$VALIDATION_SHARDS" >&2
    exit 1
  fi
  printf 'git_sha=%s\ndataset=HuggingFaceFW/fineweb-edu\nconfig=sample-100BT\ntokenizer=NousResearch/Llama-2-7b-hf\ntrain_tokens=15000000000\nvalidation_tokens=4194304\n' \
    "$(git -C "$AUTOMODEL_CHECKOUT" rev-parse HEAD)" >"$STAGING_ROOT/READY"
  if [[ -e "$DATA_ROOT" ]]; then
    echo "refusing to replace incomplete data root: $DATA_ROOT" >&2
    exit 1
  fi
  mv "$STAGING_ROOT" "$DATA_ROOT"
fi

TRAIN_JOB_ID=$(env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK \
  sbatch --parsable \
    --chdir="$AUTOMODEL_CHECKOUT" \
    examples/llm_pretrain/slurm/cwdfw_titans_170m_full.sbatch)
echo "Dataset ready at $DATA_ROOT"
echo "Submitted full Titans training job $TRAIN_JOB_ID"
