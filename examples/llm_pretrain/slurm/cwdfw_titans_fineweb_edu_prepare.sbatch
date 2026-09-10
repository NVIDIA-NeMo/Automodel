#!/bin/bash
#SBATCH --job-name=titans-fineweb-prepare
#SBATCH --partition=cpu
#SBATCH --account=coreai_dlalgo_compeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel/logs/%x_%j.out

# Prepare the reusable FineWeb-Edu pilot shard with AutoModel's NanoGPT
# processor. On success, submit the eight-H100 pilot from the same checkout.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-pilot
TRAIN_DATA=$DATA_ROOT/train/fineweb_edu_max_tokens_64M/dataset.bin
VALIDATION_ROOT=$DATA_ROOT/validation-health
READY_MARKER=$DATA_ROOT/READY

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$DATA_ROOT/train" "$AUTOMODEL_CHECKOUT/slurm_jobs"

export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache

srun \
  --container-image="$TITANS_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$AUTOMODEL_CHECKOUT" \
  --export=ALL \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$AUTOMODEL_CHECKOUT'

    if [[ ! -s '$TRAIN_DATA' || ! -s '\${TRAIN_DATA%.bin}.bos.idx' ]]; then
      python tools/nanogpt_data_processor.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --set-name sample-10BT \
        --tokenizer NousResearch/Llama-2-7b-hf \
        --output-dir '$DATA_ROOT/train/fineweb_edu' \
        --data-cache-dir '$TITANS_WORK_ROOT/cache/fineweb-edu' \
        --max-tokens 64M \
        --num-workers 32 \
        --chunk-size 128 \
        --prefetch 2
    fi

    python tools/make_titans_smoke_data.py \
      --output-dir '$VALIDATION_ROOT' \
      --seq-len 4096 \
      --train-documents 16 \
      --validation-documents 16 \
      --vocab-size 32000

    test -s '$TRAIN_DATA'
    test -s '\${TRAIN_DATA%.bin}.bos.idx'
    test -s '$VALIDATION_ROOT/validation/fineweb_edu_smoke.bin'
    printf 'dataset=HuggingFaceFW/fineweb-edu\\nconfig=sample-10BT\\ntokenizer=NousResearch/Llama-2-7b-hf\\nmax_tokens=64000000\\n' \
      > '$READY_MARKER'
  "

PILOT_JOB_ID=$(sbatch --parsable \
  --chdir="$AUTOMODEL_CHECKOUT" \
  examples/llm_pretrain/slurm/cwdfw_titans_170m_pilot.sbatch)
echo "Prepared $TRAIN_DATA"
echo "Submitted Titans pilot job $PILOT_JOB_ID"
