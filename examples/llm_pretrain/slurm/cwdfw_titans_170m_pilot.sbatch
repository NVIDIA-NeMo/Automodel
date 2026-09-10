#!/bin/bash
#SBATCH --job-name=titans-170m-pilot
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_compeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=slurm_jobs/%x_%j.out

# Ten-step production-shaped pilot for the 170M memory-only LMM. It trains on
# 5.24M FineWeb-Edu tokens with the paper's effective global batch and writes
# a checkpoint. Run cwdfw_titans_fineweb_edu_prepare.sbatch first.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-pilot
TRAIN_DATA=$DATA_ROOT/train/fineweb_edu_max_tokens_64000000/dataset.bin
VALIDATION_ROOT=$DATA_ROOT/validation-health
READY_MARKER=$DATA_ROOT/READY
CHECKPOINT_ROOT=$TITANS_WORK_ROOT/outputs/titans-170m-lmm-pilot-$SLURM_JOB_ID

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$AUTOMODEL_CHECKOUT/slurm_jobs"
test -s "$READY_MARKER"
test -s "$TRAIN_DATA"
test -s "${TRAIN_DATA%.bin}.bos.idx"
test -s "$VALIDATION_ROOT/validation/fineweb_edu_smoke.bin"

export TITANS_DATA_ROOT=$DATA_ROOT
export TITANS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT
export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache
export TRITON_CACHE_DIR=$TITANS_WORK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TITANS_WORK_ROOT/cache/torchinductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
export MASTER_PORT=${MASTER_PORT:-13743}

srun \
  --container-image="$TITANS_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$AUTOMODEL_CHECKOUT" \
  --export=ALL \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$AUTOMODEL_CHECKOUT'

    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node=8 \
      -m nemo_automodel.cli.app \
      examples/llm_pretrain/titans_170m_lmm.yaml \
      --dataset.file_pattern='$TRAIN_DATA' \
      --validation_dataset.file_pattern='$VALIDATION_ROOT/validation/*.bin' \
      --step_scheduler.max_steps=10 \
      --step_scheduler.ckpt_every_steps=10 \
      --step_scheduler.val_every_steps=10 \
      --step_scheduler.log_remote_every_steps=1 \
      --lr_scheduler.lr_warmup_steps=1 \
      --checkpoint.checkpoint_dir='$CHECKPOINT_ROOT'
  "
