#!/bin/bash
#SBATCH --job-name=titans-170m-4k-smoke
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_compeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=slurm_jobs/%x_%j.out

# Full-shape execution gate for the Titans 170M LMM recipe on cw-dfw H100.
# Submit from the root of a synced AutoModel checkout. This job generates
# deterministic synthetic tokens and runs one optimizer step; it does not
# download FineWeb-Edu or begin the 15B-token training run.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
DATA_ROOT=$TITANS_WORK_ROOT/data/smoke-4k

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$AUTOMODEL_CHECKOUT/slurm_jobs"

export TITANS_DATA_ROOT=$DATA_ROOT
export TITANS_CHECKPOINT_ROOT=$TITANS_WORK_ROOT/outputs/titans-170m-lmm
export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache
export TRITON_CACHE_DIR=$TITANS_WORK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TITANS_WORK_ROOT/cache/torchinductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
export MASTER_PORT=${MASTER_PORT:-13742}

srun \
  --container-image="$TITANS_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$AUTOMODEL_CHECKOUT" \
  --export=ALL \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$AUTOMODEL_CHECKOUT'
    python tools/make_titans_smoke_data.py \
      --output-dir '$DATA_ROOT' \
      --seq-len 4096 \
      --train-documents 16 \
      --validation-documents 2 \
      --vocab-size 32000
    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node=8 \
      -m nemo_automodel.cli.app \
      examples/llm_pretrain/titans_170m_lmm.yaml \
      --dataset.file_pattern='$DATA_ROOT/train/*.bin' \
      --validation_dataset.file_pattern='$DATA_ROOT/validation/*.bin' \
      --step_scheduler.global_batch_size=8 \
      --step_scheduler.local_batch_size=1 \
      --step_scheduler.max_steps=1 \
      --checkpoint.enabled=false
  "
