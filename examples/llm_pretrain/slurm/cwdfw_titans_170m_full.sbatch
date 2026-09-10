#!/bin/bash
#SBATCH --job-name=titans-170m-lmm-15B
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

# One resumable segment of the 170M LMM 15B-token reproduction. Every segment
# retains the global 28,610-step schedule and advances at most 1,800 steps.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
TITANS_DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-15B
TITANS_CHECKPOINT_ROOT=$TITANS_WORK_ROOT/outputs/titans-170m-lmm-15B
RUN_SHA_FILE=$TITANS_CHECKPOINT_ROOT/git_sha
WANDB_RUN_ID=${WANDB_RUN_ID:-titans170m15bv1}
TOTAL_STEPS=28610

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$TITANS_CHECKPOINT_ROOT" \
  "$TITANS_WORK_ROOT/cache"/{triton,torchinductor,wandb,wandb-data} \
  "$TITANS_WORK_ROOT/.config/wandb" "$AUTOMODEL_CHECKOUT/slurm_jobs"

test -s "$TITANS_DATA_ROOT/READY"
CURRENT_SHA=$(git -C "$AUTOMODEL_CHECKOUT" rev-parse HEAD)
if [[ -s "$RUN_SHA_FILE" ]]; then
  test "$(cat "$RUN_SHA_FILE")" = "$CURRENT_SHA"
else
  printf '%s\n' "$CURRENT_SHA" >"$RUN_SHA_FILE"
fi
if ! grep -qs "api.wandb.ai" "$HOME/.netrc" && [[ -z ${WANDB_API_KEY:-} ]]; then
  echo "W&B authentication is required for the full reproduction" >&2
  exit 1
fi

export TITANS_DATA_ROOT TITANS_CHECKPOINT_ROOT WANDB_RUN_ID
export WANDB_MODE=online
export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache
export TRITON_CACHE_DIR=$TITANS_WORK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TITANS_WORK_ROOT/cache/torchinductor
export WANDB_CACHE_DIR=$TITANS_WORK_ROOT/cache/wandb
export WANDB_DATA_DIR=$TITANS_WORK_ROOT/cache/wandb-data
export WANDB_CONFIG_DIR=$TITANS_WORK_ROOT/.config/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
export MASTER_PORT=${MASTER_PORT:-13744}

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
      --step_scheduler.max_steps_per_run=1800 \
      --checkpoint.checkpoint_dir='$TITANS_CHECKPOINT_ROOT' \
      --wandb.enable=true \
      --wandb.entity=nvidia \
      --wandb.project=titans-paper-reproduction \
      --wandb.name=titans-170m-lmm-15B \
      --wandb.group=lmm-170m \
      --wandb.mode=online \
      --wandb.id='$WANDB_RUN_ID' \
      --wandb.resume=allow \
      --wandb.dir='$TITANS_CHECKPOINT_ROOT/wandb'
  "

LATEST_TARGET=$(readlink "$TITANS_CHECKPOINT_ROOT/LATEST")
LATEST_STEP=${LATEST_TARGET##*_step_}
if (( LATEST_STEP + 1 < TOTAL_STEPS )); then
  NEXT_JOB_ID=$(sbatch --parsable --chdir="$AUTOMODEL_CHECKOUT" "$0")
  echo "Completed through step $LATEST_STEP; submitted continuation $NEXT_JOB_ID"
else
  echo "Titans 170M reproduction completed at step $LATEST_STEP"
fi
