#!/bin/bash
#SBATCH --job-name=titans-170m-ablation
#SBATCH --partition=batch
#SBATCH --account=coreai_dlalgo_compeval
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel/logs/%x_%j.out

# Resumable multi-node runner for the paper's 170M component ablations.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_EXPERIMENT=${TITANS_EXPERIMENT:?set TITANS_EXPERIMENT to an ablation name}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
TITANS_DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-15B
TOTAL_STEPS=${TITANS_TOTAL_STEPS:-28610}
WARMUP_STEPS=${TITANS_WARMUP_STEPS:-286}
NODES=$SLURM_NNODES
WORLD_SIZE=$((NODES * 8))

if (( 128 % WORLD_SIZE != 0 )); then
  echo "global batch 128 must be divisible by world size $WORLD_SIZE" >&2
  exit 2
fi

case "$TITANS_EXPERIMENT" in
  baseline)
    MODEL_OVERRIDES=""
    ;;
  no_persistent)
    MODEL_OVERRIDES="--model.config.num_persistent_memory_tokens=0 --model.config.memory_batch_size=4096"
    ;;
  no_convolution)
    MODEL_OVERRIDES="--model.config.qkv_conv_kernel_size=0"
    ;;
  no_momentum)
    MODEL_OVERRIDES="--model.config.momentum=false"
    ;;
  no_weight_decay)
    MODEL_OVERRIDES="--model.config.forget=false"
    ;;
  depth3)
    MODEL_OVERRIDES="--model.config.mem_depth=3"
    ;;
  depth4)
    MODEL_OVERRIDES="--model.config.mem_depth=4"
    ;;
  linear_memory)
    echo "linear_memory is blocked until the momentum recurrence has a vectorized kernel" >&2
    exit 2
    ;;
  *)
    echo "unknown Titans ablation: $TITANS_EXPERIMENT" >&2
    exit 2
    ;;
esac

RUN_SUFFIX=$TITANS_EXPERIMENT
if (( TOTAL_STEPS < 28610 )); then
  RUN_SUFFIX=${RUN_SUFFIX}-pilot-${TOTAL_STEPS}
fi
RUN_NAME=titans-170m-${RUN_SUFFIX}
TITANS_CHECKPOINT_ROOT=$TITANS_WORK_ROOT/outputs/$RUN_NAME
RUN_SHA_FILE=$TITANS_CHECKPOINT_ROOT/git_sha
WANDB_RUN_ID=${WANDB_RUN_ID:-$(printf '%s-v1' "$RUN_NAME" | tr -cd '[:alnum:]-')}
MAX_STEPS_PER_RUN=${TITANS_MAX_STEPS_PER_RUN:-$((1500 * NODES))}

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$TITANS_CHECKPOINT_ROOT" \
  "$TITANS_WORK_ROOT/cache"/{triton,torchinductor,wandb,wandb-data} \
  "$TITANS_WORK_ROOT/.config/wandb" "$AUTOMODEL_CHECKOUT/slurm_jobs"

test -s "$TITANS_DATA_ROOT/READY"
CURRENT_SHA=$(git -C "$AUTOMODEL_CHECKOUT" rev-parse HEAD)
if [[ -s "$RUN_SHA_FILE" ]]; then
  RECORDED_SHA=$(cat "$RUN_SHA_FILE")
  if [[ "$RECORDED_SHA" != "$CURRENT_SHA" && -e "$TITANS_CHECKPOINT_ROOT/LATEST" ]]; then
    echo "refusing to resume checkpoints from $RECORDED_SHA with source $CURRENT_SHA" >&2
    exit 1
  fi
fi
printf '%s\n' "$CURRENT_SHA" >"$RUN_SHA_FILE"

if ! grep -qs "api.wandb.ai" "$HOME/.netrc" && [[ -z ${WANDB_API_KEY:-} ]]; then
  echo "W&B authentication is required for Titans ablations" >&2
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
export MASTER_PORT=${MASTER_PORT:-13745}

SCHEDULER_OVERRIDES=""
if (( TOTAL_STEPS < 28610 )); then
  SCHEDULER_OVERRIDES="--step_scheduler.ckpt_every_steps=$TOTAL_STEPS --step_scheduler.val_every_steps=$TOTAL_STEPS --step_scheduler.log_remote_every_steps=1"
fi

srun \
  --nodes="$NODES" \
  --ntasks="$NODES" \
  --ntasks-per-node=1 \
  --container-image="$TITANS_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$AUTOMODEL_CHECKOUT" \
  --export=ALL \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$AUTOMODEL_CHECKOUT'
    torchrun \
      --nnodes=\$SLURM_NNODES \
      --nproc-per-node=8 \
      --rdzv-backend=c10d \
      --rdzv-endpoint='$MASTER_ADDR:$MASTER_PORT' \
      --rdzv-id='$SLURM_JOB_ID' \
      -m nemo_automodel.cli.app \
      examples/llm_pretrain/titans_170m_lmm.yaml \
      --step_scheduler.max_steps='$TOTAL_STEPS' \
      --step_scheduler.max_steps_per_run='$MAX_STEPS_PER_RUN' \
      --lr_scheduler.lr_warmup_steps='$WARMUP_STEPS' \
      --checkpoint.checkpoint_dir='$TITANS_CHECKPOINT_ROOT' \
      --dist_env.timeout_minutes=60 \
      --wandb.enable=true \
      --wandb.entity=nvidia \
      --wandb.project=titans-paper-ablations \
      --wandb.name='$RUN_NAME' \
      --wandb.group=170m-table4 \
      --wandb.mode=online \
      --wandb.id='$WANDB_RUN_ID' \
      --wandb.resume=allow \
      --wandb.dir='$TITANS_CHECKPOINT_ROOT/wandb' \
      $MODEL_OVERRIDES \
      $SCHEDULER_OVERRIDES
  "

LATEST_TARGET=$(readlink "$TITANS_CHECKPOINT_ROOT/LATEST")
LATEST_STEP=${LATEST_TARGET##*_step_}
if (( LATEST_STEP + 1 < TOTAL_STEPS )); then
  NEXT_JOB_ID=$(env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK \
    sbatch --parsable \
      --partition=batch \
      --account=coreai_dlalgo_compeval \
      --nodes="$NODES" \
      --ntasks="$NODES" \
      --ntasks-per-node=1 \
      --cpus-per-task=128 \
      --gpus-per-node=8 \
      --exclusive \
      --mem=0 \
      --time=4:00:00 \
      --dependency=singleton \
      --job-name="$SLURM_JOB_NAME" \
      --chdir="$AUTOMODEL_CHECKOUT" \
      --output="$TITANS_WORK_ROOT/logs/${SLURM_JOB_NAME}_%j.out" \
      "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/cwdfw_titans_170m_ablation.sbatch")
  echo "Completed through step $LATEST_STEP; submitted continuation $NEXT_JOB_ID"
else
  echo "Titans ablation $TITANS_EXPERIMENT completed at step $LATEST_STEP"
fi
