#!/bin/bash
# Resumable full Titans LMM/component run on a resolved Blackwell topology.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:?set AUTOMODEL_CHECKOUT}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:?set TITANS_WORK_ROOT}
TITANS_MOUNT_ROOT=${TITANS_MOUNT_ROOT:?set TITANS_MOUNT_ROOT}
TITANS_SCALE=${TITANS_SCALE:?set TITANS_SCALE}
TITANS_EXPERIMENT=${TITANS_EXPERIMENT:-baseline}
TITANS_GPUS_PER_NODE=${TITANS_GPUS_PER_NODE:?set TITANS_GPUS_PER_NODE}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
MAX_STEPS_PER_RUN=${TITANS_MAX_STEPS_PER_RUN:-1700}
TARGET_SEGMENT_SECONDS=${TITANS_TARGET_SEGMENT_SECONDS:-13200}
WORLD_SIZE=$((SLURM_NNODES * TITANS_GPUS_PER_NODE))
if (( 128 % WORLD_SIZE != 0 )); then
  echo "global batch 128 must be divisible by world size $WORLD_SIZE" >&2
  exit 2
fi
MAX_LOCAL_BATCH_SIZE=$((128 / WORLD_SIZE))

case "$TITANS_SCALE" in
  170m|340m)
    TOTAL_STEPS=28610
    TOKEN_BUDGET=15B
    ;;
  760m)
    TOTAL_STEPS=57220
    TOKEN_BUDGET=30B
    ;;
  *) echo "unsupported Titans scale: $TITANS_SCALE" >&2; exit 2 ;;
esac
case "$TITANS_SCALE" in
  170m) DEFAULT_LOCAL_BATCH_SIZE=8 ;;
  340m) DEFAULT_LOCAL_BATCH_SIZE=4 ;;
  760m) DEFAULT_LOCAL_BATCH_SIZE=2 ;;
esac
(( DEFAULT_LOCAL_BATCH_SIZE > MAX_LOCAL_BATCH_SIZE )) && \
  DEFAULT_LOCAL_BATCH_SIZE=$MAX_LOCAL_BATCH_SIZE
LOCAL_BATCH_SIZE=${TITANS_LOCAL_BATCH_SIZE:-$DEFAULT_LOCAL_BATCH_SIZE}
ACTIVATION_CHECKPOINTING=${TITANS_ACTIVATION_CHECKPOINTING:-false}
if (( LOCAL_BATCH_SIZE < 1 || 128 % (WORLD_SIZE * LOCAL_BATCH_SIZE) != 0 )); then
  echo "local batch $LOCAL_BATCH_SIZE does not divide global batch 128 across $WORLD_SIZE ranks" >&2
  exit 2
fi
export TITANS_LOCAL_BATCH_SIZE=$LOCAL_BATCH_SIZE
export TITANS_ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING
echo "Blackwell tuning: world_size=$WORLD_SIZE local_batch_size=$LOCAL_BATCH_SIZE activation_checkpointing=$ACTIVATION_CHECKPOINTING"
if [[ $TITANS_SCALE != 170m && $TITANS_EXPERIMENT != baseline ]]; then
  echo "component ablations are currently defined only for the 170M scale" >&2
  exit 2
fi

case "$TITANS_EXPERIMENT" in
  baseline) MODEL_OVERRIDES="" ;;
  no_persistent)
    MODEL_OVERRIDES="--model.config.num_persistent_memory_tokens=0 --model.config.memory_batch_size=4096"
    ;;
  no_convolution) MODEL_OVERRIDES="--model.config.qkv_conv_kernel_size=0" ;;
  no_momentum) MODEL_OVERRIDES="--model.config.momentum=false" ;;
  no_weight_decay) MODEL_OVERRIDES="--model.config.forget=false" ;;
  depth3) MODEL_OVERRIDES="--model.config.mem_depth=3" ;;
  depth4) MODEL_OVERRIDES="--model.config.mem_depth=4" ;;
  linear_memory)
    echo "linear_memory remains blocked until its momentum recurrence is vectorized" >&2
    exit 2
    ;;
  *) echo "unknown Titans experiment: $TITANS_EXPERIMENT" >&2; exit 2 ;;
esac

RECIPE=examples/llm_pretrain/titans_${TITANS_SCALE}_lmm.yaml
DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-$TOKEN_BUDGET
RUN_NAME=titans-${TITANS_SCALE}-${TITANS_EXPERIMENT}-${TOKEN_BUDGET}-blackwell
CHECKPOINT_ROOT=$TITANS_WORK_ROOT/outputs/$RUN_NAME
RUN_SHA_FILE=$CHECKPOINT_ROOT/git_sha
WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN_NAME}-v1}
WANDB_GROUP=$([[ $TITANS_EXPERIMENT == baseline ]] && echo "lmm-$TITANS_SCALE" || echo "ablations-170m")

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$CHECKPOINT_ROOT" \
  "$TITANS_WORK_ROOT/cache"/{triton,torchinductor,wandb,wandb-data} \
  "$TITANS_WORK_ROOT/.config/wandb" "$AUTOMODEL_CHECKOUT/slurm_jobs"
test -s "$DATA_ROOT/READY"

CURRENT_SHA=$(git -C "$AUTOMODEL_CHECKOUT" rev-parse HEAD)
if [[ -s "$RUN_SHA_FILE" ]]; then
  RECORDED_SHA=$(cat "$RUN_SHA_FILE")
  if [[ $RECORDED_SHA != "$CURRENT_SHA" && -e "$CHECKPOINT_ROOT/LATEST" ]]; then
    echo "refusing to resume checkpoints from $RECORDED_SHA with source $CURRENT_SHA" >&2
    exit 1
  fi
fi
printf '%s\n' "$CURRENT_SHA" >"$RUN_SHA_FILE"

if ! grep -qs "api.wandb.ai" "$HOME/.netrc" && [[ -z ${WANDB_API_KEY:-} ]]; then
  echo "W&B authentication is required for full Titans runs" >&2
  exit 1
fi

export TITANS_DATA_ROOT=$DATA_ROOT
export TITANS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT
export WANDB_RUN_ID
export WANDB_MODE=online
export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache
export TRITON_CACHE_DIR=$TITANS_WORK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TITANS_WORK_ROOT/cache/torchinductor
export WANDB_CACHE_DIR=$TITANS_WORK_ROOT/cache/wandb
export WANDB_DATA_DIR=$TITANS_WORK_ROOT/cache/wandb-data
export WANDB_CONFIG_DIR=$TITANS_WORK_ROOT/.config/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
MASTER_PORT=${MASTER_PORT:-13749}
export MASTER_ADDR MASTER_PORT

START_STEP=-1
if [[ -L "$CHECKPOINT_ROOT/LATEST" ]]; then
  START_TARGET=$(readlink "$CHECKPOINT_ROOT/LATEST")
  START_STEP=${START_TARGET##*_step_}
fi
SEGMENT_START_EPOCH=$(date +%s)

srun \
  --nodes="$SLURM_NNODES" \
  --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --container-image="$TITANS_IMAGE" \
  --container-mounts="$TITANS_MOUNT_ROOT:$TITANS_MOUNT_ROOT" \
  --container-workdir="$AUTOMODEL_CHECKOUT" \
  --export=ALL \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$AUTOMODEL_CHECKOUT'
    torchrun \
      --nnodes=\$SLURM_NNODES \
      --nproc-per-node='$TITANS_GPUS_PER_NODE' \
      --rdzv-backend=c10d \
      --rdzv-endpoint='$MASTER_ADDR:$MASTER_PORT' \
      --rdzv-id='$SLURM_JOB_ID' \
      -m nemo_automodel.cli.app \
      '$RECIPE' \
      --step_scheduler.max_steps='$TOTAL_STEPS' \
      --step_scheduler.max_steps_per_run='$MAX_STEPS_PER_RUN' \
      --step_scheduler.local_batch_size='$LOCAL_BATCH_SIZE' \
      --distributed.activation_checkpointing='$ACTIVATION_CHECKPOINTING' \
      --checkpoint.checkpoint_dir='$CHECKPOINT_ROOT' \
      --wandb.enable=true \
      --wandb.entity=nvidia \
      --wandb.project=titans-paper-reproduction \
      --wandb.name='$RUN_NAME' \
      --wandb.group='$WANDB_GROUP' \
      --wandb.mode=online \
      --wandb.id='$WANDB_RUN_ID' \
      --wandb.resume=allow \
      --wandb.dir='$CHECKPOINT_ROOT/wandb' \
      $MODEL_OVERRIDES
  "

LATEST_TARGET=$(readlink "$CHECKPOINT_ROOT/LATEST")
LATEST_STEP=${LATEST_TARGET##*_step_}
SEGMENT_ELAPSED=$(( $(date +%s) - SEGMENT_START_EPOCH ))
SEGMENT_STEPS=$((LATEST_STEP - START_STEP))
if (( SEGMENT_ELAPSED > 0 && SEGMENT_STEPS > 0 )); then
  PROPOSED_STEPS=$((SEGMENT_STEPS * TARGET_SEGMENT_SECONDS / SEGMENT_ELAPSED))
  MIN_STEPS=$((MAX_STEPS_PER_RUN * 80 / 100))
  MAX_STEPS=$((MAX_STEPS_PER_RUN * 125 / 100))
  (( PROPOSED_STEPS < MIN_STEPS )) && PROPOSED_STEPS=$MIN_STEPS
  (( PROPOSED_STEPS > MAX_STEPS )) && PROPOSED_STEPS=$MAX_STEPS
  NEXT_SEGMENT_STEPS=$((PROPOSED_STEPS / 50 * 50))
  (( NEXT_SEGMENT_STEPS < 50 )) && NEXT_SEGMENT_STEPS=50
else
  NEXT_SEGMENT_STEPS=$MAX_STEPS_PER_RUN
fi
export TITANS_MAX_STEPS_PER_RUN=$NEXT_SEGMENT_STEPS
echo "Segment completed $SEGMENT_STEPS steps in ${SEGMENT_ELAPSED}s; next segment target is $NEXT_SEGMENT_STEPS steps"

if (( LATEST_STEP + 1 < TOTAL_STEPS )); then
  NEXT_JOB_ID=$(env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK \
    sbatch --parsable \
      --partition="$SLURM_JOB_PARTITION" \
      --account="$SLURM_JOB_ACCOUNT" \
      --nodes="$SLURM_NNODES" \
      --ntasks="$SLURM_NNODES" \
      --ntasks-per-node=1 \
      --cpus-per-task="$SLURM_CPUS_PER_TASK" \
      --gpus-per-node="$TITANS_GPUS_PER_NODE" \
      --mem=0 \
      --time="$SLURM_TIMELIMIT" \
      --dependency=singleton \
      --job-name="$SLURM_JOB_NAME" \
      --chdir="$AUTOMODEL_CHECKOUT" \
      --output="$TITANS_WORK_ROOT/logs/${SLURM_JOB_NAME}_%j.out" \
      "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/titans_blackwell_lmm_full.sbatch")
  echo "Completed through step $LATEST_STEP; submitted continuation $NEXT_JOB_ID"
else
  echo "Titans $TITANS_SCALE/$TITANS_EXPERIMENT completed at step $LATEST_STEP"
fi
