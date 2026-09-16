#!/bin/bash
# Portable ten-step pilot for Titans LMM scales and 170M component ablations.
# Resource directives are supplied by tools/submit_titans_blackwell.sh after
# slurm-cli resolves the cluster's 8-GPU or NVL72 topology.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:?set AUTOMODEL_CHECKOUT}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:?set TITANS_WORK_ROOT}
TITANS_MOUNT_ROOT=${TITANS_MOUNT_ROOT:?set TITANS_MOUNT_ROOT}
TITANS_SCALE=${TITANS_SCALE:?set TITANS_SCALE}
TITANS_EXPERIMENT=${TITANS_EXPERIMENT:-baseline}
TITANS_GPUS_PER_NODE=${TITANS_GPUS_PER_NODE:?set TITANS_GPUS_PER_NODE}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}

case "$TITANS_SCALE" in
  170m|340m|760m) ;;
  *) echo "unsupported Titans scale: $TITANS_SCALE" >&2; exit 2 ;;
esac
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
DATA_ROOT=$TITANS_WORK_ROOT/data/fineweb-edu-llama2-pilot
TRAIN_DATA=$DATA_ROOT/train/fineweb_edu_max_tokens_64M/dataset.bin
TRAIN_INDEX=${TRAIN_DATA%.bin}.bos.idx
VALIDATION_ROOT=$DATA_ROOT/validation-health-$SLURM_NNODES-$TITANS_GPUS_PER_NODE
RUN_NAME=titans-${TITANS_SCALE}-${TITANS_EXPERIMENT}-blackwell-pilot
CHECKPOINT_ROOT=$TITANS_WORK_ROOT/outputs/$RUN_NAME-$SLURM_JOB_ID

mkdir -p "$TITANS_WORK_ROOT"/{cache,logs,outputs} "$DATA_ROOT/train" \
  "$TITANS_WORK_ROOT/cache"/{triton,torchinductor,wandb,wandb-data} \
  "$TITANS_WORK_ROOT/.config/wandb" "$AUTOMODEL_CHECKOUT/slurm_jobs"

if ! grep -qs "api.wandb.ai" "$HOME/.netrc" && [[ -z ${WANDB_API_KEY:-} ]]; then
  echo "W&B authentication is required for Blackwell pilots" >&2
  exit 1
fi

export TITANS_DATA_ROOT=$DATA_ROOT
export TITANS_CHECKPOINT_ROOT=$CHECKPOINT_ROOT
export HF_HOME=$TITANS_WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$TITANS_WORK_ROOT/cache
export TRITON_CACHE_DIR=$TITANS_WORK_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$TITANS_WORK_ROOT/cache/torchinductor
export WANDB_CACHE_DIR=$TITANS_WORK_ROOT/cache/wandb
export WANDB_DATA_DIR=$TITANS_WORK_ROOT/cache/wandb-data
export WANDB_CONFIG_DIR=$TITANS_WORK_ROOT/.config/wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prepare the small pilot corpus exactly once, even for a multi-node target.
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
    if [[ ! -s '$TRAIN_DATA' || ! -s '$TRAIN_INDEX' ]]; then
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
    test -s '$TRAIN_DATA'
    test -s '$TRAIN_INDEX'
    python tools/make_titans_smoke_data.py \
      --output-dir '$VALIDATION_ROOT' \
      --seq-len 4096 \
      --train-documents 16 \
      --validation-documents 16 \
      --vocab-size 32000
  "

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR==1 {print; exit}')
MASTER_PORT=${MASTER_PORT:-13747}
export MASTER_ADDR MASTER_PORT

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
      --dataset.file_pattern='$TRAIN_DATA' \
      --validation_dataset.file_pattern='$VALIDATION_ROOT/validation/*.bin' \
      --step_scheduler.max_steps=10 \
      --step_scheduler.ckpt_every_steps=10 \
      --step_scheduler.val_every_steps=10 \
      --step_scheduler.log_remote_every_steps=1 \
      --lr_scheduler.lr_warmup_steps=1 \
      --checkpoint.checkpoint_dir='$CHECKPOINT_ROOT' \
      --wandb.enable=true \
      --wandb.entity=nvidia \
      --wandb.project=titans-paper-blackwell-pilots \
      --wandb.name='$RUN_NAME' \
      --wandb.group='${TITANS_SCALE}-lmm' \
      --wandb.mode=online \
      --wandb.id='${RUN_NAME}-${SLURM_JOB_ID}' \
      --wandb.resume=never \
      --wandb.dir='$CHECKPOINT_ROOT/wandb' \
      $MODEL_OVERRIDES
  "
