#!/bin/bash
# MFSDP LLaMA3-70B: TP=1 PP=1 CP=1 DP=64, MBS=1, GBS=128, SeqLen=4096
# Reproduced on current cluster to match EOS run (MFSDP-llama3-70b-new/log)
#
# Changes from tp2/mbs4 template:
#   [1] --tensor-model-parallel-size 2 → 1
#   [2] --micro-batch-size 4 → 1
#   [3] removed --sequence-parallel (log: sequence_parallel=False)
#   [4] restored --cross-entropy-loss-fusion + --cross-entropy-fusion-impl te (keep TE CE)
#   [5] removed --apply-layernorm-1p (log: not set)
#   [6] updated MEGATRON_PATH, OUTPUT_PATH, CONTAINER_IMAGE to current cluster
#   [7] account: coreai_devtech_all → coreai_dlalgo_nemorl
#   [8] profile steps: 13-15 → 10-12  (matches log profile_step_start=10, end=12)

source /lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh
JOB_NAME="llama3_70b_mfsdp_tp1_pp1_ep1_cp1_hsdp1_alltoall_mbs1_gbs128_seqlen4096_cw_n8_full-recompute_no_flag_0"

# User config
MEGATRON_PATH=${MEGATRON_PATH:-"/lustre/fsw/portfolios/coreai/users/zhiyul/benchmark-fsdp/Megatron-LM-xuwen"}   # [6] xuwen fork has MFSDP dtype patches
CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com/xuwenc/docker_pytorch:pytorch26.03_te2.14_deepep_x86"}
OUTPUT_PATH=${OUTPUT_PATH:-"/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/MFSDP-llama3-70b-repro"}  # [6]
PROFILE=${PROFILE:-1}   # Set to 1 to enable nsys profiling
WANDB=${WANDB:-1}       # Set to 1 to enable WandB logging

# Environment
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_NODEID
export NCCL_IB_TIMEOUT=19
qxport NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
unset CUDA_DEVICE_MAX_CONNECTIONS

# Benchmark args
PRETRAIN_ARGS=(
    --tensor-model-parallel-size 1    # [1] was 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --num-distributed-optimizer-instances 1
    --micro-batch-size 1              # [2] was 4
    --global-batch-size 128
    --seq-length 4096
    --train-iters 15
)

# Extra args (CLI passthrough)
PRETRAIN_ARGS+=(
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# Architecture args
PRETRAIN_ARGS+=(
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings 4096
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
)

# Training args
PRETRAIN_ARGS+=(
    # --sequence-parallel              # [3] removed — log: sequence_parallel=False
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-timeout-minutes 60
    --disable-bias-linear
    --transformer-impl transformer_engine
    --attention-backend fused
    # --cross-entropy-loss-fusion        # D1: removed — align with FSDP2 (MaskedCrossEntropy)
    # --cross-entropy-fusion-impl te     # D1: removed — align with FSDP2
    # --apply-layernorm-1p             # [5] removed — log: not set
    --init-method-std 0.0134
    --use-mcore-models
    # --train-samples 1953125  # disabled: using --train-iters instead
    --exit-duration-in-mins 230
    --manual-gc
    --manual-gc-interval 10
    --clip-grad 0.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 0.00015
    --min-lr 1e-05
    --lr-decay-style cosine
    # --lr-decay-samples 1949218  # disabled: using --train-iters
    # --lr-warmup-samples 3907    # disabled: using --train-iters
    --lr-decay-iters 15
    --lr-warmup-iters 1
    --bf16
    --mock-data
    --vocab-size 128000
    --split 99,1,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --tokenizer-type NullTokenizer
    --eval-iters 32
    --eval-interval 100
    --auto-detect-ckpt-format
    --load ${OUTPUT_PATH}/checkpoints
    --save ${OUTPUT_PATH}/checkpoints
    --save-interval 500
    --dist-ckpt-strictness log_all
    --log-throughput
    --log-interval 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --tensorboard-dir ${OUTPUT_PATH}/tensorboard
    --outer-dp-sharding-strategy no_shard
)

# Backend args (megatron_fsdp)
PRETRAIN_ARGS+=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --init-model-with-meta-device
    --calculate-per-token-loss
    --fsdp-double-buffer
    --use-nccl-ub
    --fsdp-manual-registration
    --ckpt-format fsdp_dtensor
)

# Profiling (PROFILE=1 bash ...)
if [ "${PROFILE}" = 1 ]; then
    PRETRAIN_ARGS+=(
        --profile
        --profile-step-start 10    # [8] was 13; matches log profile_step_start=10
        --profile-step-end 12      # [8] was 15; matches log profile_step_end=12
        --profile-ranks 0
    )
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,cublas,cudnn \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node --cuda-memory-usage=true \
        -f true -x true -o ${OUTPUT_PATH}/nsys/${JOB_NAME}"
else
    PROFILE_CMD=""
fi

# WandB (WANDB=1 bash ...)
if [ "${WANDB}" = 1 ]; then
    export WANDB_API_KEY=${WANDB_API_KEY}
    PRETRAIN_ARGS+=(
        --wandb-project automodel-dev-zhiyul
        --wandb-entity nvidia
        --wandb-exp-name ${JOB_NAME}
    )
fi

# Training command
TRAINING_CMD="cd ${MEGATRON_PATH} && \
    ${PROFILE_CMD} python pretrain_gpt.py ${PRETRAIN_ARGS[@]}"

# Submit SLURM job
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS}
mkdir -p ${OUTPUT_PATH}/nsys
mkdir -p ${OUTPUT_PATH}/checkpoints
mkdir -p ${OUTPUT_PATH}/tensorboard

set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:20:00
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --output=${SLURM_LOGS}/${JOB_NAME}_%j.log
#SBATCH --exclusive
#SBATCH --dependency=singleton

srun \
    --mpi=pmix -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts="/lustre:/lustre" \
    --container-workdir=${MEGATRON_PATH} \
    bash -x -c "${TRAINING_CMD}"

EOF
set -e
