#!/bin/bash
set -euo pipefail

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "N_NODES: $N_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
printf "Arguments:"
printf " %q" "$@"
printf "\n"

# /opt/venv/bin/python -m pip install -e .
/opt/venv/bin/python -m pip install -e . --no-deps
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/triton_cache"
pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils

# Use torchrun for distributed training
if [ "$N_NODES" -gt 1 ]; then
    echo "Running multi-node training with torchrun"
    torchrun \
        --nproc-per-node="$GPUS_PER_NODE" \
        --nnodes="$N_NODES" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        examples/retrieval/bi_encoder/finetune.py \
        "$@"
else
    echo "Running single-node training with torchrun"
    torchrun \
        --nproc-per-node="$GPUS_PER_NODE" \
        examples/retrieval/bi_encoder/finetune.py \
        "$@"
fi

# Sample command
# python3 submit_nemo_run.py \
#     --cluster draco \
#     --experiment_name nemorun_gabriel_stage_1_epoch_1_image_26_06 \
#     --num_nodes 8 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/models/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/models/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B \
#         --optimizer.lr 1.0e-4 \
#         --freeze_config.freeze_vision_tower true \
#         --freeze_config.freeze_language_model true \
#     " \
#     --num_training_rounds 1
