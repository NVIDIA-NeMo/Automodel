#!/bin/bash
set -euo pipefail

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "N_NODES: $N_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "Arguments: $@"

# /opt/venv/bin/python -m pip install -e /lustre/fs12/portfolios/llmservice/projects/llmservice_nlp_retro/users/ybabakhin/git/automodel-private
/opt/venv/bin/python -m pip install -e .
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/ybabakhin/triton_cache"
pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils

# Use torchrun for distributed training
if [ "$N_NODES" -gt 1 ]; then
    echo "Running multi-node training with torchrun"
    torchrun \
        --nproc-per-node=$GPUS_PER_NODE \
        --nnodes=$N_NODES \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        examples/retrieval/bi_encoder/finetune.py \
        "$@"
else
    echo "Running single-node training with torchrun"
    torchrun \
        --nproc-per-node=$GPUS_PER_NODE \
        examples/retrieval/bi_encoder/finetune.py \
        "$@"
fi

# Example usage for multi-node training:
# python3 submit_nemo_run.py \
#     --experiment_name release_model_bidirectional \
#     --num_nodes 2 \
#     --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-25.11.00.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/llama3_2_1b.yaml \
#     --eval_script ./scripts/yauhen/eval_script_rteb.sh \
#     --eval_datasets RTEB_QUICK \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path meta-llama/Llama-3.2-1B \
#         --tokenizer.pretrained_model_name_or_path meta-llama/Llama-3.2-1B \
#         --optimizer.lr 5.0e-6 \
#     " \
#     --more_eval_args " \
#         --max_length 4096 \
#         --batch_size 4 \
#         --pooling avg \
#         --padding_side left \
#         --precision bf16 \
#     " \
#     --num_training_rounds 1
