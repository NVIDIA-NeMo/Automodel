#!/bin/bash
set -euo pipefail

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "N_NODES: $N_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
printf "Arguments:"
printf " %q" "$@"
printf "\n"

# TODO: change TRITON_CACHE_DIR or add it to .bashrc
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/triton_cache"

# /opt/venv/bin/python -m pip install -e .
/opt/venv/bin/python -m pip install -e . --no-deps
/opt/venv/bin/python -m pip uninstall -y mistral_common
pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils

# Use torchrun for distributed training
if [ "$N_NODES" -gt 1 ]; then
    echo "Running multi-node training with torchrun"
    torchrun \
        --nproc-per-node="$GPUS_PER_NODE" \
        --nnodes="$N_NODES" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        examples/retrieval/cross_encoder/finetune.py \
        "$@"
else
    echo "Running single-node training with torchrun"
    torchrun \
        --nproc-per-node="$GPUS_PER_NODE" \
        examples/retrieval/cross_encoder/finetune.py \
        "$@"
fi

# Example usage for multi-node training:
# python3 submit_nemo_run.py \
#     --cluster draco \
#     --experiment_name ministral3_1b_reranker_image_text_768k \
#     --num_nodes 8 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --yaml_config examples/retrieval/cross_encoder/yauhen/ministral3_image_text_ranker_768k.yaml \
#     --train_script ./scripts/yauhen/train_reranker.sh \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ronaya/model_checkpoints/ministral3_1b_jiauri_20260731_174359/consolidated \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ronaya/model_checkpoints/ministral3_1b_jiauri_20260731_174359/consolidated \
#         --optimizer.lr 5.0e-6 \
#     " \
#     --num_training_rounds 1
