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
    # --cluster nrt \
    # --experiment_name nemorun_gabriel_stage_1_epoch_1_new_start \
    # --num_nodes 4 \
    # --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    # --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    # --train_script ./scripts/yauhen/train_biencoder.sh \
    # --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
    # --more_train_args " \
    #     --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
    #     --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
    #     --optimizer.lr 1.0e-4 \
    #     --freeze_config.freeze_vision_tower true \
    #     --freeze_config.freeze_language_model true \
    # " \
    # --eval_script ./scripts/yauhen/eval_script_vl.sh \
    # --eval_datasets 20250405 small_textqa_eval_20250508 \
    # --more_eval_args " \
    #     --model_type mistral3 \
    #     --batch_size 16 \
    #     --max_length 10240 \
    #     --pooling avg \
    #     --corpus_modalities image \
    # " \
    # --num_training_rounds 1
