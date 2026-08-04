#!/bin/bash

# Stage 1 epoch 1
python3 submit_nemo_run.py \
    --experiment_name nemorun_gabriel_stage_1_epoch_1 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image nvcr.io/nvidian/nemo-automodel:26.04.rc10 \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/models/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/models/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1

# Stage 1 epoch 2
python3 submit_nemo_run.py \
    --experiment_name nemorun_gabriel_stage_1_epoch_2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image nvcr.io/nvidian/nemo-automodel:26.04.rc10 \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path  \
        --tokenizer.pretrained_model_name_or_path  \
        --optimizer.lr 3.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1

