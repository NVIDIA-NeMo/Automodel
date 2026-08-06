#!/bin/bash

# Draco ##############################################################

# Stage 1 epoch 1
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_1_epoch_1_image_26_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
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
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_1_epoch_2_image_26_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
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

# Stage 2
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_2_image_26_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path  \
        --tokenizer.pretrained_model_name_or_path  \
        --optimizer.lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
    " \
    --num_training_rounds 1

# NRT ##############################################################

# Stage 1 epoch 1
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_1_epoch_1_image_26_06_nrt \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
    --partition batch_block1 \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1

# Stage 1 epoch 2
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_1_epoch_2_image_26_06_nrt \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
    --partition batch_block1 \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path  \
        --tokenizer.pretrained_model_name_or_path  \
        --optimizer.lr 3.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1

# Stage 2
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_2_image_26_06_nrt \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
    --partition batch_block1 \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path  \
        --tokenizer.pretrained_model_name_or_path  \
        --optimizer.lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
    " \
    --num_training_rounds 1
