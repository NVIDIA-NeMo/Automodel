#!/bin/bash

# Draco ##############################################################

# # Stage 1 epoch 1
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

# # Stage 1 epoch 2
# python3 submit_nemo_run.py \
#     --cluster draco \
#     --experiment_name nemorun_gabriel_stage_1_epoch_2_image_26_06 \
#     --num_nodes 8 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path  \
#         --tokenizer.pretrained_model_name_or_path  \
#         --optimizer.lr 3.0e-4 \
#         --freeze_config.freeze_vision_tower true \
#         --freeze_config.freeze_language_model true \
#     " \
#     --num_training_rounds 1

# # Stage 2
# python3 submit_nemo_run.py \
#     --cluster draco \
#     --experiment_name nemorun_gabriel_stage_2_image_26_06 \
#     --num_nodes 8 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path  \
#         --tokenizer.pretrained_model_name_or_path  \
#         --optimizer.lr 1.0e-5 \
#         --freeze_config.freeze_vision_tower false \
#         --freeze_config.freeze_language_model false \
#     " \
#     --num_training_rounds 1

# NRT ##############################################################

# Stage 1 epoch 1. Minimalistic example
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_1_epoch_1_minimalistic \
    --num_nodes 1 \
    --partition interactive \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets 20250405 \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --pooling avg \
        --corpus_modalities image \
    " \
    --num_training_rounds 1

# Stage 1 epoch 1
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_1_epoch_1_new_start \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets 20250405 small_textqa_eval_20250508 \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --pooling avg \
        --corpus_modalities image \
    " \
    --num_training_rounds 1


# Stage 1 epoch 2
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_1_epoch_2_new_start \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_1_new_start_08-09-192241/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_1_new_start_08-09-192241/LATEST/model/consolidated \
        --optimizer.lr 3.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets 20250405 small_textqa_eval_20250508 \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --pooling avg \
        --corpus_modalities image \
    " \
    --num_training_rounds 1


# Stage 2
python3 submit_nemo_run.py \
    --cluster nrt \
    --experiment_name nemorun_gabriel_stage_2_new_start \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_2_new_start_08-10-074549/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_2_new_start_08-10-074549/LATEST/model/consolidated \
        --optimizer.lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets 20250405 small_textqa_eval_20250508 \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --pooling avg \
        --corpus_modalities image \
    " \
    --num_training_rounds 1

# Evaluation ###########################################################

# Draco ##############################################################
# python3 submit_nemo_run.py \
#     --cluster draco \
#     --experiment_name eval_ministral3_vl \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --num_nodes 1 \
#     --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.06.sqsh \
#     --train_script "" \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets small_textqa_eval_20250508 \
#     --more_eval_args " \
#         --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_1b_vl/stage_1_2_epochs_lr_1e4_cosine_warmup_50/LATEST/model/consolidated \
#         --model_type mistral3 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --pooling avg \
#         --corpus_modalities image \
#     " \
#     --num_training_rounds 0

# NRT ##############################################################
python3 submit_nemo_run.py \
    --cluster nrt \
    --partition interactive \
    --experiment_name eval_ministral3_vl_new \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL VISION_EVAL_MULTILINGUAL TEXT_PLUS_VISION_EVAL \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_2_new_start_08-10-130144/LATEST/model/consolidated \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 0


# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_2_new_start_08-10-130144/LATEST/model/consolidated
# 20250405 small_textqa_eval_20250508

# Nemotron-3-embed eval ##############################################################
# python3 submit_nemo_run.py \
#     --cluster nrt \
#     --partition interactive \
#     --experiment_name eval_nemotron3_embed_1b \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --num_nodes 1 \
#     --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --train_script "" \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets small_textqa_eval_20250508 \
#     --more_eval_args " \
#         --model_name_or_path nvidia/Nemotron-3-Embed-1B-BF16 \
#         --model_type HF \
#         --embedding_dim 2048 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --precision bf16 \
#         --attn_implementation flash_attention_2 \
#         --padding_side left \
#         --pooling avg \
#         --normalize_embeddings True \
#         --corpus_modalities text \
#     " \
#     --num_training_rounds 0

