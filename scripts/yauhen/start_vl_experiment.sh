#!/bin/bash

# Draco ##############################################################

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_refit \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --optimizer.lr 3.0e-4 \
        --lr_scheduler.lr_warmup_steps 100 \
        --lr_scheduler.lr_decay_style cosine \
        --step_scheduler.num_epochs 3 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name stage_1_single_experiment_2_epochs_lr_3e4_cosine_warmup_100_8B_3e_4_08-14-135149 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_8B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_8B \
        --optimizer.lr 3.0e-4 \
        --lr_scheduler.lr_warmup_steps 100 \
        --lr_scheduler.lr_decay_style cosine \
        --step_scheduler.num_epochs 2 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1 \
    --no_append_timestamp


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name eager_stage_1_epoch_1 \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name eager_stage_1_epoch_1_refit \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --num_training_rounds 1


# Stage 1 epoch 1
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_1_epoch_1_draco_new_code_v2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --optimizer.lr 1.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1


# Stage 1 epoch 2
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_1_epoch_2_draco \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_1_draco_08-12-121116/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_1_draco_08-12-121116/LATEST/model/consolidated \
        --optimizer.lr 3.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1


# Stage 2
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_2_draco \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_2_draco_08-12-191914/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_2_draco_08-12-191914/LATEST/model/consolidated \
        --optimizer.lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_gabriel_stage_2_draco_from_3_epochs \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --optimizer.lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name fk_nemorun_gabriel_stage_2_draco_from_3_epochs_08-13-204729 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --optimizer.lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
        --step_scheduler.num_epochs 3 \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1 \
    --no_append_timestamp


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name fk_nemorun_gabriel_stage_2_draco_from_3_epochs_lr_5e5 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --optimizer.lr 3.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
        --step_scheduler.num_epochs 3 \
    " \
    --num_training_rounds 3

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name full_nemorun_gabriel_stage_2_draco_from_4_epochs_lr_2e7_4_nodes \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --optimizer.lr 2.0e-7 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
        --step_scheduler.num_epochs 6 \
    " \
    --num_training_rounds 2 \
    --use_remote_code_symlink


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name full_nemorun_gabriel_stage_2_draco_from_4_epochs_lr_4e7 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --optimizer.lr 4.0e-7 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
        --step_scheduler.num_epochs 6 \
    " \
    --num_training_rounds 6 \
    --use_remote_code_symlink


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name full_nemorun_gabriel_stage_2_draco_from_4_epochs_lr_7e5 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_3_epochs_lr_3e4_cosine_warmup_100_08-13-101639/LATEST/model/consolidated \
        --optimizer.lr 7.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
        --step_scheduler.num_epochs 6 \
    " \
    --num_training_rounds 3 \
    --use_remote_code_symlink



# # NRT ##############################################################

# # Stage 1 epoch 1. Minimalistic example
# python3 submit_nemo_run.py \
#     --cluster nrt \
#     --experiment_name nemorun_gabriel_stage_1_epoch_1_minimalistic \
#     --num_nodes 1 \
#     --partition interactive \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
#         --optimizer.lr 1.0e-4 \
#         --freeze_config.freeze_vision_tower true \
#         --freeze_config.freeze_language_model true \
#     " \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets 20250405 \
#     --more_eval_args " \
#         --model_type mistral3 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --pooling avg \
#         --corpus_modalities image \
#     " \
#     --num_training_rounds 1

# # Stage 1 epoch 1
# python3 submit_nemo_run.py \
#     --cluster nrt \
#     --experiment_name nemorun_gabriel_stage_1_epoch_1_new_start \
#     --num_nodes 4 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
#         --optimizer.lr 1.0e-4 \
#         --freeze_config.freeze_vision_tower true \
#         --freeze_config.freeze_language_model true \
#     " \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets 20250405 small_textqa_eval_20250508 \
#     --more_eval_args " \
#         --model_type mistral3 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --pooling avg \
#         --corpus_modalities image \
#     " \
#     --num_training_rounds 1


# # Stage 1 epoch 2
# python3 submit_nemo_run.py \
#     --cluster nrt \
#     --experiment_name nemorun_gabriel_stage_1_epoch_2_new_start \
#     --num_nodes 4 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_1.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_1_new_start_08-09-192241/LATEST/model/consolidated \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_1_new_start_08-09-192241/LATEST/model/consolidated \
#         --optimizer.lr 3.0e-4 \
#         --freeze_config.freeze_vision_tower true \
#         --freeze_config.freeze_language_model true \
#     " \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets 20250405 small_textqa_eval_20250508 \
#     --more_eval_args " \
#         --model_type mistral3 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --pooling avg \
#         --corpus_modalities image \
#     " \
#     --num_training_rounds 1


# # Stage 2
# python3 submit_nemo_run.py \
#     --cluster nrt \
#     --experiment_name nemorun_gabriel_stage_2_new_start \
#     --num_nodes 4 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/yauhen/nrt_gabriel_stage_2.yaml \
#     --more_train_args " \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_2_new_start_08-10-074549/LATEST/model/consolidated \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_gabriel_stage_1_epoch_2_new_start_08-10-074549/LATEST/model/consolidated \
#         --optimizer.lr 1.0e-5 \
#         --freeze_config.freeze_vision_tower false \
#         --freeze_config.freeze_language_model false \
#     " \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets 20250405 small_textqa_eval_20250508 \
#     --more_eval_args " \
#         --model_type mistral3 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --pooling avg \
#         --corpus_modalities image \
#     " \
#     --num_training_rounds 1

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
    --cluster draco \
    --experiment_name eval_ministral3_vl_new \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/fk_nemorun_gabriel_stage_2_draco_from_3_epochs_08-13-204729/LATEST/model/consolidated \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name eval_ministral3_vl_new \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_2_08-14-150136/LATEST/model/consolidated \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 0



        # --max_length 10240 \
        # --image_longest_edge 1284 \

        #         --max_length 10248 \
        # --image_longest_edge 1540 \

        # --max_length 10232 \
        # --image_longest_edge 980 \

        #         --max_length 10224 \
        # --image_longest_edge 700 \
        # Do Vision + Text eval!

#     --partition interactive \
# VISION_EVAL TEXT_EVAL VISION_EVAL_MULTILINGUAL TEXT_PLUS_VISION_EVAL

cd /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/stage_1_single_experiment_2_epochs_lr_3e4_cosine_warmup_100_8B_3e_4_08-14-135149/LATEST/model/consolidated/eval_results

# 20250405 small_textqa_eval_20250508

# Nemotron-3-embed eval ##############################################################
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name eval_ministral3_vl_new \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets TEXT_PLUS_VISION_EVAL \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_2_08-14-150136/LATEST/model/consolidated \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name eval_ministral3_vl_new \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_1_08-14-104819/LATEST/model/consolidated \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --cluster draco \
    --partition interactive \
    --experiment_name eval_nemotron3_embed_1b \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets TEXT_EVAL \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Nemotron-3-Embed-1B-BF16 \
        --model_type HF \
        --embedding_dim 2048 \
        --batch_size 16 \
        --max_length 10240 \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --padding_side left \
        --pooling avg \
        --normalize_embeddings True \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
  --cluster draco \
  --experiment_name eval_llama_nemotron_embed_vl_1b_v2 \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets TEXT_PLUS_VISION_EVAL \
  --more_eval_args " \
    --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/llama-nemotron-embed-vl-1b-v2 \
    --model_type eagle_llama \
    --max_input_tiles 6 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
  " \
  --num_training_rounds 0


# VISION_EVAL TEXT_EVAL RTEB_QUICK
