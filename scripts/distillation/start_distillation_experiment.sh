#!/bin/bash

# Draco ##############################################################
#
# This script is intended as a collection of independent experiment command examples, not an executable chained pipeline.
# Run one block at a time. Round 2, round 3, and eval use explicit checkpoint
# inputs; this script does not create a connected round1 -> round2 -> round3 pipeline.

ROUND2_INPUT_CKPT=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/ministral3_1b_vl/ministral3_1b_all_fsdp2_lr1e4_connector_vlm_teacher_distill_from_gmoreira_3b_ac_projection_warmup100_20260731_174359/epoch_0_step_2047/model/consolidated
ROUND3_INPUT_CKPT=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/ministral3_1b_vl/ministral3_1b_all_fsdp2_lr3e4_connector_vlm_teacher_distill_from_gmoreira_3b_ac_projection_warmup100_from_distill_round1_20260802_133324/epoch_0_step_2047/model/consolidated
EVAL_CKPT=/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/jiaruic/model_checkpoints/nemorun_ministral3_vl_distill_stage_1_epoch_1_image_26_06_08-13-191620/epoch_0_step_2047/model/consolidated

# Stage 1 distillation round 1
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_ministral3_vl_distill_stage_1_epoch_1_image_26_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/jiaruic/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/distillation/train_distill_biencoder.sh \
    --yaml_config examples/retrieval/distillation/ministral3_vl_stage1_vlm_teacher_distill_from_gmoreira_3b_ac_projection_lr1e4_warmup100.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B \
        --teacher_model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/model_checkpoints/ministral3_3b_vl/ministral3_3b_ac_all_fsdp2_lr2e6_llm_replaced-20260710/epoch_0_step_2047/model/consolidated \
        --optimizer.lr 1.0e-4 \
        --optimizer.distill_projection_lr 1.0e-3 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1

# Stage 1 distillation round 2
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_ministral3_vl_distill_stage_1_epoch_2_image_26_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/jiaruic/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/distillation/train_distill_biencoder.sh \
    --yaml_config examples/retrieval/distillation/ministral3_vl_stage1_vlm_teacher_distill_from_gmoreira_3b_ac_projection_lr3e4_warmup100_from_distill_round1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path ${ROUND2_INPUT_CKPT} \
        --tokenizer.pretrained_model_name_or_path ${ROUND2_INPUT_CKPT} \
        --model.restore_projection true \
        --teacher_model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/model_checkpoints/ministral3_3b_vl/ministral3_3b_ac_all_fsdp2_lr2e6_llm_replaced-20260710/epoch_0_step_2047/model/consolidated \
        --optimizer.lr 3.0e-4 \
        --optimizer.distill_projection_lr 3.0e-4 \
        --freeze_config.freeze_vision_tower true \
        --freeze_config.freeze_language_model true \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1

# Stage 2 distillation round 3
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name nemorun_ministral3_vl_distill_stage_2_image_text_26_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/jiaruic/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/distillation/train_distill_biencoder.sh \
    --yaml_config examples/retrieval/distillation/ministral3_vl_stage1_vlm_teacher_distill_from_gmoreira_3b_ac_projection_lr1e5_from_continue_distill_round2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path ${ROUND3_INPUT_CKPT} \
        --tokenizer.pretrained_model_name_or_path ${ROUND3_INPUT_CKPT} \
        --model.restore_projection true \
        --teacher_model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/users/gmoreira/model_checkpoints/ministral3_3b_vl/ministral3_3b_ac_all_fsdp2_lr2e6_llm_replaced-20260710/epoch_0_step_2047/model/consolidated \
        --optimizer.lr 1.0e-5 \
        --optimizer.distill_projection_lr 1.0e-5 \
        --freeze_config.freeze_vision_tower false \
        --freeze_config.freeze_language_model false \
    " \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 1

# Evaluation ###########################################################

# Draco ##############################################################
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name eval_ministral3_vl_distill \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/jiaruic/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl.sh \
    --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
    --more_eval_args " \
        --model_name_or_path ${EVAL_CKPT} \
        --model_type mistral3 \
        --batch_size 16 \
        --max_length 10240 \
        --log_wandb False \
    " \
    --num_training_rounds 0
