#!/bin/bash

# Cross-encoder distillation submission example: Ministral3 1B student, frozen 8B teacher.
# Run one block at a time; this is a collection of command examples, not a pipeline.

STUDENT_CKPT=/lustre/fsw/portfolios/llmservice/users/ronaya/model_checkpoints/Ministral-3-1B-Instruct-2512-BF16_llm_replaced_nemotron_3_embed_1B
TEACHER_CKPT=/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ronaya/model_checkpoints/ministral3_8b_reranker_Gabriel143548_text_768k_frozenvision_lr1e4_08-27-041548/epoch_0_step_5399/model/consolidated

# Draco ##############################################################
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name ministral3_1b_reranker_distill_from_8b_kdt01_w05 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ronaya/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --yaml_config examples/retrieval/distillation/ministral3_vl_ranker_distill.yaml \
    --train_script ./scripts/distillation/train_distill_crossencoder.sh \
    --more_train_args " \
        --model.pretrained_model_name_or_path ${STUDENT_CKPT} \
        --tokenizer.pretrained_model_name_or_path ${STUDENT_CKPT} \
        --teacher_model.pretrained_model_name_or_path ${TEACHER_CKPT} \
        --optimizer.lr 5.0e-6 \
        --kd_temperature 0.1 \
        --kd_loss_weight 0.5 \
        --ce_loss_weight 0.5 \
    " \
    --eval_script ./scripts/yauhen/eval_reranker_rny.sh \
    --eval_datasets RERANK_TEXT_EVAL RERANK_VISION_EVAL \
    --results_dir_name eval_ranker_results_top50 \
    --more_eval_args " \
        --embedding_results embedding_results/ensemble_eagle_top_models_lines_2_101_weights_25_75_image.json \
        --multimodal True \
        --model_type mistral3_vl \
        --batch_size 8 \
        --max_length 10240 \
        --topk_candidates 50 \
        --log_json True \
        --log_wandb False \
        --skip_existing_results True \
    " \
    --num_training_rounds 1

# SFT-equivalence control: no teacher is built, loss is plain grouped cross-entropy.
# Use this to confirm the distillation recipe reproduces the existing baseline.
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name ministral3_1b_reranker_distill_ce_only_control \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ronaya/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --yaml_config examples/retrieval/distillation/ministral3_vl_ranker_distill.yaml \
    --train_script ./scripts/distillation/train_distill_crossencoder.sh \
    --more_train_args " \
        --model.pretrained_model_name_or_path ${STUDENT_CKPT} \
        --tokenizer.pretrained_model_name_or_path ${STUDENT_CKPT} \
        --optimizer.lr 5.0e-6 \
        --kd_loss_weight 0.0 \
        --ce_loss_weight 1.0 \
    " \
    --num_training_rounds 1
