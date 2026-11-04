#!/bin/bash

# Training experiments ##############################################################
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name oliver_stock_mr_stage_1_epoch_1_refit \
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
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name oliver_stock_mr_stage_1_epoch_2_refit \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_1.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_1_08-14-104819/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_1_08-14-104819/LATEST/model/consolidated \
        --optimizer.lr 3.0e-4 \
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

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name oliver_stock_mr_stage_2_refit \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/yauhen/gabriel_stage_2.yaml \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_2_08-14-123844/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/oliver_stock_mr_stage_1_epoch_2_08-14-123844/LATEST/model/consolidated \
        --optimizer.lr 1.0e-5 \
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

#############################################################

# Reranker

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name ministral3_1b_reranker_image_text_768k_export_refit_v2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --yaml_config examples/retrieval/cross_encoder/yauhen/ministral3_image_text_ranker_768k.yaml \
    --train_script ./scripts/yauhen/train_reranker.sh \
    --more_train_args " \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ronaya/model_checkpoints/ministral3_1b_jiauri_20260731_174359/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ronaya/model_checkpoints/ministral3_1b_jiauri_20260731_174359/consolidated \
        --optimizer.lr 5.0e-6 \
    " \
    --eval_script ./scripts/yauhen/eval_reranker.sh \
    --eval_datasets RERANK_VISION_EVAL RERANK_TEXT_EVAL \
    --results_dir_name eval_ranker_results_31_1B \
    --more_eval_args " \
        --multimodal True \
        --model_type mistral3_vl \
        --batch_size 8 \
        --max_length 10240 \
        --topk_candidates 50 \
        --log_wandb False \
    " \
    --num_training_rounds 1


# Eval
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name ministral3_1b_reranker_image_text_768k_eval_v2 \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_reranker.sh \
    --eval_datasets RERANK_VISION_EVAL RERANK_TEXT_EVAL \
    --results_dir_name eval_ranker_results_31_1B \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_1b_reranker_stage2_from_125136_text_768k_frozenvision_lr1e6_08-27-195326 \
        --multimodal True \
        --model_type mistral3_vl \
        --batch_size 8 \
        --max_length 10240 \
        --topk_candidates 50 \
        --log_wandb False \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name ministral3_1b_reranker_image_text_768k_eval_v2 \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_reranker.sh \
    --eval_datasets RERANK_LONG_EMBED \
    --results_dir_name eval_ranker_results_31_1B \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_1b_reranker_stage2_from_125136_text_768k_frozenvision_lr1e6_08-27-195326 \
        --multimodal True \
        --model_type mistral3_vl \
        --batch_size 8 \
        --max_length 10240 \
        --topk_candidates 50 \
        --log_wandb False \
    " \
    --num_training_rounds 0


# RERANK_VISION_EVAL RERANK_TEXT_EVAL RERANK_TEXT_PLUS_VISION_EVAL RERANK_RTEB_QUICK RERANK_LONG_EMBED
