#!/bin/bash

# MIRACL_VISION CHINESE_KOREAN_VIDORE image
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_2b \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets MIRACL_VISION CHINESE_KOREAN_VIDORE \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only true \
        --query_instruction \"Find a screenshot that is relevant to the user's question.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
        --corpus_modalities image \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_8b \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets MIRACL_VISION CHINESE_KOREAN_VIDORE \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-8B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only true \
        --query_instruction \"Find a screenshot that is relevant to the user's question.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
        --corpus_modalities image \
    " \
    --num_training_rounds 0


# CHINESE_KOREAN_VIDORE image + text
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_8b_text_image \
    --partition interactive \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets CHINESE_KOREAN_VIDORE \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-8B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only false \
        --query_instruction \"Retrieve images or text relevant to the user's query.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
        --corpus_modalities image text complex_ocr \
    " \
    --num_training_rounds 0

# CHINESE_KOREAN_VIDORE text
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_8b_text \
    --partition interactive \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets CHINESE_KOREAN_VIDORE \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-8B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only false \
        --query_instruction \"Retrieve images or text relevant to the user's query.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
        --corpus_modalities text complex_ocr \
    " \
    --num_training_rounds 0


# TEXT_PLUS_VISION_EVAL
python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_2b_image \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets VISION_EVAL \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only true \
        --query_instruction \"Find a screenshot that is relevant to the user's question.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_8b_image \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets VISION_EVAL \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-8B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only true \
        --query_instruction \"Find a screenshot that is relevant to the user's question.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
    " \
    --num_training_rounds 0


# Text evals
python3 submit_nemo_run.py \
    --cluster draco \
    --partition interactive \
    --experiment_name qwen3_2b_text \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets LONG_EMBED \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only false \
        --query_instruction \"Retrieve images or text relevant to the user's query.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
    " \
    --num_training_rounds 0


# TEXT_EVAL TEXT_PLUS_VISION_EVAL LONG_EMBED

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_8b_text \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --num_nodes 1 \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
    --eval_datasets TEXT_EVAL TEXT_PLUS_VISION_EVAL \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-8B \
        --model_type qwen3_vl_embedding \
        --batch_size 4 \
        --max_length 10240 \
        --attn_implementation flash_attention_2 \
        --document_image_only false \
        --query_instruction \"Retrieve images or text relevant to the user's query.\" \
        --document_instruction \"Represent the user's input.\" \
        --torch_dtype bfloat16 \
        --log_wandb False \
    " \
    --num_training_rounds 0


# Reranker evals

model_size=8B
model_tag=8b

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_vl_reranker_${model_tag}_eval \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_qwen3_vl_reranker.sh \
    --eval_datasets RERANK_VISION_EVAL RERANK_TEXT_EVAL RERANK_TEXT_PLUS_VISION_EVAL RERANK_RTEB_QUICK \
    --results_dir_name eval_ranker_results_31_1B \
    --more_eval_args " \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Reranker-${model_size} \
        --rerank_instruction \"Retrieve images or text relevant to the user's query.\" \
        --batch_size 2 \
        --max_length 10240 \
        --topk_candidates 50 \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --log_wandb False \
    "


model_size=8B
model_tag=8b

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_vl_reranker_${model_tag}_eval \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_qwen3_vl_reranker.sh \
    --eval_datasets RERANK_RTEB_QUICK \
    --results_dir_name eval_ranker_results_qwen3_vl_2b \
    --more_eval_args " \
        --embedding_results /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/embedding_predictions_qwen3_vl_embedding_2b \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Reranker-${model_size} \
        --rerank_instruction \"Retrieve images or text relevant to the user's query.\" \
        --batch_size 2 \
        --max_length 10240 \
        --topk_candidates 50 \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --log_wandb False \
    "

python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_vl_reranker_${model_tag}_eval \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_qwen3_vl_reranker.sh \
    --eval_datasets RERANK_VISION_EVAL \
    --results_dir_name eval_ranker_results_qwen3_vl_2b \
    --more_eval_args " \
        --embedding_results embedding_results/qwen3_vl_embedding_2b_image.json \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Reranker-${model_size} \
        --rerank_instruction \"Retrieve images or text relevant to the user's query.\" \
        --batch_size 2 \
        --max_length 10240 \
        --topk_candidates 50 \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --log_wandb False \
    "


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_vl_reranker_${model_tag}_eval \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_qwen3_vl_reranker.sh \
    --eval_datasets RERANK_TEXT_EVAL \
    --results_dir_name eval_ranker_results_qwen3_vl_2b \
    --more_eval_args " \
        --embedding_results embedding_results/qwen3_vl_embedding_2b_text.json \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Reranker-${model_size} \
        --rerank_instruction \"Retrieve images or text relevant to the user's query.\" \
        --batch_size 2 \
        --max_length 10240 \
        --topk_candidates 50 \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --log_wandb False \
    "


python3 submit_nemo_run.py \
    --cluster draco \
    --experiment_name qwen3_vl_reranker_${model_tag}_eval \
    --num_nodes 1 \
    --num_training_rounds 0 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
    --train_script "" \
    --eval_script ./scripts/yauhen/eval_qwen3_vl_reranker.sh \
    --eval_datasets RERANK_TEXT_PLUS_VISION_EVAL \
    --results_dir_name eval_ranker_results_qwen3_vl_2b \
    --more_eval_args " \
        --embedding_results embedding_results/qwen3_vl_embedding_2b_image_text_complex_ocr.json \
        --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Reranker-${model_size} \
        --rerank_instruction \"Retrieve images or text relevant to the user's query.\" \
        --batch_size 2 \
        --max_length 10240 \
        --topk_candidates 50 \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --log_wandb False \
    "

