#!/bin/bash

# # Llama-Nemotron-Embed-VL-1B-v2
# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_llama_nemotron_embed_vl_1b_v2 \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl.sh \
#   --eval_datasets MIRACL_VISION VISION_EVAL CHINESE_KOREAN_VIDORE_VISION \
#   --more_eval_args " \
#     --output_filename embedding_results/llama-nemotron-embed-vl-1b-v2_image.json \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/llama-nemotron-embed-vl-1b-v2 \
#     --model_type eagle_llama \
#     --max_input_tiles 6 \
#     --batch_size 16 \
#     --max_length 10240 \
#     --log_wandb False \
#   " \
#   --num_training_rounds 0

# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name text_eval_llama_nemotron_embed_vl_1b_v2 \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl.sh \
#   --eval_datasets TEXT_EVAL CHINESE_KOREAN_VIDORE_TEXT \
#   --more_eval_args " \
#     --output_filename embedding_results/llama-nemotron-embed-vl-1b-v2_text.json \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/llama-nemotron-embed-vl-1b-v2 \
#     --model_type eagle_llama \
#     --max_input_tiles 6 \
#     --batch_size 16 \
#     --max_length 10240 \
#     --log_wandb False \
#   " \
#   --num_training_rounds 0

# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_llama_nemotron_embed_vl_1b_v2 \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl.sh \
#   --eval_datasets CHINESE_KOREAN_VIDORE_TEXT_PLUS_VISION TEXT_PLUS_VISION_EVAL \
#   --more_eval_args " \
#     --output_filename embedding_results/llama-nemotron-embed-vl-1b-v2_image_text_complex_ocr.json \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/llama-nemotron-embed-vl-1b-v2 \
#     --model_type eagle_llama \
#     --max_input_tiles 6 \
#     --batch_size 16 \
#     --max_length 10240 \
#     --log_wandb False \
#   " \
#   --num_training_rounds 0

# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name text_eval_llama_nemotron_embed_vl_1b_v2 \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl.sh \
#   --eval_datasets RTEB_QUICK \
#   --more_eval_args " \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/llama-nemotron-embed-vl-1b-v2 \
#     --model_type eagle_llama \
#     --max_input_tiles 6 \
#     --batch_size 16 \
#     --max_length 10240 \
#     --log_wandb False \
#     --save_predictions \
#     --prediction_folder /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/embedding_predictions_llama_nemotron_embed_vl_1b_v2 \
#   " \
#   --num_training_rounds 0


# # Qwen3-VL-Embedding-2B
# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_qwen3_vl_embedding_2b \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
#   --eval_datasets MIRACL_VISION VISION_EVAL CHINESE_KOREAN_VIDORE_VISION \
#   --more_eval_args " \
#     --output_filename embedding_results/qwen3_vl_embedding_2b_image.json \
#       --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
#       --model_type qwen3_vl_embedding \
#       --batch_size 4 \
#       --max_length 10240 \
#       --attn_implementation flash_attention_2 \
#       --document_image_only true \
#       --query_instruction \"Find a screenshot that is relevant to the user's question.\" \
#       --document_instruction \"Represent the user's input.\" \
#       --torch_dtype bfloat16 \
#       --log_wandb False \
#   " \
#   --num_training_rounds 0


# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_qwen3_vl_embedding_2b \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
#   --eval_datasets TEXT_EVAL CHINESE_KOREAN_VIDORE_TEXT \
#   --more_eval_args " \
#     --output_filename embedding_results/qwen3_vl_embedding_2b_text.json \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
#     --model_type qwen3_vl_embedding \
#     --batch_size 4 \
#     --max_length 10240 \
#     --attn_implementation flash_attention_2 \
#     --document_image_only false \
#     --query_instruction \"Retrieve images or text relevant to the user's query.\" \
#     --document_instruction \"Represent the user's input.\" \
#     --torch_dtype bfloat16 \
#     --log_wandb False \
#   " \
#   --num_training_rounds 0


# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_qwen3_vl_embedding_2b \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
#   --eval_datasets CHINESE_KOREAN_VIDORE_TEXT_PLUS_VISION TEXT_PLUS_VISION_EVAL \
#   --more_eval_args " \
#     --output_filename embedding_results/qwen3_vl_embedding_2b_image_text_complex_ocr.json \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
#     --model_type qwen3_vl_embedding \
#     --batch_size 4 \
#     --max_length 10240 \
#     --attn_implementation flash_attention_2 \
#     --document_image_only false \
#     --query_instruction \"Retrieve images or text relevant to the user's query.\" \
#     --document_instruction \"Represent the user's input.\" \
#     --torch_dtype bfloat16 \
#     --log_wandb False \
#   " \
#   --num_training_rounds 0


# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_qwen3_vl_embedding_2b \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl_old.sh \
#   --eval_datasets RTEB_QUICK \
#   --more_eval_args " \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Qwen3-VL-Embedding-2B \
#     --model_type qwen3_vl_embedding \
#     --batch_size 4 \
#     --max_length 10240 \
#     --attn_implementation flash_attention_2 \
#     --document_image_only false \
#     --query_instruction \"Retrieve images or text relevant to the user's query.\" \
#     --document_instruction \"Represent the user's input.\" \
#     --torch_dtype bfloat16 \
#     --log_wandb False \
#     --save_predictions \
#     --prediction_folder /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/embedding_predictions_qwen3_vl_embedding_2b \
#   " \
#   --num_training_rounds 0


# Mistral 1B
python3 submit_nemo_run.py \
  --cluster draco \
  --experiment_name eval_31_model \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets MIRACL_VISION VISION_EVAL CHINESE_KOREAN_VIDORE_VISION \
  --more_eval_args " \
    --output_filename embedding_results/gabriel_31_1B_08-26-034305_image.json \
    --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_ministral3_1b_distill_8b_model_24_stage_2_08-26-034305 \
    --model_type mistral3 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
  " \
  --num_training_rounds 0

python3 submit_nemo_run.py \
  --cluster draco \
  --experiment_name eval_31_model \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets TEXT_EVAL CHINESE_KOREAN_VIDORE_TEXT \
  --more_eval_args " \
    --output_filename embedding_results/gabriel_31_1B_08-26-034305_text.json \
    --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_ministral3_1b_distill_8b_model_24_stage_2_08-26-034305 \
    --model_type mistral3 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
  " \
  --num_training_rounds 0

python3 submit_nemo_run.py \
  --cluster draco \
  --experiment_name eval_31_model \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets CHINESE_KOREAN_VIDORE_TEXT_PLUS_VISION TEXT_PLUS_VISION_EVAL \
  --more_eval_args " \
    --output_filename embedding_results/gabriel_31_1B_08-26-034305_image_text_complex_ocr.json \
    --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_ministral3_1b_distill_8b_model_24_stage_2_08-26-034305 \
    --model_type mistral3 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
  " \
  --num_training_rounds 0


python3 submit_nemo_run.py \
  --cluster draco \
  --partition interactive \
  --experiment_name eval_31_model \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets RTEB_QUICK LONG_EMBED \
  --more_eval_args " \
    --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_ministral3_1b_distill_8b_model_24_stage_2_08-26-034305 \
    --model_type mistral3 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
    --save_predictions \
    --prediction_folder /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/embedding_predictions_31_gabriel_08-26-034305 \
  " \
  --num_training_rounds 0




DATASET_ROOT=/lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/01_datasets/01_evaluation_datasets

find "$DATASET_ROOT" \
    -path "*/embedding_results/qwen3_vl_embedding_2b_text.json" \
    -type f | sort



# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/nemorun_ministral_8b_stage_2_full_lr_1e6_pass_4_image_08-24-222331

# RC1

# Mistral 1B
python3 submit_nemo_run.py \
  --cluster draco \
  --experiment_name eval_31_model \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets VISION_EVAL \
  --more_eval_args " \
    --output_filename embedding_results/rc1_nemotron_3_5_embed_image.json \
    --model_name_or_path nvidia/rc1-nemotron-3.5-embed \
    --model_type mistral3 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
  " \
  --num_training_rounds 0


python3 submit_nemo_run.py \
  --cluster draco \
  --experiment_name eval_31_model \
  --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
  --num_nodes 1 \
  --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
  --train_script "" \
  --eval_script ./scripts/yauhen/eval_script_vl.sh \
  --eval_datasets TEXT_EVAL \
  --more_eval_args " \
    --output_filename embedding_results/rc1_nemotron_3_5_embed_text.json \
    --model_name_or_path nvidia/rc1-nemotron-3.5-embed \
    --model_type mistral3 \
    --batch_size 16 \
    --max_length 10240 \
    --log_wandb False \
  " \
  --num_training_rounds 0

