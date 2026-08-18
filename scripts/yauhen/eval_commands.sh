#!/bin/bash

# MAIN_EVAL: VISION_EVAL, TEXT_EVAL, RTEB_QUICK
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

# TEXT_PLUS_VISION_EVAL: (Image + Text) DC10k, Earnings-v2, Vidore-v3
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


# Other models ##############################################################
# # Nemotron-3-Embed-1B
# python3 submit_nemo_run.py \
#     --cluster draco \
#     --partition interactive \
#     --experiment_name eval_nemotron3_embed_1b \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --num_nodes 1 \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#     --train_script "" \
#     --eval_script ./scripts/yauhen/eval_script_vl.sh \
#     --eval_datasets TEXT_EVAL RTEB_QUICK \
#     --more_eval_args " \
#         --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/Nemotron-3-Embed-1B-BF16 \
#         --model_type HF \
#         --embedding_dim 2048 \
#         --batch_size 16 \
#         --max_length 10240 \
#         --precision bf16 \
#         --attn_implementation flash_attention_2 \
#         --padding_side left \
#         --pooling avg \
#         --normalize_embeddings True \
#     " \
#     --num_training_rounds 0

# # Llama-Nemotron-Embed-VL-1B-v2
# python3 submit_nemo_run.py \
#   --cluster draco \
#   --experiment_name eval_llama_nemotron_embed_vl_1b_v2 \
#   --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#   --num_nodes 1 \
#   --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh \
#   --train_script "" \
#   --eval_script ./scripts/yauhen/eval_script_vl.sh \
#   --eval_datasets VISION_EVAL TEXT_EVAL RTEB_QUICK \
#   --more_eval_args " \
#     --model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/llama-nemotron-embed-vl-1b-v2 \
#     --model_type eagle_llama \
#     --max_input_tiles 6 \
#     --batch_size 16 \
#     --max_length 10240 \
#     --log_wandb False \
#   " \
#   --num_training_rounds 0


