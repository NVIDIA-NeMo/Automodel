#!/bin/bash

# # Ministral 3B Instruct
# python3 submit_nemo_run.py \
#     --experiment_name ministral3_3b_bidir_last_pooling_eager_t_004 \
#     --num_nodes 4 \
#     --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
#     --eval_script ./scripts/yauhen/eval_script_rteb.sh \
#     --eval_datasets RTEB_QUICK \
#     --more_train_args " \
#         --optimizer.lr 2.0e-6 \
#         --model.pooling last \
#     " \
#     --more_eval_args " \
#         --max_length 4096 \
#         --batch_size 4 \
#         --pooling last \
#         --padding_side left \
#         --precision bf16 \
#         --attn_implementation eager \
#     " \
#     --num_training_rounds 1

# # Llama 3.2 1B
# python3 submit_nemo_run.py \
#     --experiment_name llama_1b_bidir_avg_pooling_eager \
#     --num_nodes 4 \
#     --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_release.yaml \
#     --eval_script ./scripts/yauhen/eval_script_rteb.sh \
#     --eval_datasets RTEB_QUICK \
#     --more_train_args " \
#         --optimizer.lr 5.0e-6 \
#         --model.pooling avg \
#     " \
#     --more_eval_args " \
#         --max_length 4096 \
#         --batch_size 4 \
#         --pooling avg \
#         --padding_side left \
#         --precision bf16 \
#         --attn_implementation eager \
#     " \
#     --num_training_rounds 1

# Ministral 3B Instruct New Baseline
python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_change_sql_dataset_rm_swe_bench \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_change_sql_dataset_rm_swe_bench_default_rm_accord \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --model.attn_implementation eager \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name FT_ministral_3b_49th_with_22nd_mix \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_5e_6_t_002_mtp_v3_instruct_05-31-132304/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_5e_6_t_002_mtp_v3_instruct_05-31-132304/LATEST/model/consolidated \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_bidir_last_pooling_eager_new_baseline_bs1 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
        --step_scheduler.local_batch_size 1 \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name llama_1b_bidir_avg_pooling_eager_new_baseline_indomain \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name FT_llama_1b_11M_fixed_mlqa_paq_inbatch_neg_mask_new_mix \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/llama_1b_pretrain_11M_fixed_mlqa_paq_inbatch_neg_mask_05-25-110159/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/llama_1b_pretrain_11M_fixed_mlqa_paq_inbatch_neg_mask_05-25-110159/LATEST/model/consolidated \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name llama_1b_bidir_avg_pooling_eager_mtp_v2_pretrain \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/llama_1b_mtp_v2_11M_pretrain_05-23-105239/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/llama_1b_mtp_v2_11M_pretrain_05-23-105239/LATEST/model/consolidated \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1


# PRE-TRAIN STAGE 2.

python3 submit_nemo_run.py \
    --experiment_name llama_1b_bidir_avg_pooling_11M_new_pretrain \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 3

python3 submit_nemo_run.py \
    --experiment_name llama_1b_pretrain_11M_fixed_mlqa_paq_eval \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 2

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_1e_5 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 1.0e-5 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_3e_6 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 3.0e-6 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_5e_6_t_002 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --temperature 0.02 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
        --lr_scheduler.lr_warmup_steps 50 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_50M_in_batch_negs_bs_4 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --temperature 0.02 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
        --lr_scheduler.lr_warmup_steps 1000 \
        --step_scheduler.local_batch_size 4 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 5


python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_5e_6_t_002_in_batch_negs \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --temperature 0.02 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
        --lr_scheduler.lr_warmup_steps 50 \
        --model.do_distributed_inbatch_negative true \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_5e_6_t_002_mtp_v3_instruct \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --temperature 0.02 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
        --lr_scheduler.lr_warmup_steps 50 \
        --model.pooling avg \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_ministral3b/ministral3-mtp-datamix6-gbz1280lbz20-lr5e6-4node-maskratio015-random-instruct_05-29-181116/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_ministral3b/ministral3-mtp-datamix6-gbz1280lbz20-lr5e6-4node-maskratio015-random-instruct_05-29-181116/LATEST/model/consolidated \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --experiment_name llama_1b_pretrain_11M_fixed_mlqa_paq_inbatch_neg_mask \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
        --model.do_distributed_inbatch_negative true \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 2


python3 submit_nemo_run.py \
    --experiment_name llama_1b_mtp_v1_11M_pretrain \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_inline_parquet_multi_ds.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_llama1b/llama1b-mtp-pretrain6M-gbz2048lbz32-lr5e6-4node-mask01_05-08-010228/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_llama1b/llama1b-mtp-pretrain6M-gbz2048lbz32-lr5e6-4node-mask01_05-08-010228/LATEST/model/consolidated \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 3



# Ministral 8B Instruct
python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_bidir_last_pooling_eager_nq \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_codex_suggestions \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 4000 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 2

python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_fa2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 4000 \
        --model.attn_implementation flash_attention_2 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 2


python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_lr_3e_6 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 3.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 4000 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 2

python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_lr_1_5e_6 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 1.5e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 4000 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
    " \
    --num_training_rounds 2


# Evaluation
python3 submit_nemo_run.py \
    --experiment_name eval \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_lr_3e_6_05-26-214718/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


# RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
python3 submit_nemo_run.py \
    --experiment_name eval_old_text \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/release_models_01_06/llama-nemotron-embed-1b-v2 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_3b_ensemble \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path  \
  /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_fa2_slerp_0p50_05-31-083647_083847/epoch_0_step_slerp/model/consolidated  " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_FA2_8B \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_05-27-225523/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_FA2_llama_codex \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/llama_1b_bidir_avg_rteb_default_refresh_65only_20260530_05-30-130008/LATEST/model/consolidated \
    " \
    --num_training_rounds 0
# Stage 2. evals

python3 submit_nemo_run.py \
    --experiment_name eval_FA2_59 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/FT_ministral_3b_59th_with_22nd_mix_05-31-132606/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_FA2_49 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/FT_ministral_3b_49th_with_22nd_mix_06-01-084224/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

# /lustre/fsw/portfolios/llmservice/users/ronaya/distillation_ckps/rteb-mse-w1-new-gbs256-8nodes-hardnegs-lr1e4-tnp10/step_3311/student

python3 submit_nemo_run.py \
    --experiment_name eval_FA2_distill_2b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/rteb-mse-w1-new-gbs256-8nodes-hardnegs-lr1e4-tnp10 \
    " \
    --num_training_rounds 0


# RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL
python3 submit_nemo_run.py \
    --experiment_name eval_llama_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/release_models_01_06/llama-embed-nemotron-8b \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_ronay_1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/automodel_ministral3_2b_rteb_blend_projection_sync_fix_20260530 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_ronay_2 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_pruned_rank_gbs256_t002_lr2e_6_05-28-163449 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_padding_right \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side right \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_default_padding_side_06-03-174932/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_padding_right \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side right \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_default_padding_side_06-03-174932/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_padding_right_old \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side right \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_05-27-105407/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


