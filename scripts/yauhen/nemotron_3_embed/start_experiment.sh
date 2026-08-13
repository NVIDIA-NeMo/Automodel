#!/bin/bash

# Ministral 8B Instruct
python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_65_datasets_multilingual_finepdf_x2_filtered_unrolled \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 6300 \
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
    --experiment_name ministral3_8b_stage_1_65_datasets_multilingual_finepdf \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_ministral8b/ministral3-8b-mtp-datamix6-gbz1280lbz10-lr5e6-8node-maskratio015-random-instruct_05-30-165450/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_ministral8b/ministral3-8b-mtp-datamix6-gbz1280lbz10-lr5e6-8node-maskratio015-random-instruct_05-30-165450/LATEST/model/consolidated \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 4500 \
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
    --experiment_name FT_ministral_8b_17M_with_22nd_mix_lr_2e_6 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 4500 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
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
    --experiment_name ministral3_8b_68_datasets_aila_itself_mlqa_finepdfs_x2_from_scratch_100k_filtered \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_0_shot.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 6300 \
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


# export NEMO_RUN_REMOTE_CODE_SYMLINK=/lustre/fs12/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/git/automodel-private

python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_rteb_target_slim \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_rteb_slim.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 1.5e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 6100 \
        --seed 24 \
        --dataloader.dataset.seed 24 \
        --validation_dataloader.dataset.seed 24 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
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
    --experiment_name ministral3_8b_rteb_target_6_HNs_FT_17M_07-10-230406 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_rteb_08_07.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 1.7e-6 \
        --model.pooling avg \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 4800 \
        --dataloader.dataset.n_passages 7 \
        --seed 1333 \
        --dataloader.dataset.seed 1333 \
        --validation_dataloader.dataset.seed 1333 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1 \
    --no_append_timestamp


python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_final_from_base_code_v2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_teacher_slim.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 1.5e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 3000 \
        --seed 730 \
        --dataloader.dataset.seed 730 \
        --validation_dataloader.dataset.seed 730 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 4


python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_final_FT_from_17M_v2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_teacher.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 1.5e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 2950 \
        --seed 240 \
        --dataloader.dataset.seed 240 \
        --validation_dataloader.dataset.seed 240 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
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
    --experiment_name ministral3_8b_rteb_target_FT_from_50M \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_rteb_target.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 3e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 6170 \
        --seed 145 \
        --dataloader.dataset.seed 145 \
        --validation_dataloader.dataset.seed 145 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_50M_in_batch_negs_06-06-104111/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_50M_in_batch_negs_06-06-104111/LATEST/model/consolidated \
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




# python3 submit_nemo_run.py \
#     --experiment_name ministral3_8b_teacher_FT_from_17M_28_06 \
#     --num_nodes 8 \
#     --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
#     --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
#     --train_script ./scripts/yauhen/train_biencoder.sh \
#     --yaml_config examples/retrieval/bi_encoder/ministral3b_teacher.yaml \
#     --eval_script ./scripts/yauhen/eval_script_rteb.sh \
#     --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
#     --more_train_args " \
#         --optimizer.lr 1.5e-6 \
#         --model.pooling avg \
#         --step_scheduler.local_batch_size 2 \
#         --step_scheduler.ckpt_every_steps 5600 \
#         --seed 123 \
#         --dataloader.dataset.seed 123 \
#         --validation_dataloader.dataset.seed 123 \
#         --model.extract_submodel none \
#         --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
#         --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
#     " \
#     --more_eval_args " \
#         --max_length 4096 \
#         --batch_size 4 \
#         --pooling avg \
#         --padding_side left \
#         --precision bf16 \
#         --attn_implementation flash_attention_2 \
#     " \
#     --num_training_rounds 5


python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_teacher_FT_from_17M_all_datasets_1k \
    --num_nodes 16 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_teacher.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 1.5e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 1 \
        --step_scheduler.ckpt_every_steps 5000 \
        --seed 13 \
        --dataloader.dataset.seed 13 \
        --validation_dataloader.dataset.seed 13 \
        --dataloader.collate_fn.q_max_len 1024 \
        --dataloader.collate_fn.p_max_len 1024 \
        --validation_dataloader.collate_fn.q_max_len 1024 \
        --validation_dataloader.collate_fn.p_max_len 1024 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312/LATEST/model/consolidated \
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
    --experiment_name ministral3_3b_parent_FT_from_17M_seed_15_28_06 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_teacher.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
    --more_train_args " \
        --optimizer.lr 2e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 5600 \
        --seed 15 \
        --dataloader.dataset.seed 15 \
        --validation_dataloader.dataset.seed 15 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_pretrain_17M_in_batch_negs_06-04-164207/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_pretrain_17M_in_batch_negs_06-04-164207/LATEST/model/consolidated \
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
    --experiment_name ministral3_8b_68_datasets_finepdf_x2_unrolled_mlqa_aila_statutes_itself_FT_from_50M_seed_123 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_0_shot.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 6200 \
        --seed 123 \
        --dataloader.dataset.seed 123 \
        --validation_dataloader.dataset.seed 123 \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_50M_in_batch_negs_06-06-104111/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_pretrain_50M_in_batch_negs_06-06-104111/LATEST/model/consolidated \
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
    --experiment_name ministral3_8b_65_datasets_zero_shot_3_new_codex_finepdf_multilingual \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_0_shot.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 6300 \
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
    --experiment_name ministral_3b_65_datasets_zero_shot_3_new_codex_finepdf_x2_unrolled_mlqa_06-13-132755 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_0_shot.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 8000 \
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
    --num_training_rounds 1 \
    --no_append_timestamp


# Ministral 3B Instruct New Baseline
python3 submit_nemo_run.py \
    --experiment_name ministral_3b_65_datasets_multilingual_finepdf_x2_filtered \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --experiment_name ministral_3b_balanced \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_balanced.yaml \
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
    --experiment_name ministral3_3b_stage_1_65_datasets_multilingual_finepdf \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release_new_baseline.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
        --model.extract_submodel none \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_ministral3b/ministral3-mtp-datamix6-gbz1280lbz20-lr5e6-4node-maskratio015-random-instruct_05-29-181116/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints/pretrain_ministral3b/ministral3-mtp-datamix6-gbz1280lbz20-lr5e6-4node-maskratio015-random-instruct_05-29-181116/LATEST/model/consolidated \
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
    --experiment_name ministral_3b_65_datasets_zero_shot \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_0_shot.yaml \
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
    --experiment_name FT_ministral_3b_50M_with_22nd_mix_lr_2e_6 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_pretrain_50M_in_batch_negs_bs_4_06-03-140858/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_pretrain_50M_in_batch_negs_bs_4_06-03-140858/LATEST/model/consolidated \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/llama_1b_pretrain_11M_fixed_mlqa_paq_inbatch_neg_mask_05-25-110159/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/llama_1b_pretrain_11M_fixed_mlqa_paq_inbatch_neg_mask_05-25-110159/LATEST/model/consolidated \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/llama3_2_1b_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling avg \
        --model.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/llama_1b_mtp_v2_11M_pretrain_05-23-105239/LATEST/model/consolidated \
        --tokenizer.pretrained_model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/llama_1b_mtp_v2_11M_pretrain_05-23-105239/LATEST/model/consolidated \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --experiment_name ministral3_3b_pretrain_50M_in_batch_negs_bs_4_06-03-140858 \
    --num_nodes 8 \
    --account datascience_nemo_retriever \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --num_training_rounds 6 \
    --no_append_timestamp

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_17M_in_batch_negs \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_sample.yaml \
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
    --num_training_rounds 6

python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_pretrain_17M_in_batch_negs_06-05-175312 \
    --num_nodes 8 \
    --account datascience_nemo_retriever \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_inline_parquet_sample.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --temperature 0.02 \
        --model.pooling avg \
        --step_scheduler.global_batch_size 1024 \
        --lr_scheduler.lr_warmup_steps 1000 \
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 600 \
        --step_scheduler.val_every_steps 100000 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 6 \
    --no_append_timestamp


python3 submit_nemo_run.py \
    --experiment_name ministral3_8b_pretrain_50M_in_batch_negs_06-06-104111 \
    --num_nodes 8 \
    --account datascience_nemo_retriever \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --tokenizer.pretrained_model_name_or_path mistralai/Ministral-3-8B-Instruct-2512 \
        --step_scheduler.local_batch_size 2 \
        --step_scheduler.ckpt_every_steps 600 \
        --step_scheduler.val_every_steps 100000 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 5 \
    --no_append_timestamp




python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_pretrain_11M_fixed_mlqa_paq_eval_gbs_1024_lr_5e_6_t_002_mtp_v3_instruct \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --experiment_name ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_fa2 \
    --num_nodes 8 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_lr_3e_6_05-26-214718/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


# RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL \
python3 submit_nemo_run.py \
    --experiment_name eval_old_text \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/release_models_01_06/llama-nemotron-embed-1b-v2 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_3b_ensemble \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
  /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_fa2_slerp_0p50_05-31-083647_083847/epoch_0_step_slerp/model/consolidated  " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_FA2_8B \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_new_baseline_coir_indomain_codex_indomain_8_nodes_eval_avg_pooling_sub_sql_05-27-225523/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_FA2_llama_codex \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/llama_1b_bidir_avg_rteb_default_refresh_65only_20260530_05-30-130008/LATEST/model/consolidated \
    " \
    --num_training_rounds 0
# Stage 2. evals

python3 submit_nemo_run.py \
    --experiment_name eval_FA2_59 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/FT_ministral_3b_59th_with_22nd_mix_05-31-132606/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_FA2_49 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/FT_ministral_3b_49th_with_22nd_mix_06-01-084224/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

# /lustre/fsw/portfolios/llmservice/users/ronaya/distillation_ckps/rteb-mse-w1-new-gbs256-8nodes-hardnegs-lr1e4-tnp10/step_3311/student

python3 submit_nemo_run.py \
    --experiment_name eval_FA2_distill_2b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb-mse-w1-new-gbs256-8nodes-hardnegs-lr1e4-tnp10 \
    " \
    --num_training_rounds 0


# RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL
python3 submit_nemo_run.py \
    --experiment_name eval_llama_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/release_models_01_06/llama-embed-nemotron-8b \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_ronay_1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_pruned_rank_gbs256_t002_lr2e_6_05-28-163449 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_padding_right \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_default_padding_side_06-03-174932/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_padding_right \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_change_sql_dataset_rm_swe_bench_default_padding_side_06-03-174932/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_ronay_15b \
    --num_nodes 1 \
    --account datascience_nemo_retriever \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb-mse-w1-gbs512-distill-ministral1p5b_lr1e4_060715 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_vinay_13b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/target_1p3b_domain_embedding_cka \
    " \
    --num_training_rounds 0


# 5 iters
python3 submit_nemo_run.py \
    --experiment_name eval_8b_multilingual_finepdf \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_65_datasets_multilingual_finepdf_06-06-214049/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_8b_base_multilingual_finepdf \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_base_65_datasets_multilingual_finepdf_06-06-215652/LATEST/model/consolidated \
    " \
    --num_training_rounds 0



python3 submit_nemo_run.py \
    --experiment_name eval_8b_0_shot \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_65_datasets_zero_shot_06-06-214330/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_ensemble_v6 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_multislerp_karcher_mean_06-05-163204_05-27-105407_06-06-135319_weights_0_25_0_50_0_25/LATEST/model/consolidated \
    " \
    --num_training_rounds 0






python3 submit_nemo_run.py \
    --experiment_name eval_rteb_90_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_65_datasets_zero_shot_wikisql_finqa_order_target_top_3_06-09-213738/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_13b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3-1_3B-VR-06-08-2026 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_instruct_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_65_datasets_zero_shot_wikisql_finqa_order_target_top_3_instructions_06-11-170227/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_instruct_3b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets vidore_v3_complex_ocr  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral_3b_65_datasets_zero_shot_wikisql_finqa_order_target_top_3_instructions_06-11-170512/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_8b_ensemble \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr  \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_slerp_ensemble_06-12-220426_06-12-215552_alpha_0_5/LATEST/model/consolidated \
    " \
    --num_training_rounds 0



python3 submit_nemo_run.py \
    --experiment_name eval_8b_best_4096_bs_8 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_all_extra_seed_13_06-17-211115/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_8b_teacher_mmteb \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_PRIVATE_LOCAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_8b_teacher_long_embed_16384 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 16384 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_8b_teacher_long_embed_8192 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 8192 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_8b_teacher_long_embed_32768 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0



python3 submit_nemo_run.py \
    --experiment_name eval_8b_teacher_mteb \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_8b_teacher_text_emb \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets text_embedding_eval \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_teacher_FT_from_17M_28_06_06-28-204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_3b_parent_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_all_extra_seed_15_06-28-153956/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_3b_parent_v2_mmteb \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MMTEB_RETRIEVAL_QUICK  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_parent_FT_from_17M_seed_15_28_06_06-28-160026/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_3b_parent_v2_mteb \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MTEB_ENG_RETRIEVAL  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_3b_parent_FT_from_17M_seed_15_28_06_06-28-160026/LATEST/model/consolidated \
    " \
    --num_training_rounds 0



python3 submit_nemo_run.py \
    --experiment_name eval_8b_ens_4096_bs_8 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB  \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_slerp_ensemble_06-17-211115_06-17-212000_alpha_0_5/LATEST/model/consolidated \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_1b_new_code \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/Nemotron-3-Embed-1B-BF16-NIM-test-July-2nd \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_1b_distill \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ronay \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_student_jiarui_new_code \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/jiarui_4096_04_07 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_v1_long \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/jiarui_4096_04_07 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_student_v2 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_5e5_lr_07-04-081301/step_7330/student \
    " \
    --num_training_rounds 0



python3 submit_nemo_run.py \
    --experiment_name rteb_ens_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rtebquick_multislerp_equal_all_extra_teacher15_teacher28_0617211115_0625114306_0628204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


# export NEMO_RUN_REMOTE_CODE_SYMLINK=/lustre/fs12/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/git/automodel-private


python3 submit_nemo_run.py \
    --experiment_name best_ensemble_new_code_bs_4 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB \
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
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/Nemotron-3-Embed-8B-BF16 \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name rteb_32768_1b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/Nemotron-3-Embed-1B-BF16 \
    " \
    --num_training_rounds 0


# RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL
python3 submit_nemo_run.py \
    --experiment_name ensemble_v13 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v12_0712191040/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name swebench_4096 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets SWEbenchCodeRetrieval \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_cache_chunk_size 1024 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v12_0712191040/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name longembed_16384 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 16384 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v12_0712191040/LATEST/model/consolidated \
    " \
    --num_training_rounds 0



# SWEbenchCodeRetrieval -- 2048
# LongEmbed -- 16384
# LongEmbed -- 32768

# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v1_0712113457/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v1_average_0712114416/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v1_multislerp_0712131210/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v2_0712125452/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v3_0712181849/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v5_0712182601/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v7_0712182859/LATEST/model/consolidated
# v8: /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v7_0712185519/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v9_0712190200/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v10_0712190555/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v11_0712191519/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v12_0712191040/LATEST/model/consolidated
# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ensemble/ensemble_v13_0712193127/LATEST/model/consolidated

python3 submit_nemo_run.py \
    --experiment_name rteb_ens_5 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets MMTEB_RETRIEVAL_QUICK \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_multislerp_karcher_mean_teacher28_rtebtarget_ft17m_rtebtarget_slim_v3_0628204743_0707143433_0703195935_0710223809_equal_weights_0711073114/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name rteb_ens_2 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_slerp_ensemble_06-28-204743_07-07-143433_equal_weights/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


# RTEB vidore_v3_complex_ocr



# RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL

# 3B Ministral 
# # 1 round: RTEB + VIDORE + TEXT_EMBED + start MMTEB
# # 2 round: done

# 8B Ministral 
# # 1 round: RTEB + VIDORE + start TEXT_EMBED
# # 2 round: RTEB + VIDORE + TEXT_EMBED + start MMTEB
# # 3 round: RTEB + VIDORE + TEXT_EMBED + MMTEB + start MTEB
# 4 rounds?
# # 5 round: RTEB + VIDORE + TEXT_EMBED + MMTEB + running MTEB (Touche is Missing)


python3 submit_nemo_run.py \
    --experiment_name eval_student_256_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 256 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_07-04-075159/step_7330/student \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_long_256_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 256 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_07-04-075159/step_7330/student \
    " \
    --num_training_rounds 0


python3 submit_nemo_run.py \
    --experiment_name eval_student_long_256_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 256 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_07-04-075159/step_7330/student \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_long_1024_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 1024 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_07-04-075159/step_7330/student \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_long_2048_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_07-04-075159/step_7330/student \
    " \
    --num_training_rounds 0




# /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_mrl_score_07-05-054035/step_7330/student

python3 submit_nemo_run.py \
    --experiment_name eval_student_256_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
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
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 256 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_mrl_score_07-05-054035/step_7330/student \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_long_256_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 16384 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 256 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_mrl_score_07-05-054035/step_7330/student \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_long_256_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 256 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_mrl_score_07-05-054035/step_7330/student \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name eval_student_long_1024_slice \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 1024 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/rteb_distill_v1_2048_1e4_lr_mrl_score_07-05-054035/step_7330/student \
    " \
    --num_training_rounds 0


###### ENSEMBLE MODELS ######

python3 submit_nemo_run.py \
    --experiment_name rteb_target_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB MMTEB_RETRIEVAL_QUICK RTEB_PRIVATE_LOCAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rteb_target_07-03-195935/LATEST/model/consolidated \
    " \
    --num_training_rounds 0



python3 submit_nemo_run.py \
    --experiment_name rteb_ens_8b \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL RTEB_PRIVATE_LOCAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ministral3_8b_rtebquick_multislerp_equal_all_extra_teacher15_teacher28_0617211115_0625114306_0628204743/LATEST/model/consolidated \
    " \
    --num_training_rounds 0


# FINAL EVALS

python3 submit_nemo_run.py \
    --experiment_name ensemble_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/ensemble_v1 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name ensemble_v2 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling avg \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/ensemble_v2 \
    " \
    --num_training_rounds 0









python3 submit_nemo_run.py \
    --experiment_name yauhen_4k_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL RTEB_PRIVATE_LOCAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/yauhen_4k \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name yauhen_4k_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 16384 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/yauhen_4k \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name yauhen_4k_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/yauhen_4k \
    " \
    --num_training_rounds 0




python3 submit_nemo_run.py \
    --experiment_name ens_v4_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL RTEB_PRIVATE_LOCAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/ensemble_v4 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name ens_v4_v1 \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/ensemble_v4 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name best \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB vidore_v3_complex_ocr text_embedding_eval MMTEB_RETRIEVAL_QUICK MTEB_ENG_RETRIEVAL RTEB_PRIVATE_LOCAL \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 8 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/Nemotron-3-Embed-1B-BF16 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name best_long \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 16384 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/Nemotron-3-Embed-1B-BF16 \
    " \
    --num_training_rounds 0

python3 submit_nemo_run.py \
    --experiment_name best_long_32k \
    --num_nodes 1 \
    --checkpoint_directory /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets LONGEMBED \
    --more_train_args " \
        --optimizer.lr 2.0e-6 \
        --model.pooling last \
    " \
    --more_eval_args " \
        --max_length 32768 \
        --batch_size 1 \
        --pooling avg \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
        --embedding_dim 2048 \
        --model_name_or_path /lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/model_checkpoints/ga_release/Nemotron-3-Embed-1B-BF16 \
    " \
    --num_training_rounds 0

