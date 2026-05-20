#!/bin/bash

# Ministral 3B Instruct
python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_bidir_last_pooling_eager_t_004 \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh \
    --train_script ./scripts/yauhen/train_biencoder.sh \
    --yaml_config examples/retrieval/bi_encoder/ministral3b_instruct_release.yaml \
    --eval_script ./scripts/yauhen/eval_script_rteb.sh \
    --eval_datasets RTEB_QUICK \
    --more_train_args " \
        --optimizer.lr 5.0e-6 \
        --model.pooling last \
        --temperature 0.04 \
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

# Llama 3.2 1B
python3 submit_nemo_run.py \
    --experiment_name llama_1b_bidir_avg_pooling_eager \
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
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation eager \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps_05-15-215458/LATEST/model/consolidated \
    " \
    --num_training_rounds 0
