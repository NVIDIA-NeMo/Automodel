python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps \
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
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps_eager \
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
        --model.attn_implementation eager \
        --model.use_sdpa_patching false \
        --model.use_liger_kernel false \
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
    --experiment_name ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps_fa2_no_kernels \
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
        --model.attn_implementation flash_attention_2 \
        --model.use_sdpa_patching false \
        --model.use_liger_kernel false \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1



python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps_temperature_0_04 \
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
        --temperature 0.04 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1

python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps_temperature_0_04 \
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
        --temperature 0.04 \
    " \
    --more_eval_args " \
        --max_length 4096 \
        --batch_size 4 \
        --pooling last \
        --padding_side left \
        --precision bf16 \
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1



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
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/ministral3_3b_true_bidirectional_last_explicit_config_fresh_pull_tokenizer_arch_5_5_0_deps_05-15-215458/LATEST/model/consolidated \
    " \
    --num_training_rounds 0




python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_real_bidirectional_last_explicit_config_fresh_pull_tokenizer_25_11_deps \
    --num_nodes 4 \
    --checkpoint_directory /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints \
    --container_image /lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-25.11.00.sqsh \
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
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1


python3 submit_nemo_run.py \
    --experiment_name llama_1b_bidirectional_avg_explicit_config_fresh_pull_arch_5_5_0_deps \
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
        --attn_implementation flash_attention_2 \
    " \
    --num_training_rounds 1
