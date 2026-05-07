python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_bidirectional_last_explicit_config \
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
    --experiment_name llama_1b_bidirectional_avg_explicit_config \
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
        --attn_implementation flash_attention_2 \
        --model_name_or_path /lustre/fsw/portfolios/llmservice/users/ybabakhin/model_checkpoints/llama_1b_bidirectional_avg_poc_config_4_nodes_t_0_02_05-06-211552/epoch_0_step_4283/model/consolidated \
    " \
    --num_training_rounds 0
