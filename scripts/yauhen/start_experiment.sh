python3 submit_nemo_run.py \
    --experiment_name ministral3_3b_bidirectional_last_pooling_lr_2e-6_26_04 \
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
