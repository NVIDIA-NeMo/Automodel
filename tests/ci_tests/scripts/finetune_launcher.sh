# Test variables
CONFIG="--config /opt/Automodel/${CONFIG_PATH} \
        --checkpoint.checkpoint_dir $PIPELINE_DIR/$TEST_NAME/checkpoint"

# Configure local batch size
if [[ -n "$LOCAL_BATCH_SIZE" ]]; then
  CONFIG="${CONFIG} \
         --step_scheduler.local_batch_size ${LOCAL_BATCH_SIZE}"
fi

# For convergence runs
if [ "$TEST_LEVEL" = "convergence" ]; then
  export WANDB_API_KEY="${WANDB_AUTOMODEL_API_KEY}"
  export TEST_DATE=$(date +%Y%m%d)
  CONFIG="${CONFIG} \
         --step_scheduler.ckpt_every_steps 200 \
         --step_scheduler.max_steps 200 \
         --step_scheduler.val_every_steps 200 \
         --wandb.project automodel-nemo-ci-convergence-test-${TEST_DATE} \
         --wandb.entity Nemo-automodel \
         --wandb.name ${TEST_NAME} \
         --wandb.dir /tmp/wandb/"
else
  CONFIG="${CONFIG} \
        --step_scheduler.ckpt_every_steps 100 \
        --step_scheduler.max_steps ${MAX_STEPS:-100} \
        --step_scheduler.val_every_steps 100"
fi

# Command to execute, defaults to torchrun
CMD="torchrun --nproc-per-node=${NPROC_PER_NODE} \
              --nnodes=${TEST_NODE_COUNT} \
              --rdzv_backend=c10d \
              --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
              --rdzv_id=${SLURM_JOB_ID}"
if [ "$EXEC_CMD" = "python" ]; then CMD="python"; fi
if [ "$EXEC_CMD" = "uv_python" ]; then CMD="uv run python"; fi

cd /opt/Automodel
RUN_CMD="${CMD} ${TEST_SCRIPT_PATH} ${CONFIG} ${FINETUNE_ARGS}"
eval $RUN_CMD
