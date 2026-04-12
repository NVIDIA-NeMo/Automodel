# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
elif [ "$TEST_LEVEL" = "perf" ]; then
  CONFIG="${CONFIG}"
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

# Checkpoint robustness variables
ROBUSTNESS_COMMON="--config /opt/Automodel/${CONFIG_PATH} \
  --checkpoint.checkpoint_dir $PIPELINE_DIR/$TEST_NAME/robustness_checkpoint \
  --checkpoint.enabled true \
  --checkpoint.model_save_format safetensors \
  --checkpoint.save_consolidated true \
  --step_scheduler.max_steps 5 \
  --step_scheduler.ckpt_every_steps 5 \
  --step_scheduler.val_every_steps 5 \
  --step_scheduler.global_batch_size 32 \
  --step_scheduler.local_batch_size 2"

if [[ "${CONFIG_PATH}" == *peft* ]] || [[ "${CONFIG_PATH}" == *lora* ]]; then
  ROBUSTNESS_COMMON="${ROBUSTNESS_COMMON} --peft.use_triton false"
fi

ROBUSTNESS_CMD="${CMD} --tee 3 --log-dir $PIPELINE_DIR/$TEST_NAME/robustness_logs \
  -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
  ${ROBUSTNESS_COMMON}"

# --- Finetune ---
cd /opt/Automodel
RUN_CMD="${CMD} ${TEST_SCRIPT_PATH} ${CONFIG} ${FINETUNE_ARGS}"
echo "============================================"
echo "[finetune] Running finetune..."
echo "============================================"
FINETUNE_START=$SECONDS

eval $RUN_CMD
FINETUNE_EXIT_CODE=$?

FINETUNE_ELAPSED=$((SECONDS - FINETUNE_START))
echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"finetune\",\"seconds\":${FINETUNE_ELAPSED}}" >> $PIPELINE_DIR/$TEST_NAME/timing.jsonl
echo "[timing] Finetune completed in ${FINETUNE_ELAPSED}s"

if [[ "$FINETUNE_EXIT_CODE" -ne 0 ]]; then
  echo "[finetune] Failed with exit code ${FINETUNE_EXIT_CODE}, skipping robustness test"
  exit $FINETUNE_EXIT_CODE
fi

# --- Checkpoint Robustness ---
if [[ "$HAS_ROBUSTNESS" == "true" ]]; then
  echo "============================================"
  echo "[checkpoint_robustness] Running robustness test..."
  echo "============================================"
  ROBUSTNESS_START=$SECONDS

  eval $ROBUSTNESS_CMD
  ROBUSTNESS_EXIT_CODE=$?

  ROBUSTNESS_ELAPSED=$((SECONDS - ROBUSTNESS_START))
  echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"robustness\",\"seconds\":${ROBUSTNESS_ELAPSED}}" >> $PIPELINE_DIR/$TEST_NAME/timing.jsonl
  echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"total\",\"seconds\":$((SECONDS)),\"allocated\":\"${TIME}\"}" >> $PIPELINE_DIR/$TEST_NAME/timing.jsonl
  echo "[timing] Robustness completed in ${ROBUSTNESS_ELAPSED}s (total: ${SECONDS}s)"

  if [[ "$ROBUSTNESS_EXIT_CODE" -ne 0 ]]; then
    echo "[checkpoint_robustness] Failed with exit code ${ROBUSTNESS_EXIT_CODE}"
    exit $ROBUSTNESS_EXIT_CODE
  fi
fi
