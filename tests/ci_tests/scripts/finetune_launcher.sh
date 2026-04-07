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

# Unified CI launcher for LLM, VLM, and Diffusion finetuning.
#
# For diffusion pipelines (detected by PROCESSOR_NAME being set), this also
# runs data preprocessing before finetuning and inference validation after.

set -e

# ── Stage 1: Data Preprocessing (diffusion only) ──
if [[ -n "${PROCESSOR_NAME:-}" ]]; then
  echo "========================================"
  echo "Stage 1: Data Preprocessing"
  echo "========================================"
  cd /opt/Automodel
  bash tests/ci_tests/scripts/diffusion_preprocess.sh
fi

# ── Stage 2: Finetuning ──

# Test variables
CONFIG="--config /opt/Automodel/${CONFIG_PATH} \
        --checkpoint.checkpoint_dir $PIPELINE_DIR/$TEST_NAME/checkpoint"

# For diffusion, point dataloader at preprocessed cache
if [[ -n "${PROCESSOR_NAME:-}" ]]; then
  CONFIG="${CONFIG} \
         --data.dataloader.cache_dir $PIPELINE_DIR/$TEST_NAME/data_cache"
fi

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

# Diffusion recipes have no validation loop; remove val_every_steps
if [[ -n "${PROCESSOR_NAME:-}" ]]; then
  CONFIG=$(echo "$CONFIG" | sed 's/--step_scheduler.val_every_steps [0-9]*//')
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

# ── Stage 3: Inference Validation (diffusion only) ──
if [[ -n "${PROCESSOR_NAME:-}" ]]; then
  echo "========================================"
  echo "Stage 3: Inference Validation"
  echo "========================================"
  cd /opt/Automodel
  bash tests/ci_tests/scripts/diffusion_inference_validator.sh
fi
