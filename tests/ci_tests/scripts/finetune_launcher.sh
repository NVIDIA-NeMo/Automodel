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

# Finetune launcher. Config resolution happens in config_resolver.py.
#
# Env required: CONFIG_PATH, PIPELINE_DIR, TEST_NAME, TEST_LEVEL, TEST_SCRIPT_PATH,
#   TEST_NODE_COUNT, NPROC_PER_NODE, MASTER_ADDR, MASTER_PORT, SLURM_JOB_ID, HAS_ROBUSTNESS
# Env optional: EXEC_CMD, RDZV_TIMEOUT, MAX_STEPS, LOCAL_BATCH_SIZE,
#   CONFIG_NPROC_PER_NODE, FINETUNE_ARGS, NEMO_CI_PATH, WANDB_AUTOMODEL_API_KEY, TIME

cd /opt/Automodel

CONFIG_RESOLVER="python3 /opt/Automodel/tests/ci_tests/scripts/config_resolver.py"
TEST_DIR="$PIPELINE_DIR/$TEST_NAME"
mkdir -p "$TEST_DIR"

# --- Resolve finetune config ---
RESOLVED_FINETUNE_CONFIG=$($CONFIG_RESOLVER \
  --base "/opt/Automodel/${CONFIG_PATH}" \
  --phase "${TEST_LEVEL}" \
  --output "$TEST_DIR/finetune_config.yaml")

# WANDB_API_KEY is a runtime secret, not a config key.
if [ "$TEST_LEVEL" = "convergence" ]; then
  export WANDB_API_KEY="${WANDB_AUTOMODEL_API_KEY}"
fi

# --- Pick executor ---
NPROC_PER_NODE=${CONFIG_NPROC_PER_NODE:-$NPROC_PER_NODE}
CMD="torchrun --nproc-per-node=${NPROC_PER_NODE} \
              --nnodes=${TEST_NODE_COUNT} \
              --rdzv_backend=c10d \
              --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
              --rdzv_id=${SLURM_JOB_ID} \
              --rdzv_conf=timeout=${RDZV_TIMEOUT:-600}"
if [ "$EXEC_CMD" = "python" ]; then CMD="python"; fi
if [ "$EXEC_CMD" = "uv_python" ]; then CMD="uv run python"; fi

# --- Finetune ---
RUN_CMD="${CMD} ${TEST_SCRIPT_PATH} --config ${RESOLVED_FINETUNE_CONFIG} ${FINETUNE_ARGS:-}"
echo "============================================"
echo "[finetune] Running finetune..."
echo "============================================"
FINETUNE_START=$SECONDS

eval $RUN_CMD
FINETUNE_EXIT_CODE=$?

FINETUNE_ELAPSED=$((SECONDS - FINETUNE_START))
echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"finetune\",\"seconds\":${FINETUNE_ELAPSED}}" >> $TEST_DIR/timing.jsonl
echo "[timing] Finetune completed in ${FINETUNE_ELAPSED}s"

# Performance benchmark artifact
if [ "$TEST_LEVEL" = "performance" ]; then
  echo "[benchmark] Collecting benchmark artifact..."
  python3 /opt/Automodel/tests/ci_tests/scripts/collect_benchmark_artifact.py \
    --config /opt/Automodel/${CONFIG_PATH} \
    --log $PIPELINE_DIR/${TEST_NAME}_slurm_${SLURM_JOB_ID}.out \
    --output $TEST_DIR/benchmark_results.json || true
fi

if [[ "$FINETUNE_EXIT_CODE" -ne 0 ]]; then
  echo "[finetune] Failed with exit code ${FINETUNE_EXIT_CODE}, skipping robustness test"
  exit $FINETUNE_EXIT_CODE
fi

# --- Checkpoint Robustness ---
if [[ "$HAS_ROBUSTNESS" == "true" ]]; then
  RESOLVED_ROBUSTNESS_CONFIG=$($CONFIG_RESOLVER \
    --base "/opt/Automodel/${CONFIG_PATH}" \
    --phase checkpoint_robustness \
    --output "$TEST_DIR/robustness_config.yaml")

  ROBUSTNESS_CMD="${CMD} --tee 3 --log-dir $TEST_DIR/robustness_logs \
    -m pytest --tb=short tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
    --config ${RESOLVED_ROBUSTNESS_CONFIG}"

  echo "============================================"
  echo "[checkpoint_robustness] Running robustness test..."
  echo "============================================"
  ROBUSTNESS_START=$SECONDS

  eval $ROBUSTNESS_CMD
  ROBUSTNESS_EXIT_CODE=$?

  ROBUSTNESS_ELAPSED=$((SECONDS - ROBUSTNESS_START))
  echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"robustness\",\"seconds\":${ROBUSTNESS_ELAPSED}}" >> $TEST_DIR/timing.jsonl
  echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"total\",\"seconds\":$((SECONDS)),\"allocated\":\"${TIME}\"}" >> $TEST_DIR/timing.jsonl
  echo "[timing] Robustness completed in ${ROBUSTNESS_ELAPSED}s (total: ${SECONDS}s)"

  if [[ "$ROBUSTNESS_EXIT_CODE" -ne 0 ]]; then
    echo "[checkpoint_robustness] Failed with exit code ${ROBUSTNESS_EXIT_CODE}"
    exit $ROBUSTNESS_EXIT_CODE
  fi
fi
