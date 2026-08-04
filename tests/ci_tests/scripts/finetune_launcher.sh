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

# VLM recipes need the opt-in vlm-media packages kept out of the image. Install
# those packages directly instead of resolving the local project again, which
# can refetch unrelated Git dependencies already present in the image.
case "$CONFIG_PATH" in
    *vlm_finetune*)
        VLM_MEDIA_PACKAGES=(
            albumentations
            "opencv-python-headless==4.10.0.84"
            qwen-omni-utils
            qwen-vl-utils
        )
        if [[ "$(uname -m)" == "x86_64" ]]; then
            VLM_MEDIA_PACKAGES+=(decord)
        fi
        uv pip install "${VLM_MEDIA_PACKAGES[@]}"
        ;;
esac

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

if [[ "$FINETUNE_EXIT_CODE" -eq 0 && "${REQUIRE_FINITE_METRICS:-false}" == "true" ]]; then
  python3 /opt/Automodel/tests/ci_tests/scripts/assert_finite_train_metrics.py \
    --log "$PIPELINE_DIR/${TEST_NAME}_slurm_${SLURM_JOB_ID}.out" \
    || FINETUNE_EXIT_CODE=$?
fi

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

  ROBUSTNESS_TEST_MODULE="tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm"
  case "$CONFIG_PATH" in
    *retrieval/bi_encoder/*)
      ROBUSTNESS_TEST_MODULE="tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_biencoder"
      ;;
    *vlm_finetune*)
      ROBUSTNESS_TEST_MODULE="tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_vlm"
      ;;
  esac

  ROBUSTNESS_LAUNCH_CMD="$CMD"
  if [[ "$CMD" == torchrun* ]]; then
    # A completed rendezvous cannot be reused reliably by the second torchrun.
    ROBUSTNESS_LAUNCH_CMD="${CMD/--rdzv_id=${SLURM_JOB_ID}/--rdzv_id=${SLURM_JOB_ID}-robustness}"
  fi

  echo "============================================"
  echo "[checkpoint_robustness] Running robustness test..."
  echo "============================================"
  ROBUSTNESS_START=$SECONDS

  # Repeated model teardown/reload phases can fragment the CUDA allocator
  # before the resume-training check. Preserve any caller-provided setting.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  ROBUSTNESS_EXIT_CODE=0

  if [[ "${CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION:-false}" == "true" ]]; then
    # A real checkpoint restart crosses a process boundary. Large distributed
    # recipes also retain enough CUDA, PP, scheduler, and dataloader ownership
    # that rebuilding several trainers in one interpreter is not reliable.
    read -r -a ROBUSTNESS_PHASES <<< \
      "${CHECKPOINT_ROBUSTNESS_PHASES:-train_and_save automodel_reload resume}"
    # Preserve the old harness's deferred-comparison behavior: record a failed
    # parity phase, continue independent phases, and return the first failure
    # only after every reachable phase has reported a result.
    ROBUSTNESS_PHASE_RESULTS=()
    ROBUSTNESS_FAILED_PHASES=""
    SOURCE_LOAD_REFERENCE_FAILED=false
    TRAIN_AND_SAVE_FAILED=false
    for ROBUSTNESS_PHASE in "${ROBUSTNESS_PHASES[@]}"; do
      if [[ "$ROBUSTNESS_PHASE" == "source_load_parity" && "$SOURCE_LOAD_REFERENCE_FAILED" == "true" ]]; then
        ROBUSTNESS_PHASE_RESULTS+=("${ROBUSTNESS_PHASE}|SKIP|-|0|source_load_reference_failed")
        continue
      fi
      if [[ "$TRAIN_AND_SAVE_FAILED" == "true" ]]; then
        ROBUSTNESS_PHASE_RESULTS+=("${ROBUSTNESS_PHASE}|SKIP|-|0|train_and_save_failed")
        continue
      fi
      PHASE_LAUNCH_CMD="$ROBUSTNESS_LAUNCH_CMD"
      if [[ "$PHASE_LAUNCH_CMD" == torchrun* ]]; then
        PHASE_LAUNCH_CMD="${PHASE_LAUNCH_CMD/--rdzv_id=${SLURM_JOB_ID}-robustness/--rdzv_id=${SLURM_JOB_ID}-robustness-${ROBUSTNESS_PHASE}}"
        PHASE_LAUNCH_ARGS="--tee 3 --log-dir $TEST_DIR/robustness_logs/${ROBUSTNESS_PHASE}"
      else
        PHASE_LAUNCH_ARGS=""
      fi
      ROBUSTNESS_CMD="${PHASE_LAUNCH_CMD} ${PHASE_LAUNCH_ARGS} \
        -m ${ROBUSTNESS_TEST_MODULE} \
        --isolated_phase ${ROBUSTNESS_PHASE} \
        --config ${RESOLVED_ROBUSTNESS_CONFIG}"
      echo "[checkpoint_robustness] Starting isolated phase: ${ROBUSTNESS_PHASE}"
      ROBUSTNESS_PHASE_START=$SECONDS
      eval $ROBUSTNESS_CMD
      ROBUSTNESS_PHASE_EXIT_CODE=$?
      ROBUSTNESS_PHASE_ELAPSED=$((SECONDS - ROBUSTNESS_PHASE_START))
      if [[ "$ROBUSTNESS_PHASE_EXIT_CODE" -eq 0 ]]; then
        ROBUSTNESS_PHASE_RESULTS+=("${ROBUSTNESS_PHASE}|PASS|0|${ROBUSTNESS_PHASE_ELAPSED}|-")
        continue
      fi

      ROBUSTNESS_PHASE_RESULTS+=(
        "${ROBUSTNESS_PHASE}|FAIL|${ROBUSTNESS_PHASE_EXIT_CODE}|${ROBUSTNESS_PHASE_ELAPSED}|phase_process_failed"
      )
      if [[ "$ROBUSTNESS_EXIT_CODE" -eq 0 ]]; then
        ROBUSTNESS_EXIT_CODE=$ROBUSTNESS_PHASE_EXIT_CODE
      fi
      if [[ -z "$ROBUSTNESS_FAILED_PHASES" ]]; then
        ROBUSTNESS_FAILED_PHASES="$ROBUSTNESS_PHASE"
      else
        ROBUSTNESS_FAILED_PHASES="${ROBUSTNESS_FAILED_PHASES},${ROBUSTNESS_PHASE}"
      fi
      if [[ "$ROBUSTNESS_PHASE" == "source_load_reference" ]]; then
        SOURCE_LOAD_REFERENCE_FAILED=true
      elif [[ "$ROBUSTNESS_PHASE" == "train_and_save" ]]; then
        TRAIN_AND_SAVE_FAILED=true
      fi
      if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
        echo "[checkpoint_robustness][phase-failure] phase=${ROBUSTNESS_PHASE} exit_code=${ROBUSTNESS_PHASE_EXIT_CODE}"
      fi
    done

    if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
      echo "[checkpoint_robustness] Isolated phase result summary:"
      for ROBUSTNESS_PHASE_RESULT in "${ROBUSTNESS_PHASE_RESULTS[@]}"; do
        IFS="|" read -r RESULT_PHASE RESULT_STATUS RESULT_EXIT_CODE RESULT_SECONDS RESULT_REASON <<< \
          "$ROBUSTNESS_PHASE_RESULT"
        echo "[checkpoint_robustness][phase-result] phase=${RESULT_PHASE} status=${RESULT_STATUS} exit_code=${RESULT_EXIT_CODE} seconds=${RESULT_SECONDS} reason=${RESULT_REASON}"
      done
      if [[ "$ROBUSTNESS_EXIT_CODE" -eq 0 ]]; then
        echo "[checkpoint_robustness][result] status=PASS"
      else
        echo "[checkpoint_robustness][result] status=FAIL failed_phases=${ROBUSTNESS_FAILED_PHASES}"
      fi
    fi
  else
    if [[ "$ROBUSTNESS_LAUNCH_CMD" == torchrun* ]]; then
      ROBUSTNESS_LAUNCH_ARGS="--tee 3 --log-dir $TEST_DIR/robustness_logs"
    else
      ROBUSTNESS_LAUNCH_ARGS=""
    fi
    ROBUSTNESS_CMD="${ROBUSTNESS_LAUNCH_CMD} ${ROBUSTNESS_LAUNCH_ARGS} \
      -m ${ROBUSTNESS_TEST_MODULE} \
      --config ${RESOLVED_ROBUSTNESS_CONFIG}"
    eval $ROBUSTNESS_CMD
    ROBUSTNESS_EXIT_CODE=$?
  fi

  ROBUSTNESS_ELAPSED=$((SECONDS - ROBUSTNESS_START))
  echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"robustness\",\"seconds\":${ROBUSTNESS_ELAPSED}}" >> $TEST_DIR/timing.jsonl
  echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"total\",\"seconds\":$((SECONDS)),\"allocated\":\"${TIME}\"}" >> $TEST_DIR/timing.jsonl
  echo "[timing] Robustness completed in ${ROBUSTNESS_ELAPSED}s (total: ${SECONDS}s)"

  if [[ "$ROBUSTNESS_EXIT_CODE" -ne 0 ]]; then
    echo "[checkpoint_robustness] Failed with exit code ${ROBUSTNESS_EXIT_CODE}"
    exit $ROBUSTNESS_EXIT_CODE
  fi
fi
