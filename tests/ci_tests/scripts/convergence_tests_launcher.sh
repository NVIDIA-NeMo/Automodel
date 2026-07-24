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

# Convergence launcher (weekly Tulu-3 flow): train the recipe to completion, then
# run downstream IFEval on the consolidated checkpoint and gate on the score
# staying within k*stderr of the recorded baseline (ci.downstream_eval).
#
# Steps mirror examples/convergence/tulu3/models/<model>/run_te_fusedadam.md:
#   1. train (config_resolver phase=convergence -> full 1000 steps, save consolidated)
#   2. one-time, model-agnostic eval-env setup (setup_lm_eval.sh: uv + [vllm] +
#      torchcodec removal + cutlass-dsl pin) -- idempotent, skipped if already built
#   3. eval + threshold gate (convergence_eval.py)
#
# Env required: CONFIG_PATH, PIPELINE_DIR, TEST_NAME, TEST_LEVEL, TEST_SCRIPT_PATH,
#   TEST_NODE_COUNT, NPROC_PER_NODE, MASTER_ADDR, MASTER_PORT, SLURM_JOB_ID
# Env optional: EXEC_CMD, RDZV_TIMEOUT, CONFIG_NPROC_PER_NODE, FINETUNE_ARGS,
#   WANDB_AUTOMODEL_API_KEY

cd /opt/Automodel

# VLM recipes (e.g. gemma4) need qwen-vl-utils/opencv from the opt-in vlm-media extra.
case "$CONFIG_PATH" in
    *vlm_finetune*|*gemma4*) uv pip install ".[vlm-media]" ;;
esac

CONFIG_RESOLVER="python3 /opt/Automodel/tests/ci_tests/scripts/config_resolver.py"
TEST_DIR="$PIPELINE_DIR/$TEST_NAME"
mkdir -p "$TEST_DIR"

# --- Resolve finetune config (phase=convergence; recipe.ci.convergence restores 1000 steps) ---
RESOLVED_FINETUNE_CONFIG=$($CONFIG_RESOLVER \
  --base "/opt/Automodel/${CONFIG_PATH}" \
  --phase convergence \
  --output "$TEST_DIR/finetune_config.yaml")

export WANDB_API_KEY="${WANDB_AUTOMODEL_API_KEY}"
# wandb stays disabled in CI: the recipes ship `wandb.enable: false` and we do NOT override
# it on. The CI token (WANDB_AUTOMODEL_API_KEY) has no write access to the recipes' wandb
# entity, so wandb.init() raises CommError on rank0 and strands the other ranks at the
# checkpoint-consolidation gloo barrier. The convergence gate is downstream IFEval, not
# wandb. To enable CI wandb later, grant the token write access to the entity and pass
# `--wandb.enable true` on the training command below.

# Entry script by recipe type. Convergence recipes live under examples/convergence/
# (mixed LLM/VLM), so the path-based heuristic templates use does not apply -- pick the
# entry from the recipe's `recipe:` field.
RECIPE_KIND=$(python3 -c "import yaml; print(yaml.safe_load(open('${RESOLVED_FINETUNE_CONFIG}')).get('recipe',''))")
if [ "$RECIPE_KIND" = "FinetuneRecipeForVLM" ]; then
  TEST_SCRIPT_PATH="examples/vlm_finetune/finetune.py"
else
  TEST_SCRIPT_PATH="examples/llm_finetune/finetune.py"
fi

# --- Prefilter (LLM recipes only) ---
# The LLM recipes (moonlight/qwen) train on raw allenai/tulu-3-sft-mixture with
# truncation: false; over-length samples spike memory on the large-vocab MoEs and OOM.
# Hence prefilter to seq_length first: resolve (or
# build once) the filtered cache and point both dataset paths at it. gemma4 (VLM) packs
# with drop_long_samples and is skipped.
if [ "$RECIPE_KIND" != "FinetuneRecipeForVLM" ]; then
  CACHED_DATASET=$(python3 /opt/Automodel/tests/ci_tests/scripts/convergence_prefilter.py \
    --config "${RESOLVED_FINETUNE_CONFIG}")
  if [ -z "${CACHED_DATASET}" ]; then
    echo "[convergence] prefilter failed to resolve a cache path"; exit 1
  fi
  echo "[convergence] prefiltered dataset: ${CACHED_DATASET}"
  FINETUNE_ARGS="--dataset.path_or_dataset_id ${CACHED_DATASET} --validation_dataset.path_or_dataset_id ${CACHED_DATASET} ${FINETUNE_ARGS:-}"
fi

# --- Executor ---
NPROC_PER_NODE=${CONFIG_NPROC_PER_NODE:-$NPROC_PER_NODE}
CMD="torchrun --nproc-per-node=${NPROC_PER_NODE} \
              --nnodes=${TEST_NODE_COUNT} \
              --rdzv_backend=c10d \
              --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
              --rdzv_id=${SLURM_JOB_ID} \
              --rdzv_conf=timeout=${RDZV_TIMEOUT:-600}"
if [ "$EXEC_CMD" = "python" ]; then CMD="python"; fi
if [ "$EXEC_CMD" = "uv_python" ]; then CMD="uv run python"; fi

# --- 1. Train ---
echo "============================================"
echo "[convergence] Training ${TEST_NAME}..."
echo "============================================"
TRAIN_START=$SECONDS
eval "${CMD} ${TEST_SCRIPT_PATH} --config ${RESOLVED_FINETUNE_CONFIG} ${FINETUNE_ARGS:-}"
TRAIN_EXIT_CODE=$?
echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"train\",\"seconds\":$((SECONDS - TRAIN_START))}" >> "$TEST_DIR/timing.jsonl"
if [[ "$TRAIN_EXIT_CODE" -ne 0 ]]; then
  echo "[convergence] Training failed with exit code ${TRAIN_EXIT_CODE}, skipping eval"
  exit $TRAIN_EXIT_CODE
fi

# --- 2. Eval-env setup (model-agnostic, idempotent) ---
echo "[convergence] Setting up lm-evaluation-harness..."
export HOME=/root
export PATH="/root/.local/bin:$PATH"
# Allow Hub access for eval: lm-eval fetches the IFEval dataset (google/IFEval), which the CI's
# HF cache does not pre-warm, so the default offline mode raises OfflineModeIsEnabled.
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
bash examples/convergence/tulu3/eval/setup_lm_eval.sh /opt/lm-evaluation-harness
uv pip uninstall --python /opt/lm-evaluation-harness/.venv/bin/python torchcodec || true

# --- 3. Downstream eval + threshold gate ---
# checkpoint_dir may be absolute (CI computes {PIPELINE_DIR}/{TEST_NAME}/checkpoint) or
# relative to /opt/Automodel (recipe default); resolve to absolute either way.
CHECKPOINT_DIR=$(python3 -c "import yaml,os; cd=yaml.safe_load(open('${RESOLVED_FINETUNE_CONFIG}'))['checkpoint']['checkpoint_dir']; print(cd if os.path.isabs(cd) else os.path.join('/opt/Automodel', cd))")
echo "============================================"
echo "[convergence] Eval + threshold gate (checkpoint_dir=${CHECKPOINT_DIR})..."
echo "============================================"
EVAL_START=$SECONDS
python3 /opt/Automodel/tests/ci_tests/scripts/convergence_eval.py \
  --recipe "/opt/Automodel/${CONFIG_PATH}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "$TEST_DIR"
EVAL_EXIT_CODE=$?
echo "{\"test\":\"${TEST_NAME}\",\"phase\":\"eval\",\"seconds\":$((SECONDS - EVAL_START))}" >> "$TEST_DIR/timing.jsonl"

exit $EVAL_EXIT_CODE
