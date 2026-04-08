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

#!/bin/bash
# Unified vLLM deployment test launcher.
# Determines SFT vs PEFT from CI_JOB_STAGE and passes --mode explicitly.
# Expects: CONFIG_PATH, TEST_NAME, PIPELINE_DIR, CI_JOB_STAGE
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0"

cd /opt/Automodel
uv venv /tmp/vllm_deploy_venv
source /tmp/vllm_deploy_venv/bin/activate
uv pip install -r tests/ci_tests/requirements_deploy.txt

TEST_SCRIPT="tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py"
FINETUNE_TEST_NAME="${TEST_NAME%_vllm_deploy}"
CKPT_DIR="$PIPELINE_DIR/$FINETUNE_TEST_NAME/robustness_checkpoint"
CKPT_BASE=$(ls -d "${CKPT_DIR}"/epoch_*_step_* 2>/dev/null | sort | tail -1 || true)

if [[ -z "$CKPT_BASE" ]]; then
  echo "ERROR: No checkpoint found under ${CKPT_DIR}"
  echo "Contents of $PIPELINE_DIR/$FINETUNE_TEST_NAME/:"
  ls -la "$PIPELINE_DIR/$FINETUNE_TEST_NAME/" 2>/dev/null || echo "  Directory does not exist"
  exit 1
fi
echo "Using checkpoint: ${CKPT_BASE}"

if [[ "$CI_JOB_STAGE" == *"peft"* ]]; then
    python -m pytest $TEST_SCRIPT \
        --mode peft \
        --config_path "$CONFIG_PATH" \
        --adapter_path "${CKPT_BASE}/model/" \
        --max_new_tokens 50
else
    python -m pytest $TEST_SCRIPT \
        --mode sft \
        --config_path "$CONFIG_PATH" \
        --model_path "${CKPT_BASE}/model/consolidated/" \
        --max_new_tokens 50
fi
