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

TEST_SCRIPT="tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py"
CKPT_BASE="$PIPELINE_DIR/$TEST_NAME/checkpoint/epoch_0_step_99"

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
