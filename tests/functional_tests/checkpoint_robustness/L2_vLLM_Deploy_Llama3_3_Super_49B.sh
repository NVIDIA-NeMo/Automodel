# Copyright (c) 2026, NVIDIA CORPORATION.
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
# Requires: L2_Checkpoint_Robustness_Llama3_3_Super_49B.sh to have run first.
# Note: this needs to launch in an environment with vllm installed.
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0"

SFT_CKPT_DIR="/adasif/checkpoints/robustness_super49b_sft"
PEFT_CKPT_DIR="/adasif/checkpoints/robustness_super49b_peft"

SFT_CONSOLIDATED="${SFT_CKPT_DIR}/$(ls -1 "${SFT_CKPT_DIR}" | grep epoch | sort | tail -1)/model/consolidated"
PEFT_MODEL_DIR="${PEFT_CKPT_DIR}/$(ls -1 "${PEFT_CKPT_DIR}" | grep epoch | sort | tail -1)/model"

# Step 1: vLLM SFT greedy comparison (full comparison mode — dense model)
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path "$SFT_CONSOLIDATED" \
    --tokenizer nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --max_new_tokens 50 \
    --trust_remote_code

# Step 2: vLLM PEFT native LoRA
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 \
    --adapter_path "$PEFT_MODEL_DIR" \
    --max_new_tokens 50 \
    --trust_remote_code
