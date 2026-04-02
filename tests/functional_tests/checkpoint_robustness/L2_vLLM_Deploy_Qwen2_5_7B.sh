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
# Requires: L2_Checkpoint_Robustness_Qwen2_5_7B.sh to have run first.
# Note: this needs to launch in an environment with vllm installed.
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0"

MODEL="Qwen/Qwen2.5-7B"
SFT_CKPT_DIR="/adasif/checkpoints/robustness_qwen25_sft"
PEFT_CKPT_DIR="/adasif/checkpoints/robustness_qwen25_peft"

# Find latest checkpoint directories
SFT_CONSOLIDATED=$(ls -d "$SFT_CKPT_DIR"/step_*/model/consolidated/ | grep -E 'step_[0-9]+' | sort -t_ -k2 -n | tail -1)
PEFT_MODEL_DIR=$(ls -d "$PEFT_CKPT_DIR"/step_*/model/ | grep -E 'step_[0-9]+' | sort -t_ -k2 -n | tail -1)

# Step 1: vLLM SFT greedy comparison
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path "$SFT_CONSOLIDATED" \
    --tokenizer "$MODEL" \
    --max_new_tokens 50

# Step 2: vLLM PEFT native LoRA
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path "$MODEL" \
    --tokenizer "$MODEL" \
    --adapter_path "$PEFT_MODEL_DIR" \
    --max_new_tokens 50
