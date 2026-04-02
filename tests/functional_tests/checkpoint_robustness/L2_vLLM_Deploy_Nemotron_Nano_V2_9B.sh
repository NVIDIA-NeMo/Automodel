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
# Requires: L2_Checkpoint_Robustness_Nemotron_Nano_V2_9B.sh to have run first.
# Note: this needs to launch in an environment with vllm installed.
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

SFT_CKPT_DIR="/adasif/checkpoints/robustness_nanov2_sft"
PEFT_CKPT_DIR="/adasif/checkpoints/robustness_nanov2_peft"

# Step 1: vLLM SFT greedy comparison (model_impl="transformers")
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path "$SFT_CKPT_DIR/LATEST/model/consolidated/" \
    --tokenizer nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --max_new_tokens 50 \
    --trust_remote_code

# Step 2: vLLM PEFT native LoRA
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --adapter_path "$PEFT_CKPT_DIR/LATEST/model/" \
    --max_new_tokens 50 \
    --trust_remote_code
