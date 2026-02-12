# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

CKPT_DIR="checkpoints"
echo "Checkpoint directory: $CKPT_DIR"

# 1. Train and save checkpoint
echo "Starting training..."
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
    examples/llm_finetune/finetune.py \
    --config examples/llm_finetune/qwen/qwen3_moe_2layer_proxy_lora.yaml \
    --step_scheduler.max_steps 5 \
    --step_scheduler.ckpt_every_steps 5 \
    --checkpoint.checkpoint_dir $CKPT_DIR \
    --model.trust_remote_code true

# 2. Resume from checkpoint
echo "Resuming training..."
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
    examples/llm_finetune/finetune.py \
    --config examples/llm_finetune/qwen/qwen3_moe_2layer_proxy_lora.yaml \
    --step_scheduler.max_steps 10 \
    --checkpoint.checkpoint_dir $CKPT_DIR \
    --model.trust_remote_code true

echo "Test passed!"
