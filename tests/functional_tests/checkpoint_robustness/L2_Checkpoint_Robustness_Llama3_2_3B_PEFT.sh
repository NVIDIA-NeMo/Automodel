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
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

CKPT_DIR="checkpoints/robustness_peft_$$"

# Step 1: Checkpoint robustness (torchrun + pytest)
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml \
    --model.pretrained_model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --step_scheduler.max_steps 5 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 2 \
    --step_scheduler.ckpt_every_steps 5 \
    --step_scheduler.val_every_steps 5 \
    --dataset.path_or_dataset $HF_CACHE/hellaswag/ \
    --validation_dataset.path_or_dataset $HF_CACHE/hellaswag/ \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$CKPT_DIR" \
    --checkpoint.save_consolidated true \
    --peft.use_triton false \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --hf_kl_threshold 5e-3

# Find the checkpoint directory for vLLM test
CKPT_STEP_DIR=$(ls -d "$CKPT_DIR"/epoch_*_step_* | sort | tail -1)

# Step 2: vLLM deployment with PEFT adapter merge (single-process pytest)
python -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_vllm_deploy.py \
    --model_path meta-llama/Llama-3.2-3B-Instruct \
    --adapter_path "$CKPT_STEP_DIR/model/" \
    --max_new_tokens 50
