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
export CUDA_VISIBLE_DEVICES="0,1"

CKPT_DIR="checkpoints/flashoptim_roundtrip_$$"
COMMON_ARGS=(
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad_flashoptim.yaml
    --model.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/
    --step_scheduler.global_batch_size 32
    --step_scheduler.local_batch_size 8
    --dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/
    --validation_dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/
    --dataset.dataset_name $HF_CACHE/squad/
    --validation_dataset.dataset_name $HF_CACHE/squad/
    --dataset.limit_dataset_samples 1000
    --checkpoint.enabled true
    --checkpoint.checkpoint_dir $CKPT_DIR
    --checkpoint.model_save_format torch_save
    --distributed.dp_size none
    --distributed.tp_size 1
    --distributed.cp_size 1
    --distributed.sequence_parallel false
)

cleanup() { rm -rf "$CKPT_DIR"; }
trap cleanup EXIT

# --- Run 1: train 10 steps, checkpoint at step 6 ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --step_scheduler.max_steps 10 \
    --step_scheduler.ckpt_every_steps 6

# Save the full training log and delete the final checkpoint so resume picks up step 5
cp "$CKPT_DIR/training.jsonl" "$CKPT_DIR/training_full.jsonl"
# Find and remove checkpoint dirs after the intermediate one (keep only ckpt_step)
for d in "$CKPT_DIR"/epoch_*; do
    step=$(echo "$d" | grep -oP 'step_\K[0-9]+')
    if [ "$step" -gt 5 ]; then
        rm -rf "$d"
    fi
done
# Remove training.jsonl so the resumed run writes fresh
rm -f "$CKPT_DIR/training.jsonl"

# --- Run 2: resume from step 5 checkpoint, train to step 10 ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --step_scheduler.max_steps 10 \
    --step_scheduler.ckpt_every_steps 10

# --- Validate: compare losses ---
python tests/functional_tests/checkpoint/test_flashoptim_dcp_roundtrip.py \
    "$CKPT_DIR/training_full.jsonl" \
    "$CKPT_DIR/training.jsonl" \
    --ckpt_step 6
