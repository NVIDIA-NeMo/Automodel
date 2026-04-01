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

SFT_CKPT_DIR="/adasif/checkpoints/robustness_mistral3_sft"
PEFT_CKPT_DIR="/adasif/checkpoints/robustness_mistral3_peft"

# Step 1: SFT checkpoint robustness
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
    --config examples/llm_finetune/mistral/ministral3_3b_squad.yaml \
    --model.pretrained_model_name_or_path mistralai/Ministral-3-3B-Instruct-2512 \
    --step_scheduler.max_steps 5 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 2 \
    --step_scheduler.ckpt_every_steps 5 \
    --step_scheduler.val_every_steps 5 \
    --dataset.limit_dataset_samples 500 \
    --validation_dataset.limit_dataset_samples 500 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$SFT_CKPT_DIR" \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated true \
    --distributed.dp_size none \
    --distributed.tp_size 2 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --hf_kl_threshold 5e-3 \
    --tokenizer_name mistralai/Ministral-3-3B-Instruct-2512 \
    --cross_tp_size 2 \
    --cross_tp_kl_threshold 5e-3 \
    --check_resume

# Step 2: PEFT checkpoint robustness
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
    --config examples/llm_finetune/mistral/ministral3_3b_squad_peft.yaml \
    --model.pretrained_model_name_or_path mistralai/Ministral-3-3B-Instruct-2512 \
    --step_scheduler.max_steps 5 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 2 \
    --step_scheduler.ckpt_every_steps 5 \
    --step_scheduler.val_every_steps 5 \
    --dataset.limit_dataset_samples 500 \
    --validation_dataset.limit_dataset_samples 500 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$PEFT_CKPT_DIR" \
    --checkpoint.save_consolidated true \
    --peft.use_triton false \
    --distributed.dp_size none \
    --distributed.tp_size 2 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --hf_kl_threshold 5e-3 \
    --tokenizer_name mistralai/Ministral-3-3B-Instruct-2512 \
    --check_resume
