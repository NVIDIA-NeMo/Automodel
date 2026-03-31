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

SFT_CKPT_DIR="/adasif/checkpoints/robustness_nano_v3_sft"
PEFT_CKPT_DIR="/adasif/checkpoints/robustness_nano_v3_peft"

# Step 1: SFT checkpoint robustness
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
    --config examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag.yaml \
    --model.pretrained_model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --step_scheduler.max_steps 5 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 2 \
    --step_scheduler.ckpt_every_steps 5 \
    --step_scheduler.val_every_steps 5 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$SFT_CKPT_DIR" \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated true \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.pp_size 1 \
    --distributed.ep_size 8 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --hf_kl_threshold 7e-2 \
    --experts_implementation grouped_mm \
    --trust_remote_code

# Step 2: PEFT checkpoint robustness
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_llm.py \
    --config examples/llm_finetune/nemotron/nemotron_nano_v3_hellaswag_peft.yaml \
    --model.pretrained_model_name_or_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --step_scheduler.max_steps 5 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 2 \
    --step_scheduler.ckpt_every_steps 5 \
    --step_scheduler.val_every_steps 5 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$PEFT_CKPT_DIR" \
    --checkpoint.save_consolidated true \
    --peft.use_triton false \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.pp_size 1 \
    --distributed.ep_size 8 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --hf_kl_threshold 1e-1 \
    --experts_implementation grouped_mm \
    --trust_remote_code
