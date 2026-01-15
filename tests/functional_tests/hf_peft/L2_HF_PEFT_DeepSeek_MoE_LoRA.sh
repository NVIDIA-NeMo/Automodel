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
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

coverage run --data-file=/workspace/.coverage --source=/workspace  \
nemo_automodel/recipes/llm/train_ft.py \
    --config tests/functional_tests/hf_peft/deepseek_moe_lora_small_for_test.yaml \
    --step_scheduler.max_steps 10 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 8 \
    --dataset.tokenizer.pretrained_model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --validation_dataset.tokenizer.pretrained_model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset.path_or_dataset rowan/hellaswag \
    --validation_dataset.path_or_dataset rowan/hellaswag \
    --dataset.split train \
    --validation_dataset.split validation \
    --validation_dataset.num_samples_limit 128 \
    --step_scheduler.ckpt_every_steps 10 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir checkpoints/deepseek_moe_lora_test \
    --checkpoint.load_base_model false \
    --peft.target_modules "[\"*.experts\"]" \
    --peft.dim 16 \
    --peft.alpha 32 \
    --peft.dropout 0.05 \
    --peft._target_ nemo_automodel.components._peft.lora.PeftConfig \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --distributed.dp_replicate_size 1
