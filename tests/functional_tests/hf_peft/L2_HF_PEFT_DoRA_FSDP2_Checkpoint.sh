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

# DoRA test using TinyLlama (non-MoE model, publicly available)
python -m torch.distributed.run \
--master-port=29503 --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace  \
-m pytest tests/functional_tests/checkpoint/test_peft.py::test_hf_peft_dora_checkpoint \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml \
    --model.pretrained_model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --step_scheduler.max_steps 10 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 8 \
    --dataset.tokenizer.pretrained_model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --validation_dataset.tokenizer.pretrained_model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset.path_or_dataset rowan/hellaswag \
    --validation_dataset.path_or_dataset rowan/hellaswag \
    --dataset.split train \
    --validation_dataset.split validation \
    --validation_dataset.num_samples_limit 64 \
    --step_scheduler.ckpt_every_steps 10 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir checkpoints/dora_test \
    --peft.match_all_linear true \
    --peft.dim 8 \
    --peft.alpha 32 \
    --peft.use_triton false \
    --peft.use_dora true \
    --peft._target_ nemo_automodel.components._peft.lora.PeftConfig \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false

