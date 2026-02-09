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

# revert back to Qwen/Qwen2.5-1.5B when we can use cache from HF_HOME
MODLE_PATH=/home/TestData/HF_HOME/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323

# override with a smaller model Qwen/Qwen2.5-1.5B for testing
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --master-port=29513 --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
nemo_automodel/recipes/llm/benchmark.py \
    --config examples/llm_finetune/qwen/custom_qwen2_5_32b_peft_benchmark.yaml \
    --model.pretrained_model_name_or_path=${MODLE_PATH} \
    --distributed.tp_size=2 \
    --distributed.pp_size=1 \
    --distributed_config.sequence_parallel=True \
    --benchmark.warmup_steps=2 \
    --step_scheduler.max_steps=4 \
    --step_scheduler.global_batch_size=2 \
    --step_scheduler.local_batch_size=1 \
    --dataset.seq_len=256