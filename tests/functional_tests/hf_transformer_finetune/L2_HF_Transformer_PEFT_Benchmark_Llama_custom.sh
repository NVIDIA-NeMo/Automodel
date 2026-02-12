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

# revert back to meta-llama/Llama-3.2-1B when we can use cache from HF_HOME
MODEL_PATH=/home/TestData/HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/

# override with a smaller model meta-llama/Llama-3.2-1B for testing
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
nemo_automodel/recipes/llm/benchmark.py \
    --config examples/llm_finetune/llama3_3/custom_llama3_3_70b_instruct_peft_benchmark.yaml \
    --model.pretrained_model_name_or_path=${MODEL_PATH} \
    --model.num_hidden_layers=2 \
    --distributed.tp_size=2 \
    --distributed.pp_size=1 \
    --distributed.sequence_parallel=False \
    --benchmark.warmup_steps=2 \
    --step_scheduler.max_steps=4