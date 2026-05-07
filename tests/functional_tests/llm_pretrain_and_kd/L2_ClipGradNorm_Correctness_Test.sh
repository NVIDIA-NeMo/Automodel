#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.
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

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -ge 8 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    NPROC=8
elif [ "$GPU_COUNT" -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    NPROC=4
elif [ "$GPU_COUNT" -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES="0,1"
    NPROC=2
else
    echo "Error: clip_grad_norm correctness tests require >=2 GPUs, found $GPU_COUNT"
    exit 1
fi

torchrun --nproc_per_node=$NPROC --nnodes=1 \
    tests/functional_tests/llm_pretrain_and_kd/run_clip_grad_norm_correctness.py
