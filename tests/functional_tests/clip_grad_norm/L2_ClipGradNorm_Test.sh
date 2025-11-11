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

set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

# Detect available GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected $GPU_COUNT GPUs"

# Set CUDA_VISIBLE_DEVICES based on available GPUs
if [ "$GPU_COUNT" -ge 8 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    NPROC=$GPU_COUNT
elif [ "$GPU_COUNT" -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    NPROC=4
elif [ "$GPU_COUNT" -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES="0,1"
    NPROC=2
else
    echo "Error: This test requires at least 2 GPUs, but found $GPU_COUNT"
    exit 1
fi

# Run clip_grad_norm tests with detected GPU count
torchrun --nproc_per_node=$NPROC --nnodes=1 \
    tests/functional_tests/clip_grad_norm/run_clip_grad_norm.py
