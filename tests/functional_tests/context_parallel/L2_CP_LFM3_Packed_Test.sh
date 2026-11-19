#!/bin/bash
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

set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

# Halo exchange and CP2 shards vs one process through NCCL, with SDPA and FlashAttention
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    tests/functional_tests/context_parallel/run_lfm3_packed_cp.py

# The recipe's composition: FSDP2 over the CP ranks, loss normalization, clipping and one optimizer step
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    tests/functional_tests/context_parallel/run_lfm3_packed_cp.py --fsdp2

# A document that spans several of four shards, through Gloo on CPU
python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 -m coverage run \
    tests/functional_tests/context_parallel/run_lfm3_packed_cp.py --cpu
