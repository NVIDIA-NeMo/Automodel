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
# limitations under the License.

# GLM-5.3 CP sequence gather: exact forward/backward and backward-buffer ownership for CP group sizes 1 and 2.
set -xeuo pipefail
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

output_dir=$(mktemp -d)
trap 'rm -rf "$output_dir"' EXIT
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    tests/functional_tests/context_parallel/run_glm5_next_gather_autograd.py \
    --backend nccl --group-sizes 1 2 --output "$output_dir/gather.json"
