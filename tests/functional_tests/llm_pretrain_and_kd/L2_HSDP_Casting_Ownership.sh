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

set -xeuo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export RUN_TE_FSDP_CASE=1

IFS=, read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#visible_devices[@]} >= 4 )); then
    export HSDP_MESH_SHAPE=2,2
    nproc=4
elif (( ${#visible_devices[@]} == 2 )); then
    # The CI runners expose two GPUs. Exercise the replicate dimension here;
    # the companion FSDP test exercises sharding on the same runners.
    export HSDP_MESH_SHAPE=2,1
    nproc=2
else
    echo "HSDP casting ownership requires at least two visible GPUs" >&2
    exit 1
fi

torchrun --nproc_per_node="$nproc" --nnodes=1 \
    tests/functional_tests/training/run_fsdp_casting_ownership.py
