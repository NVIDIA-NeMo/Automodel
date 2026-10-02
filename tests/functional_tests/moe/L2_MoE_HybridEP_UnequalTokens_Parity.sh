#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Reuse compiled kernels across variants, with a private cache by default.
if [[ -z "${NEMO_HYBRIDEP_JIT_CACHE:-}" ]]; then
    hybridep_jit_cache=$(mktemp -d)
    trap 'rm -rf "$hybridep_jit_cache"' EXIT
    export NEMO_HYBRIDEP_JIT_CACHE="$hybridep_jit_cache"
fi

for compact in 0 1; do
    for fusion in 0 1; do
        args=()
        if [[ "$compact" == 1 ]]; then
            args+=(--compact-routing)
        fi
        if [[ "$fusion" == 1 ]]; then
            args+=(--permute-fusion)
        fi
        if [[ "$compact" == 1 && "$fusion" == 1 ]]; then
            args+=(--activation-checkpointing)
        fi
        variant_start=$SECONDS
        TRANSFORMERS_OFFLINE=1 python3 \
            -m torch.distributed.run --standalone --nproc_per_node=2 --nnodes=1 \
            -m coverage run \
            tests/functional_tests/moe/run_hybridep_unequal_tokens.py "${args[@]}"
        echo "HybridEP parity: compact=$compact fusion=$fusion elapsed=$((SECONDS - variant_start))s"
    done
done
