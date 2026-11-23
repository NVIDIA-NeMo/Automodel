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

# te_nvfp4.precision_map on a small Nemotron 3 model, 2 GPUs with NVFP4 kernels (Blackwell).
#
# Leg 1 runs the recipe of nemotron_v3_te_nvfp4_proxy.yaml and checks, from Transformer Engine's state after
# each module ran, that every GEMM module used the recipe the map gives it; see run_nemotron_v3_te_nvfp4.py.
# It also checks that the two ranks hold different values in every randomly initialized block weight.
# Legs 2 and 3 are negative controls: with a tail one layer short, or with the Mamba output projection in
# NVFP4 instead of MXFP8, the same checks must fail. Leg 4 builds the Mamba projections as torch.nn.Linear
# (backend.mamba_linear: torch): the map must stop the run at startup instead of leaving them unquantized.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

run() {
    local name=$1
    shift
    TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
        tests/functional_tests/moe/run_nemotron_v3_te_nvfp4.py \
        --config tests/functional_tests/moe/nemotron_v3_te_nvfp4_proxy.yaml \
        --checkpoint.checkpoint_dir "$RUN_DIR/$name" "$@" 2>&1 | tee "$RUN_DIR/$name.log"
}

run recipe
grep -q "NVFP4_PRECISION_RESULT" "$RUN_DIR/recipe.log"

if run short_tail --model.backend.te_nvfp4.precision_map.high_precision_tail_layers 2; then
    echo "A tail of 2 layers passed the checks written for a tail of 3" >&2
    exit 1
fi
grep -q "ran {'nvfp4'}, not high_precision" "$RUN_DIR/short_tail.log"

if run out_proj_nvfp4 --model.backend.te_nvfp4.precision_map.mamba_out_proj nvfp4; then
    echo "An NVFP4 Mamba output projection passed the checks written for MXFP8" >&2
    exit 1
fi
grep -q "ran {'nvfp4'}, not mxfp8" "$RUN_DIR/out_proj_nvfp4.log"

if run torch_mamba --model.backend.mamba_linear torch; then
    echo "A precision map that quantizes torch.nn.Linear Mamba projections did not stop the run" >&2
    exit 1
fi
grep -q "module(s) that are not Transformer Engine GEMM modules" "$RUN_DIR/torch_mamba.log"
