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

# TE experts with FP32 master weights under expert parallelism.
#
# With ep_size equal to the world size, ep_shard is 1, so no FSDP mixed-precision
# wrapper casts the EP-local experts and their FP32 weights meet BF16 dispatched
# tokens. Both legs keep FP32 parameters, the default BF16 FSDP compute policy and
# the HybridEP dispatcher, and differ only in activation checkpointing, so the
# second leg also checks that recomputation repeats the expert weight casts.
#
# foreach is off because the EP-local TE expert parameters are plain tensors next
# to the FSDP DTensor parameters in one optimizer group.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

COMMON_ARGS=(
    --config tests/functional_tests/parallelism/qwen3_5_moe_proxy.yaml
    --model.dtype float32
    --model.backend.experts te
    --model.backend.dispatcher hybridep
    --optimizer.foreach false
    --step_scheduler.max_steps 4
    --distributed.ep_size 2
)

for AC in false true; do
    TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
        examples/llm_finetune/finetune.py \
        "${COMMON_ARGS[@]}" \
        --distributed.activation_checkpointing "$AC" \
        --checkpoint.checkpoint_dir "$RUN_DIR/ac_$AC"
done

python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/ac_false/training.jsonl" \
    "$RUN_DIR/ac_true/training.jsonl" \
    --axis activation_checkpointing \
    --loss-tol 1e-3 \
    --grad-norm-rtol 1e-3
