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

#!/bin/bash
# Expert-parallel parity for the DeepSeek-V4 Flash recipe.
#
# Both legs run on 2 ranks with dp_size=2 and differ only in ep_size, so the
# data sharding and FSDP wrapping are identical and the comparison isolates
# expert parallelism. That also allows a tighter bound than the PP test, whose
# single-rank baseline is not FSDP-wrapped.
#
# The proxy generates its own synthetic token sequences, so this test stages no
# tokenizer or dataset.
#
# Known gap: this catches EP changing the numbers, but not EP silently never
# being applied -- that would make both legs identical and pass. The PP test
# closes the equivalent hole by grepping for the static-metadata log line;
# `apply_ep` has no such line to grep.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

COMMON_ARGS=(
    --config tests/functional_tests/parallelism/deepseek_v4_proxy.yaml
    --step_scheduler.max_steps 6
    --step_scheduler.global_batch_size 4
    --step_scheduler.local_batch_size 2
    --distributed.tp_size 1
    --distributed.cp_size 1
    --distributed.pp_size 1
)

# --- Reference: 2 ranks, data parallel only ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/dp2" \
    --distributed.ep_size 1

# --- Expert parallel: same 2 ranks, experts sharded ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/ep2" \
    --distributed.ep_size 2

python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/dp2/training.jsonl" \
    "$RUN_DIR/ep2/training.jsonl" \
    --axis ep \
    --loss-tol 0.02 \
    --grad-norm-rtol 0.05
