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
# Pipeline-parallel parity for Qwen3.8-Flash-Next.
#
# The decoder carries four HyperConnection streams between layers and
# alternates GatedDeltaNet with QSA, so a stage boundary exchanges a
# ``[batch, sequence, hc_count * hidden]`` state between two layer types. The
# generic PP tests run a dense, single-stream model and do not cover that.
#
# The proxy generates its own synthetic token sequences, so this test stages no
# tokenizer or dataset.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

RUN_DIR=$(mktemp -d)
LOG_FILE="$RUN_DIR/pp2.log"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

COMMON_ARGS=(
    --config tests/functional_tests/parallelism/qwen3_8_flash_next_proxy.yaml
    --step_scheduler.max_steps 6
    --step_scheduler.global_batch_size 4
    --step_scheduler.local_batch_size 2
    --distributed.tp_size 1
    --distributed.cp_size 1
    --distributed.ep_size 1
)

# --- Baseline: single rank, no parallelism ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/baseline" \
    --distributed.pp_size 1

# --- Pipeline parallel: 2 ranks ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/pp2" \
    --distributed.pp_size 2 \
    2>&1 | tee "$LOG_FILE"

# The model-owned stage metas must be used; a fall back to dynamic shape
# inference would hide a wrong HC width.
if grep -Eiq "dynamic .*metadata inference" "$LOG_FILE"; then
    echo "ERROR: pipeline stages fell back to dynamic metadata inference instead of static metadata"
    exit 1
fi
if ! grep -q "Precomputed pipeline stage shapes" "$LOG_FILE"; then
    echo "ERROR: pipeline stage shapes were never precomputed; static metadata did not run"
    exit 1
fi

# Loss is ~10.4 at random init over a 32k vocab and both legs run in bf16,
# whose relative resolution at that magnitude is ~0.04 absolute; 0.10 sits
# above that floor. Gradient norm is compared relatively and stays loose
# because the single-rank baseline runs unwrapped while pp2 goes through
# FSDP2's bf16 mixed-precision policy.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/pp2/training.jsonl" \
    --axis pp \
    --loss-tol 0.10 \
    --grad-norm-rtol 0.20
