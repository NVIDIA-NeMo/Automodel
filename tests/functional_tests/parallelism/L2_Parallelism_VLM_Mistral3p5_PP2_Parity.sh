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
# Pipeline-parallel parity for the Mistral3.5 VLM recipe, which ships pp_size 8.
#
# mistral3 is one of only two model types with its own VLM pipeline forward
# (`_PP_VLM_MODEL_TYPES_WITH_DEDICATED_FORWARD`), which routes pixel values
# through the vision tower on stage 0. The generic PP tests do not cover it.
#
# Required CI environment:
#   * `mistralai/Mistral-Medium-3.5-128B` processor files in the staged HF cache.
#     Images are generated at test time, so no dataset is needed.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

RUN_DIR=$(mktemp -d)
LOG_FILE="$RUN_DIR/pp2.log"
PROXY_CKPT="$RUN_DIR/proxy"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

# Build the randomly-initialized proxy checkpoint. Both runs load the same
# weights, which is what makes the two loss trajectories comparable at all.
python tests/functional_tests/parallelism/make_mistral3p5_proxy_checkpoint.py \
    --output-dir "$PROXY_CKPT"

COMMON_ARGS=(
    --config tests/functional_tests/parallelism/mistral3p5_proxy.yaml
    --model.pretrained_model_name_or_path "$PROXY_CKPT"
    --processor.pretrained_model_name_or_path "$PROXY_CKPT"
    --step_scheduler.max_steps 6
    --step_scheduler.global_batch_size 4
    --step_scheduler.local_batch_size 2
)

# --- Baseline: single rank, no parallelism ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/baseline" \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 1

# --- Pipeline parallel: 2 ranks, pp_size=2 ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/pp2" \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 2 \
    --distributed.pipeline.pp_schedule 1f1b \
    --distributed.pipeline.pp_microbatch_size 1 \
    2>&1 | tee "$LOG_FILE"

# Guard against the `_precompute_stage_shapes` bug from PR #2983. Assert the
# static path positively as well: if the precompute is skipped outright, the
# fallback log line disappears too and the negative grep alone would pass.
if grep -Eiq "dynamic .*metadata inference" "$LOG_FILE"; then
    echo "ERROR: pipeline stages fell back to dynamic metadata inference instead of static metadata"
    exit 1
fi
if ! grep -q "Precomputed pipeline stage shapes" "$LOG_FILE"; then
    echo "ERROR: pipeline stage shapes were never precomputed; static metadata did not run"
    exit 1
fi

# The gradient-norm bound is looser than the loss bound because the single-rank
# baseline runs unwrapped while the pp2 run goes through FSDP2 in bf16.
# global_batch_size is twice local_batch_size, so each step accumulates two
# micro-batches -- the case PR #3530 fixed.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/pp2/training.jsonl" \
    --axis pp \
    --loss-tol 0.05 \
    --grad-norm-rtol 0.20
