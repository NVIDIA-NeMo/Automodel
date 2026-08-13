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
# Pipeline-parallel parity for the Gemma4 31B VLM recipe.
#
# Runs the Gemma4 31B proxy twice with the same seed and data order -- once on a
# single rank, once at pp_size=2 -- and asserts both follow the same loss and
# gradient-norm trajectory. `dp_size` is 1 in both runs, so the dataloader yields
# identical batches and any divergence is attributable to the pipeline split.
#
# Covers the gap that let PR #2983 (commit 00f40419) reach main: it disabled
# static pipeline-stage metadata for every PP model, but the only PP tests in
# GitHub CI ran stock HuggingFace Mixtral through the generic path at tp_size=1,
# so nothing failed while the real gemma4_31b_tp4_pp2 / tp4_pp4 recipes broke in
# nemo-ci.
#
# Required CI environment:
#   * `$TEST_DATA_DIR/hf_gemma4_e4b_2l/` -- staged Gemma4 processor (already
#     required by L2_HF_Transformer_VLM_Gemma4_Joint_Drafter.sh).
#   * `$HF_CACHE/mini_cord_v2/` -- the standard VLM mini dataset.

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
python tests/functional_tests/parallelism/make_gemma4_proxy_checkpoint.py \
    --output-dir "$PROXY_CKPT" \
    --processor-dir $TEST_DATA_DIR/hf_gemma4_e4b_2l/

COMMON_ARGS=(
    --config tests/functional_tests/parallelism/gemma4_31b_proxy.yaml
    --model.pretrained_model_name_or_path "$PROXY_CKPT"
    --processor.pretrained_model_name_or_path "$PROXY_CKPT"
    --dataset.path_or_dataset $HF_CACHE/mini_cord_v2/
    --dataset.split train
    --dataset.limit_dataset_samples 32
    --validation_dataset.path_or_dataset $HF_CACHE/mini_cord_v2/
    --validation_dataset.split validation
    --validation_dataset.limit_dataset_samples 8
    --step_scheduler.max_steps 6
    --step_scheduler.global_batch_size 2
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

# Guard against a silent capability loss. `_precompute_stage_shapes` falls back
# to PyTorch's dynamic shape inference when it cannot install static stage
# metadata, which keeps the run alive on the generic dense path but produces
# wrong metadata for models that declare their own PP tensor contract. This is
# the exact regression from commit 00f40419, and it is axis-independent: the
# line is emitted whatever the tp/cp/ep sizes are.
if grep -Eiq "dynamic .*metadata inference" "$LOG_FILE"; then
    echo "ERROR: pipeline stages fell back to dynamic metadata inference instead of static metadata"
    exit 1
fi

# The gradient-norm bound is looser than the loss bound for a structural reason:
# FSDP2Manager skips parallelization entirely at world_size 1, so the baseline
# runs unwrapped while the pp2 run gets FSDP2's default bf16 MixedPrecisionPolicy.
# The two are therefore not bit-comparable, and measured no-accumulation runs sit
# within ~15%. This still catches the failures worth catching -- a wrong stage
# metadata, a gradient reduced over the wrong group, or a missing scale move the
# norm by integer factors, not percent.
#
# `global_batch_size` equals `local_batch_size` here, so each step is a single
# accumulation window.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/pp2/training.jsonl" \
    --axis pp \
    --loss-tol 0.05 \
    --grad-norm-rtol 0.20
