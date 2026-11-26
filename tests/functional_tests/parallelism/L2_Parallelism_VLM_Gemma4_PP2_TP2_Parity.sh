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
# Pipeline-parallel and tensor-parallel parity for the Gemma4 31B VLM recipe.
#
# Runs the Gemma4 31B proxy twice with the same seed and data order -- once on a
# single rank, once at pp_size=2 -- and asserts both follow the same loss and
# gradient-norm trajectory. `dp_size` is 1 in both runs, so the dataloader yields
# identical batches and any divergence is attributable to the pipeline split.
#
# Covers the gap from PR #2983 (commit 00f40419).
#
# The TP leg runs the same proxy at tp_size=2. The real gemma4_31b recipes run tp_size=4,
# and their attention layout is unusual: `attention_k_eq_v`, a `global_head_dim` that
# differs from `head_dim`, and `num_global_key_value_heads` != `num_key_value_heads`. The
# proxy preserves all three so TP shards the same structure it shards in production.
#
# Required CI environment:
#   * `$TEST_DATA_DIR/hf_gemma4_e4b_2l/` -- staged Gemma4 processor (already
#     required by L2_HF_Transformer_VLM_Gemma4_Joint_Drafter.sh).
#   * `$HF_CACHE/mini_cord_v2/` -- the standard VLM mini dataset.

# The PP2 and TP2 legs share one proxy checkpoint and one single-rank baseline (both use
# global_batch_size 4, so each step accumulates two micro-batches), instead of building
# and running them twice. Each leg keeps its own tolerances.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

RUN_DIR=$(mktemp -d)
LOG_FILE="$RUN_DIR/pp2.log"
PROXY_CKPT="$RUN_DIR/proxy"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT
source tests/functional_tests/parallelism/parity_legs.sh

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
    --step_scheduler.global_batch_size 4
    --step_scheduler.local_batch_size 2
)

# --- Baseline: single rank, no parallelism ---
start_leg baseline env TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --standalone --nproc_per_node=1 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/baseline" \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 1

# --- Pipeline parallel: 2 ranks, pp_size=2 ---
start_leg pp2 env TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --standalone --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/pp2" \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 2 \
    --distributed.pipeline.pp_schedule 1f1b \
    --distributed.pipeline.pp_microbatch_size 1

# --- Tensor parallel: 2 ranks, tp_size=2 ---
# FSDP2Manager skips parallelization at world_size 1, so the baseline runs unwrapped
# while the tp2 run gets FSDP2's default bf16 MixedPrecisionPolicy; hence the looser
# gradient-norm bound.
start_leg tp2 env TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --standalone --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/tp2" \
    --distributed.tp_size 2 \
    --distributed.cp_size 1 \
    --distributed.pp_size 1

wait_legs

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

python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/tp2/training.jsonl" \
    --axis tp \
    --loss-tol 0.05 \
    --grad-norm-rtol 0.20
