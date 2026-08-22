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
# gradient-norm trajectory and validation loss. `dp_size` is 1 in both runs, so
# the dataloader yields identical batches and any divergence is attributable to
# the pipeline split.
#
# Covers the gap from PR #2983 (commit 00f40419).
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
    --step_scheduler.val_every_steps 2
    --step_scheduler.global_batch_size 2
    --step_scheduler.local_batch_size 2
    --validation_dataloader.drop_last true
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

# Guard against the `_precompute_stage_shapes` bug from PR #2983.
if grep -Eiq "dynamic .*metadata inference" "$LOG_FILE"; then
    echo "ERROR: pipeline stages fell back to dynamic metadata inference instead of static metadata"
    exit 1
fi

# The gradient-norm bound is looser than the loss bound because the single-rank
# baseline runs unwrapped while the pp2 run goes through FSDP2 in bf16.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/pp2/training.jsonl" \
    --axis pp \
    --loss-tol 0.05 \
    --grad-norm-rtol 0.20

# Both runs must execute recipe-owned validation. The parity helper also rejects
# empty validation logs, so a stale PP validation skip cannot pass this check.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/validation.jsonl" \
    "$RUN_DIR/pp2/validation.jsonl" \
    --axis pp \
    --metric val_loss \
    --loss-tol 0.05
