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
# Pipeline-parallel parity for the DeepSeek-V4 Flash recipe.
#
# DeepSeek-V4 owns `get_pipeline_stage_metas`, so it carries extra tensors across
# the stage boundary rather than just hidden states. The generic PP tests do not
# cover that contract.
#
# The proxy generates its own synthetic token sequences, so this test stages no
# tokenizer or dataset. Both legs resume from one shared native checkpoint, so
# they start from identical weights and optimizer state; initializing each leg
# from config would not, because the stage split happens before initialization
# and the RNG is rank-adjusted, so the step-0 loss gap would measure two random
# initializations rather than pipeline parallelism.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

RUN_DIR=$(mktemp -d)
LOG_FILE="$RUN_DIR/pp2.log"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

COMMON_ARGS=(
    --config tests/functional_tests/parallelism/deepseek_v4_proxy.yaml
    --step_scheduler.global_batch_size 4
    --step_scheduler.local_batch_size 2
    --distributed.tp_size 1
    --distributed.cp_size 1
    --distributed.ep_size 1
)

# --- Shared initial state: one PP2 step, saved as a native checkpoint ---
# RNG state is saved per global rank, so the checkpoint is written by the
# larger (two-rank) topology; the single-rank baseline reads rank 0's state.
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --step_scheduler.max_steps 1 \
    --step_scheduler.ckpt_every_steps 1 \
    --checkpoint.enabled true \
    --checkpoint.save_consolidated false \
    --checkpoint.checkpoint_dir "$RUN_DIR/init" \
    --distributed.pp_size 2
INIT_CKPT=$(find "$RUN_DIR/init" -mindepth 1 -maxdepth 1 -type d -name "epoch_*_step_*" | sort | tail -n 1)
if [ -z "$INIT_CKPT" ]; then
    echo "ERROR: the shared initial checkpoint was not written"
    exit 1
fi

# Three compared steps after the shared step exercise optimizer state across
# updates without a convergence-length run.
RESUME_ARGS=(
    --step_scheduler.max_steps 4
    --step_scheduler.ckpt_every_steps 1000
    --checkpoint.enabled true
    --checkpoint.save_consolidated false
    --checkpoint.restore_from "$INIT_CKPT"
)

# --- Baseline: single rank, no parallelism ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/baseline" \
    --distributed.pp_size 1

# --- Pipeline parallel: 2 ranks ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/pp2" \
    --distributed.pp_size 2 \
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

# Both legs resume from identical weights and optimizer state, so they differ
# only through bf16 reduction order (pp2 runs under FSDP2's bf16 mixed-precision
# policy while the single-rank baseline runs unwrapped). On H100 the largest
# observed gaps over three steps were 4.8e-4 in loss and below 1e-4 relative in
# gradient norm; the gates leave about 10x headroom. Legs initialized
# independently from config miss these gates: their step-0 losses differ by
# about 0.1 before any update.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/pp2/training.jsonl" \
    --axis pp \
    --loss-tol 0.005 \
    --grad-norm-rtol 0.01
