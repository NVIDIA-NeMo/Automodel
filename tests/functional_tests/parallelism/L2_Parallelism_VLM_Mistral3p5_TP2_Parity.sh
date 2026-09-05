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
# Tensor-parallel parity for the Mistral3.5 VLM recipe, which ships tp_size 8.
#
# Runs the Mistral3.5 proxy twice with the same seed and data order -- once on a
# single rank, once at tp_size=2 -- and asserts both follow the same loss and
# gradient-norm trajectory. `dp_size` is 1 in both runs, so any divergence is
# attributable to the tensor-parallel sharding.
#
# Required CI environment:
#   * `mistralai/Mistral-Medium-3.5-128B` processor files in the staged HF cache.
#     Images are generated at test time, so no dataset is needed.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

RUN_DIR=$(mktemp -d)
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
    --step_scheduler.global_batch_size 2
    --step_scheduler.local_batch_size 2
    --distributed.pp_size 1
    --distributed.cp_size 1
)

# --- Baseline: single rank, no parallelism ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/baseline" \
    --distributed.tp_size 1

# --- Tensor parallel: 2 ranks, tp_size=2 ---
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/vlm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint.checkpoint_dir "$RUN_DIR/tp2" \
    --distributed.tp_size 2

# See the PP2 test for why the gradient-norm bound is looser than the loss bound:
# FSDP2Manager skips parallelization at world_size 1, so the baseline runs
# unwrapped while the tp2 run gets FSDP2's default bf16 MixedPrecisionPolicy.
python tests/functional_tests/parallelism/compare_parallel_parity.py \
    "$RUN_DIR/baseline/training.jsonl" \
    "$RUN_DIR/tp2/training.jsonl" \
    --axis tp \
    --loss-tol 0.05 \
    --grad-norm-rtol 0.20
