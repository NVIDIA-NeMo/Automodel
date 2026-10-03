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
# Expert-parallel parity for custom-model MoE diffusion transformers in the diffusion recipe.
#
# A seeded toy custom-model MoE DiT checkpoint (diffusers layout, HF per-expert keys) is
# finetuned by TrainDiffusionRecipe with the mock video dataloader and SimpleAdapter.
# Both legs run on 2 ranks with dp_size=2 and differ only in fsdp.ep_size, so data
# sharding and FSDP wrapping match and the comparison isolates expert parallelism.
#
# Unlike the LLM EP2 parity scripts, the compare step also fails when EP is silently
# not applied: it requires the EP leg's expert parameters to be DTensors sharded
# Shard(0) over an "ep" mesh axis with n_experts / ep_size local experts, the EP-aware
# clipping path (scale_grads_and_clip_grad_norm with ep_axis_name="ep"), and the
# checkpointer to receive the MoE mesh. The checkpoint uses sigmoid routing with a
# load-balancing correction bias, so the recipe's update_moe_gate_bias() call must move
# the bias, and both legs pass model.config_overrides, which must reach the model config.
# The EP leg also saves a checkpoint through the MoE-mesh-aware checkpointer and checks
# that the saved safetensors shards reproduce the trained weights exactly.
#
# A second EP1/EP2 pair accumulates 2 microbatches per step and compares per-parameter
# first-step updates, which catches scaling errors confined to a parameter subset (e.g.
# experts or the auxiliary-loss scale). A DDP run checks that the gate-bias hook is
# still executed when DistributedDataParallel hides it behind .module.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

RUNNER=tests/functional_tests/diffusion/run_toy_moe_dit_ep_parity.py
MASTER_PORT=${TOY_MOE_DIT_MASTER_PORT:-29531}
CONFIG_OVERRIDES='{"router_aux_loss_coef": 0.02}'

CUDA_VISIBLE_DEVICES= python "$RUNNER" write-checkpoint "$RUN_DIR/model"

# --- Reference: 2 ranks, data parallel only ---
timeout 600 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port="$MASTER_PORT" \
    "$RUNNER" train \
    --model-dir "$RUN_DIR/model" \
    --checkpoint-dir "$RUN_DIR/dp2" \
    --ep-size 1 \
    --max-steps 5 \
    --config-overrides "$CONFIG_OVERRIDES" \
    --out "$RUN_DIR/dp2.json"

# --- Expert parallel: same 2 ranks, experts sharded ---
timeout 600 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port="$((MASTER_PORT + 1))" \
    "$RUNNER" train \
    --model-dir "$RUN_DIR/model" \
    --checkpoint-dir "$RUN_DIR/ep2" \
    --ep-size 2 \
    --save-checkpoint \
    --max-steps 5 \
    --config-overrides "$CONFIG_OVERRIDES" \
    --out "$RUN_DIR/ep2.json"

python "$RUNNER" compare "$RUN_DIR/dp2.json" "$RUN_DIR/ep2.json" \
    --loss-rtol 0.02 \
    --grad-norm-rtol 0.05

# --- Gradient accumulation: 2 microbatches per optimizer step, EP1 vs EP2 ---
for EP in 1 2; do
    timeout 600 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port="$((MASTER_PORT + 1 + EP))" \
        "$RUNNER" train \
        --model-dir "$RUN_DIR/model" \
        --checkpoint-dir "$RUN_DIR/accum_ep$EP" \
        --ep-size "$EP" \
        --accumulation-steps 2 \
        --max-steps 3 \
        --config-overrides "$CONFIG_OVERRIDES" \
        --out "$RUN_DIR/accum_ep$EP.json"
done

python "$RUNNER" compare "$RUN_DIR/accum_ep1.json" "$RUN_DIR/accum_ep2.json" \
    --accumulation-steps 2 \
    --loss-rtol 0.02 \
    --grad-norm-rtol 0.05 \
    --update-rtol 0.05

# --- DDP: the gate-bias hook lives on the wrapped module ---
timeout 600 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port="$((MASTER_PORT + 4))" \
    "$RUNNER" train \
    --model-dir "$RUN_DIR/model" \
    --checkpoint-dir "$RUN_DIR/ddp" \
    --ep-size 1 \
    --strategy ddp \
    --max-steps 3 \
    --config-overrides "$CONFIG_OVERRIDES" \
    --out "$RUN_DIR/ddp.json"

python "$RUNNER" check-ddp "$RUN_DIR/ddp.json"
