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
# limitations under the License.

# ERNIE 5.5 LoRA training on 2 GPUs with expert parallelism.
#
# The randomly initialized four-layer proxy of L2_Ernie5_5_MoE_EP2.sh (HCA + dense, HCA + MoE
# twice, DSA + MoE) receives the adapters of the LoRA recipe: the target patterns are read from
# examples/llm_finetune/ernie5_5/ernie5_5_flash_squad_ep_lora.yaml, so they cover the plain
# linear attention projections of both attention kinds, the routed experts, and the dense and
# shared FFN projections. The run makes sixteen optimizer updates on one repeated batch of two
# 256-token documents, one per rank, and saves one adapter checkpoint at the last step.
#
# run_ernie5_5_lora_training.py asserts on every rank that only adapter tensors changed, and on
# rank 0 that the loss fell and that the checkpoint holds the whole trained adapter. With
# ep_size equal to the world size each rank owns half of the experts and of their adapters, so
# the checkpoint is complete only if both ranks contributed their part.
#
# The DSA layer runs on eager PyTorch attention (at most 512 tokens per document), which works
# on any CUDA GPU. The proxy generates its own synthetic token sequences, so this test stages
# no tokenizer or dataset.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MAX_STEPS=16
LORA_RECIPE=examples/llm_finetune/ernie5_5/ernie5_5_flash_squad_ep_lora.yaml
TARGET_MODULES=$(python -c "import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))['peft']['target_modules']))" "$LORA_RECIPE")

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

# Every epoch of the proxy is one batch, so the number of epochs bounds the number of updates.
# A run takes about two minutes. A rank that fails inside the checkpoint save does not exit, and
# the other rank waits for it, so the run does not end by itself. The time limit then stops the
# launcher, which stops both ranks, and the script fails with exit status 124.
TRANSFORMERS_OFFLINE=1 timeout 600 \
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    tests/functional_tests/moe/run_ernie5_5_lora_training.py \
    --config tests/functional_tests/moe/ernie5_5_proxy.yaml \
    --model.backend.attn eager \
    --step_scheduler.max_steps "$MAX_STEPS" \
    --step_scheduler.num_epochs "$MAX_STEPS" \
    --peft._target_ nemo_automodel.components._peft.lora.PeftConfig \
    --peft.target_modules "$TARGET_MODULES" \
    --peft.dim 8 \
    --peft.alpha 32 \
    --checkpoint.enabled true \
    --checkpoint.save_consolidated false \
    --checkpoint.checkpoint_dir "$RUN_DIR"
