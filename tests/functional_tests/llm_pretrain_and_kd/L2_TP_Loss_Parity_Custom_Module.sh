#!/bin/bash
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

# Functional test: TP loss parity for models with custom (non-TP-plan) modules.
#
# Validates that _broadcast_replicated_params_across_tp correctly
# synchronises replicated weights so that TP=2 loss == TP=1 loss.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

KL_THRESHOLD="${KL_THRESHOLD:-1e-6}"

torchrun --nproc_per_node=2 --nnodes=1 \
    tests/functional_tests/llm_pretrain_and_kd/run_tp_loss_parity_custom_module.py \
    --kl_threshold "${KL_THRESHOLD}"
