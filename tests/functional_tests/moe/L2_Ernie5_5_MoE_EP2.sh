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

# ERNIE 5.5 training through the fine-tuning CLI on 2 GPUs with expert parallelism.
#
# A randomly initialized four-layer proxy (HCA + dense, HCA + MoE twice, DSA + MoE) runs
# eight optimizer updates on one repeated batch of two 256-token documents, one per rank.
# With ep_size equal to the world size the experts are split over the two ranks and the
# expert FSDP shard axis has size 1.
#
# ERNIE5_5_DSA_BACKEND selects the attention backend of the DSA layer:
#   eager (default)  eager PyTorch attention, at most 512 tokens per document; any CUDA GPU.
#   cudnn            cuDNN sparse attention; requires nvidia-cudnn-frontend[cutedsl] and FlashMLA;
#                    validated on SM100-class GPUs (GB200) only.
# The HCA layers run on the eager PyTorch path in both cases.
#
# The proxy generates its own synthetic token sequences, so this test stages no
# tokenizer or dataset.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

DSA_BACKEND="${ERNIE5_5_DSA_BACKEND:-eager}"
MAX_STEPS=8
# vocab_size of ernie5_5_proxy.yaml.
VOCAB_SIZE=4096

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    examples/llm_finetune/finetune.py \
    --config tests/functional_tests/moe/ernie5_5_proxy.yaml \
    --model.backend.attn "$DSA_BACKEND" \
    --step_scheduler.max_steps "$MAX_STEPS" \
    --checkpoint.checkpoint_dir "$RUN_DIR/$DSA_BACKEND"

# Every update sees the same two documents, so a model that trains lowers the loss.
python tests/functional_tests/moe/check_ernie5_5_training_log.py \
    "$RUN_DIR/$DSA_BACKEND/training.jsonl" \
    --expected-steps "$MAX_STEPS" \
    --vocab-size "$VOCAB_SIZE"
