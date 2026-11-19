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

# ERNIE 5.5 training on packed rows with context parallelism, through the fine-tuning CLI on 2 GPUs.
#
# The proxy of ernie5_5_proxy.yaml, with HCA heads of 512 channels (the width of the
# sparse-attention kernels), trains on rows of 512 tokens that pack synthetic documents of
# varying length. It runs twice with expert parallelism over both ranks: with cp_size 1,
# one row per rank, and with cp_size 2, every row split over both ranks, two rows per
# update and activation checkpointing of the decoder layers, as in the packed recipes.
# Both runs see the same rows in every update, and the log check compares them.
#
# ERNIE5_5_DSA_BACKEND selects the attention backend:
#   eager (default)  PyTorch attention in the HCA and DSA layers; any CUDA GPU.
#   cudnn            cuDNN sparse attention in the HCA and DSA layers; requires
#                    nvidia-cudnn-frontend[cutedsl] and FlashMLA; validated on SM100-class
#                    GPUs (GB200) only.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

ATTN_BACKEND="${ERNIE5_5_DSA_BACKEND:-eager}"
MAX_STEPS=4
# vocab_size of ernie5_5_proxy.yaml.
VOCAB_SIZE=4096

RUN_DIR=$(mktemp -d)
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

for CP_SIZE in 1 2; do
    if [ "$CP_SIZE" = 2 ]; then CHECKPOINTING=true; else CHECKPOINTING=false; fi
    TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
        examples/llm_finetune/finetune.py \
        --config tests/functional_tests/moe/ernie5_5_proxy.yaml \
        --model.backend.attn "$ATTN_BACKEND" \
        --model.config.head_dim 512 \
        --distributed.cp_size "$CP_SIZE" \
        --distributed.activation_checkpointing "$CHECKPOINTING" \
        --step_scheduler.max_steps "$MAX_STEPS" \
        --step_scheduler.num_epochs 1 \
        --dataset.num_sentences 64 \
        --dataset.mean_len 150 \
        --dataset.std_len 60 \
        --dataset.max_sentence_len 400 \
        --packed_sequence.packing_strategy thd \
        --packed_sequence.packed_sequence_size 512 \
        --packed_sequence.pad_to_multiple_of 128 \
        --dataloader.collate_fn._target_ nemo_automodel.components.datasets.utils.packed_sequence_thd_collater \
        --checkpoint.checkpoint_dir "$RUN_DIR/cp$CP_SIZE"
done

python tests/functional_tests/moe/check_ernie5_5_packed_cp_log.py \
    "$RUN_DIR/cp1/training.jsonl" "$RUN_DIR/cp2/training.jsonl" \
    --expected-steps "$MAX_STEPS" \
    --vocab-size "$VOCAB_SIZE"
