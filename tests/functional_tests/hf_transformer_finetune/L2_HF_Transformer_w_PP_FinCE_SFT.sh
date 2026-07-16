# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

# FusedLinearCrossEntropy under pipeline parallelism (2 ranks, pp_size=2).
#
# The last stage emits hidden states instead of logits so FusedLinearCrossEntropy
# can fuse the lm_head projection with the CE reduction. This is a regression
# guard for that path: before the fix the loss was silently swapped for
# MaskedCrossEntropy under PP. We therefore assert both that the run completes
# and that no downgrade warning was emitted.

LOG_FILE=$(mktemp)
trap 'rm -f "$LOG_FILE"' EXIT

TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run examples/llm_finetune/finetune.py \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --model.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/ \
    --loss_fn._target_ nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy \
    --step_scheduler.max_steps 10 \
    --step_scheduler.global_batch_size 32 \
    --step_scheduler.local_batch_size 8 \
    --dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/ \
    --validation_dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/ \
    --dataset.dataset_name $HF_CACHE/squad/ \
    --validation_dataset.dataset_name $HF_CACHE/squad/ \
    --dataset.limit_dataset_samples 1000 \
    --dataset.seq_length 512 \
    --validation_dataset.seq_length 512 \
    --checkpoint.enabled false \
    --distributed.dp_size 1 \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 2 \
    --distributed.sequence_parallel false \
    --distributed.pipeline.pp_schedule 1f1b \
    --distributed.pipeline.pp_microbatch_size 1 \
    --distributed.pipeline.scale_grads_in_schedule false \
    2>&1 | tee "$LOG_FILE"

# Guard against a silent fallback: both downgrade branches in
# _maybe_downgrade_loss_fn log "Using MaskedCrossEntropy instead." If the
# FusedLinearCrossEntropy-under-PP path regresses, the loss is swapped out and
# this test would otherwise pass as a plain smoke test without exercising fince.
if grep -q "Using MaskedCrossEntropy instead" "$LOG_FILE"; then
    echo "ERROR: FusedLinearCrossEntropy was silently downgraded to MaskedCrossEntropy under PP"
    exit 1
fi
