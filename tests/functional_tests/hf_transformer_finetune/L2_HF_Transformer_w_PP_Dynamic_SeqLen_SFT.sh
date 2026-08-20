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

# PP with dynamic sequence lengths: no pad_seq_len_divisible, so seq_len varies
# per batch. Tests update_seq_len() / reset_pp_stage_shapes() code path with
# both PipelineScheduleSingle (1f1b) and PipelineScheduleMulti (interleaved1f1b).
#
# The CI test model (hf_mixtral_2l) has only 2 layers. For interleaved1f1b with
# pp_size=2 we need >= 4 layers (2 virtual stages per rank), so we override
# num_hidden_layers=8 for the interleaved run. Extra layers are randomly
# initialized, which is fine for a functional smoke test.

COMMON_ARGS=(
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
    --model.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/
    --step_scheduler.max_steps 10
    --step_scheduler.val_every_steps 5
    --step_scheduler.global_batch_size 32
    --step_scheduler.local_batch_size 8
    --dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/
    --validation_dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/
    --dataset.dataset_name $HF_CACHE/squad/
    --validation_dataset.dataset_name $HF_CACHE/squad/
    --dataset.limit_dataset_samples 1000
    --dataset.seq_length 512
    --validation_dataset.seq_length 512
    --checkpoint.enabled false
    --distributed.dp_size 1
    --distributed.tp_size 1
    --distributed.cp_size 1
    --distributed.pp_size 2
    --distributed.sequence_parallel false
    --distributed.pipeline.pp_microbatch_size 1
    --distributed.pipeline.scale_grads_in_schedule false
)

LOG_FILE=$(mktemp)
trap 'rm -f "$LOG_FILE"' EXIT

# Guard against bug in commit 00f40419 (PR #2983). Checked after each schedule
# because the metadata path differs between PipelineScheduleSingle and ...Multi.
# The accumulated-gradient loss fixed by PR #3530 is a separate failure that this
# smoke run cannot see; L2_Parallelism_PP_Grad_Accum_Parity.sh covers it
# numerically.
assert_static_pp_metadata() {
    if grep -Eiq "dynamic .*metadata inference" "$LOG_FILE"; then
        echo "ERROR: pipeline stages fell back to dynamic metadata inference instead of static metadata"
        exit 1
    fi
}

# --- Run 1: 1f1b (PipelineScheduleSingle) ---
echo "=== PP dynamic seq_len: 1f1b ==="
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --distributed.pipeline.pp_schedule 1f1b \
    2>&1 | tee "$LOG_FILE"
assert_static_pp_metadata

# --- Run 2: interleaved1f1b (PipelineScheduleMulti) ---
echo "=== PP dynamic seq_len: interleaved1f1b ==="
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run examples/llm_finetune/finetune.py \
    "${COMMON_ARGS[@]}" \
    --model.num_hidden_layers 8 \
    --distributed.pipeline.pp_schedule interleaved1f1b \
    2>&1 | tee "$LOG_FILE"
assert_static_pp_metadata
