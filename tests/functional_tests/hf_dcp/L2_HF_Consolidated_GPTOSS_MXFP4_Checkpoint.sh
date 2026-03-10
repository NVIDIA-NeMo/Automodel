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
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

GPTOSS_MODEL_DIR=/tmp/gptoss_2l_mxfp4_$$
trap 'rm -rf "$GPTOSS_MODEL_DIR"' EXIT

python tests/functional_tests/checkpoint/create_gptoss_2l_mxfp4.py \
    --output-dir "$GPTOSS_MODEL_DIR" \
    --tokenizer-dir "$TEST_DATA_DIR/hf_mixtral_2l/"

TRANSFORMERS_OFFLINE=1 TORCH_COMPILE_DISABLE=1 \
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
-m pytest tests/functional_tests/checkpoint/test_hf_consolidated_gptoss_mxfp4.py \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --model.pretrained_model_name_or_path "$GPTOSS_MODEL_DIR" \
    --step_scheduler.max_steps 10 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 8 \
    --dataset.tokenizer.pretrained_model_name_or_path "$GPTOSS_MODEL_DIR" \
    --validation_dataset.tokenizer.pretrained_model_name_or_path "$GPTOSS_MODEL_DIR" \
    --dataset.dataset_name $HF_CACHE/squad/ \
    --validation_dataset.dataset_name $HF_CACHE/squad/ \
    --validation_dataset.padding true \
    --dataset.limit_dataset_samples 1000 \
    --dataset.padding true \
    --dataloader.collate_fn.pad_seq_len_divisible 512 \
    --validation_dataloader.collate_fn.pad_seq_len_divisible 512 \
    --dataset.seq_length 512 \
    --validation_dataset.seq_length 512 \
    --step_scheduler.ckpt_every_steps 10 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir checkpoints/ \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated true \
    --distributed.dp_size 2 \
    --distributed.ep_size 2 \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 1 \
    --distributed.sequence_parallel false
