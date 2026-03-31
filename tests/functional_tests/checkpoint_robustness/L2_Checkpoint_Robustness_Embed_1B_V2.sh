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
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

CKPT_DIR="/adasif/checkpoints/robustness_embed1b_sft"

# Biencoder checkpoint robustness (FSDP only, no TP/PP)
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
    -m pytest tests/functional_tests/checkpoint_robustness/test_checkpoint_robustness_biencoder.py \
    --config tests/functional_tests/llm_pretrain_and_kd/customizer_retrieval/recipe.yaml \
    --model.pretrained_model_name_or_path /home/TestData/automodel/llama-nemotron-embed-1b-v2/ \
    --tokenizer.pretrained_model_name_or_path /home/TestData/automodel/llama-nemotron-embed-1b-v2/ \
    --step_scheduler.max_steps 5 \
    --step_scheduler.global_batch_size 64 \
    --step_scheduler.local_batch_size 8 \
    --step_scheduler.ckpt_every_steps 5 \
    --step_scheduler.val_every_steps 5 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir "$CKPT_DIR" \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated true \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false \
    --cosine_threshold 0.999 \
    --check_resume
