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

TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
-m pytest tests/functional_tests/checkpoint/test_hf_consolidated_llm.py \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --model.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
    --step_scheduler.max_steps 10 \
    --step_scheduler.global_batch_size 32 \
    --step_scheduler.local_batch_size 8 \
    --dataset.tokenizer.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
    --validation_dataset.tokenizer.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
    --dataset.dataset_name /home/TestData/lite/hf_cache/squad/ \
    --validation_dataset.dataset_name /home/TestData/lite/hf_cache/squad/ \
    --dataset.limit_dataset_samples 1000 \
    --dataset.max_seq_length 512 \
    --validation_dataset.seq_length 512 \
    --step_scheduler.ckpt_every_steps 10 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir checkpoints/ \
    --checkpoint.model_save_format safetensors \
    --checkpoint.save_consolidated true \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.dp_size 1 \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.pp_size 2 \
    --distributed.sequence_parallel false \
    --autopipeline._target_ nemo_automodel.components.distributed.pipelining.AutoPipeline \
    --autopipeline.pp_schedule 1f1b \
    --autopipeline.pp_microbatch_size 1 \
    --autopipeline.round_virtual_stages_to_pp_multiple up \
    --autopipeline.scale_grads_in_schedule false
