# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Run knowledge distillation finetune flow on small, offline fixtures.
TRANSFORMERS_OFFLINE=1 coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode \
examples/llm_kd/kd.py \
  --config examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml \
  --model.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --teacher_model.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --step_scheduler.max_steps 2 \
  --step_scheduler.global_batch_size 2 \
  --step_scheduler.local_batch_size 2 \
  --step_scheduler.val_every_steps 1 \
  --loss_fn._target_ nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy \
  --dataset.tokenizer.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --validation_dataset.tokenizer.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --dataset.dataset_name /home/TestData/lite/hf_cache/squad/ \
  --dataset.limit_dataset_samples 8 \
  --validation_dataset.dataset_name /home/TestData/lite/hf_cache/squad/ \
  --validation_dataset.limit_dataset_samples 8 \
  --distributed.activation_checkpointing true


