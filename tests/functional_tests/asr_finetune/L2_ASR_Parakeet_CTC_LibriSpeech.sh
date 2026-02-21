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


set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export CUDA_VISIBLE_DEVICES="0"

python -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
examples/asr_finetune/finetune.py \
  --config examples/asr_finetune/parakeet/parakeet_ctc_0.6b_librispeech.yaml \
  --step_scheduler.max_steps 3 \
  --step_scheduler.global_batch_size 2 \
  --step_scheduler.local_batch_size 2 \
  --step_scheduler.val_every_steps 1 \
  --dataset.limit_dataset_samples 10 \
  --validation_dataset.limit_dataset_samples 10 \
  --checkpoint.enabled false
