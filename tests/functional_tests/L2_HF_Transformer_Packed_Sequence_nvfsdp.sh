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

## 12 packs get crated with 100 samples from squad dataset and packed sequence size 2048
TRANSFORMERS_OFFLINE=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace \
recipes/llm/finetune.py \
  --config recipes/llm/llama_3_2_1b_squad_nvfsdp.yaml \
  --model.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --step_scheduler.max_steps 3 \
  --step_scheduler.grad_acc_steps 2 \
  --dataset.tokenizer.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --packed_sequence.packed_sequence_size 2048 \
  --dataloader.batch_size 2 \
  --validation_dataset.tokenizer.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
  --dataset.dataset_name /home/TestData/lite/hf_cache/squad/ \
  --dataset.limit_dataset_samples 100
