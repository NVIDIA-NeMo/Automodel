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

TRANSFORMERS_OFFLINE=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace \
examples/vlm/finetune.py \
  --config examples/vlm/gemma_3_vl_4b_cord_v2.yaml \
  --model.pretrained_model_name_or_path /home/TestData/huiyingl/hf_gemma3_2l/ \
  --model.output_hidden_states true \
  --step_scheduler.max_steps 3 \
  --step_scheduler.grad_acc_steps 1 \
  --step_scheduler.val_every_steps 1 \
  --loss_fn._target_ nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy \
  --dataset._target_=nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset \
  --dataset.path_or_dataset /home/TestData/lite/hf_cache/mini_cord_v2/ \
  --dataset.limit_dataset_samples 100 \
  --validation_dataset.path_or_dataset /home/TestData/lite/hf_cache/mini_cord_v2/ \
  --validation_dataset.limit_dataset_samples 10 \
