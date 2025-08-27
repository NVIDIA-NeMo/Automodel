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
-m pytest tests/functional_tests/training/test_meta_device_sft.py \
    --config examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml \
    --model.pretrained_model_name_or_path /home/TestData/huiyingl/hf_gemma3_2l/ \
    --dataset._target_=nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset \
    --dataset.path_or_dataset /home/TestData/lite/hf_cache/mini_cord_v2/ \
    --dataset.limit_dataset_samples 100 \
    --validation_dataset.path_or_dataset /home/TestData/lite/hf_cache/mini_cord_v2/ \
    --validation_dataset.limit_dataset_samples 10 \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false
