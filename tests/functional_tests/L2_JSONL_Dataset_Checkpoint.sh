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

TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode \
-m pytest tests/functional_tests/datasets/llm/test_jsonl_dataset_checkpointing.py \
    --config examples/llm/llama_3_2_1b_fineweb_edu.yaml \
    --model.pretrained_model_name_or_path /home/TestData/akoumparouli/hf_mixtral_2l/ \
    --dataset._target_ nemo_automodel.components.datasets.llm.jsonl_dataset.JSONLDataset \
    --dataset.root_dir /home/TestData/adasif/ \
    --dataset.sources "{'fineweb': 100.0}" \
    --dataset.batch_size 8 \
    --dataset.packed_sequence_size 1024 \
    --dataset.infinite true \
    --dataset.split validation \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir checkpoints/ \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false
coverage combine
