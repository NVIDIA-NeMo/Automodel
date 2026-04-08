#!/bin/bash
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

set -exo pipefail

# Run the cross-encoder recipe with 2 GPUs (FSDP2).
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
    -m coverage run \
    -m nemo_automodel.recipes.retrieval.train_cross_encoder \
    --config \
    tests/functional_tests/retrieval/recipe_cross_encoder.yaml \
    --model.pretrained_model_name_or_path $TEST_DATA_DIR/llama-nemotron-embed-1b-v2/ \
    --tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/llama-nemotron-embed-1b-v2/ \
    --dataloader.dataset.data_dir_list $TEST_DATA_DIR/embedding_testdata/training.jsonl

# Compare baseline vs finetuned cross-encoder checkpoint.
python3 -m coverage run \
    tests/functional_tests/retrieval/compare_cross_encoder_models.py \
    $TEST_DATA_DIR/llama-nemotron-embed-1b-v2 \
    /workspace/output/cross_encoder_inline/checkpoints/epoch_0_step_31/ \
    $TEST_DATA_DIR/embedding_testdata/testing.jsonl \
    true
