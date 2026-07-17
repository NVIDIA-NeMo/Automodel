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
# Joint base + drafter SFT smoke test for Gemma 4 4B (E4B).
#
# Runs `examples/vlm_finetune/finetune.py` with the
# `gemma4_4b_joint_drafter_medpix.yaml` recipe pointed at tiny mock checkpoints
# baked into CI's `$TEST_DATA_DIR`. Verifies that the composite forward + dual-
# loss path runs end-to-end for a few optimizer steps. Does not assert
# convergence.
#
# Required CI environment:
#   * transformers>=5.8.0.dev (with `transformers.models.gemma4_assistant`).
#   * `$TEST_DATA_DIR/hf_gemma4_e4b_2l/`           -- mini base checkpoint.
#   * `$TEST_DATA_DIR/hf_gemma4_e4b_assistant_2l/` -- mini drafter checkpoint
#     produced from the same `Gemma4TextConfig` as the base (so
#     `backbone_hidden_size` matches `text_config.hidden_size`).
#   * `$HF_CACHE/mini_cord_v2/` -- the standard VLM mini dataset already staged
#     in CI (used by every other `L2_HF_Transformer_VLM_*` test). Content is
#     irrelevant for this smoke test -- we only need 3 forward + backward
#     passes through the joint composite.

set -xeuo pipefail

# PyTorch 2.13's Python-native Triton bmm_outer_product override can issue an
# illegal memory access while backpropagating through the recurrent drafter on
# pre-Blackwell GPUs. Use the regular aten::bmm kernel for this compatibility
# smoke test until the upstream native-JIT kernel is fixed.
TRANSFORMERS_OFFLINE=1 TORCH_DISABLE_NATIVE_JIT=1 coverage run \
examples/vlm_finetune/finetune.py \
  --config examples/vlm_finetune/gemma4_joint_drafter/gemma4_4b_joint_drafter_medpix.yaml \
  --model.pretrained_model_name_or_path $TEST_DATA_DIR/hf_gemma4_e4b_2l/ \
  --model.drafter_path $TEST_DATA_DIR/hf_gemma4_e4b_assistant_2l/ \
  --model.text_config.output_hidden_states true \
  --step_scheduler.max_steps 3 \
  --step_scheduler.global_batch_size 1 \
  --step_scheduler.local_batch_size 1 \
  --step_scheduler.val_every_steps 1 \
  --lr_scheduler.lr_warmup_steps 0 \
  --dataset._target_=nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset \
  --dataset.path_or_dataset $HF_CACHE/mini_cord_v2/ \
  --dataset.split train \
  --dataset.limit_dataset_samples 32 \
  --validation_dataset._target_=nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset \
  --validation_dataset.path_or_dataset $HF_CACHE/mini_cord_v2/ \
  --validation_dataset.split validation \
  --validation_dataset.limit_dataset_samples 8
