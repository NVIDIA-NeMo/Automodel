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
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0"

# Propagate -s flag if PYTEST_PROPAGATE_S is set
PYTEST_S_FLAG=""
if [ "${PYTEST_PROPAGATE_S:-}" = "1" ]; then
    PYTEST_S_FLAG="-s"
fi

# Tiny public checkpoint + committed chat fixture: the point is to prove the
# scdd wiring runs on one GPU, not to converge. Override the model with
# SCDD_SMOKE_MODEL to smoke a real dLLM checkpoint instead (its vocab size must
# then be passed via --dllm.vocab_size and its mask id via --dllm.mask_token_id).
SCDD_SMOKE_MODEL=${SCDD_SMOKE_MODEL:-hf-internal-testing/tiny-random-LlamaForCausalLM}

python \
-m pytest $PYTEST_S_FLAG tests/functional_tests/training/test_scdd_smoke.py \
    --config tests/functional_tests/dllm/scdd_smoke.yaml \
    --model.pretrained_model_name_or_path "$SCDD_SMOKE_MODEL" \
    --dataset.tokenizer.pretrained_model_name_or_path "$SCDD_SMOKE_MODEL" \
    --step_scheduler.max_steps 3
