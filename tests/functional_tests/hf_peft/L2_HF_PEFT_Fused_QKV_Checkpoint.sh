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
set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0"

# Propagate -s flag if PYTEST_PROPAGATE_S is set (note: -xvs already includes -s, but we ensure it's there)
PYTEST_S_FLAG=""
if [ "${PYTEST_PROPAGATE_S:-}" = "1" ]; then
    PYTEST_S_FLAG="-s"
fi

# Functional test: PEFT LoRA + fused QKV checkpoint save / resume / HF-PEFT
# Uses a tiny Llama model (from_config, random weights) with combined qkv_proj
# and gate_up_proj projections.  No pretrained model download needed.

python -m torch.distributed.run \
    --master-port=29504 --nproc_per_node=1 --nnodes=1 \
    -m pytest $PYTEST_S_FLAG tests/functional_tests/checkpoint/test_peft_fused_qkv.py::test_peft_fused_qkv_checkpoint -xvs
