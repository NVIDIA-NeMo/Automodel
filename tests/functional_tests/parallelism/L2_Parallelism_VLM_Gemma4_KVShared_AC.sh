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
# Activation checkpointing must cover attention on KV-shared Gemma4 (E2B/E4B).
#
# The model is built in-process from a shrunk E4B config, so this needs nothing
# staged in TEST_DATA_DIR. See run_gemma4_kv_shared_ac.py for what it asserts.

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run \
    tests/functional_tests/parallelism/run_gemma4_kv_shared_ac.py
