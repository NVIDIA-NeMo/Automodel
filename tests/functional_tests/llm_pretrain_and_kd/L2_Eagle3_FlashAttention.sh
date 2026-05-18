#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

set -xeuo pipefail

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0"

PYTEST_S_FLAG=""
if [ "${PYTEST_PROPAGATE_S:-}" = "1" ]; then
    PYTEST_S_FLAG="-s"
fi

python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run \
    -m pytest $PYTEST_S_FLAG tests/functional_tests/training/test_eagle3_flash_attention.py -vs
