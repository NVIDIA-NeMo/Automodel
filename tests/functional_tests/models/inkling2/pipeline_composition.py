# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Scheduled four-GPU PP2/DP2/EP2 qualification; excluded from two-GPU PR collection.

Run with four visible GPUs through torchrun, one case per fresh process group.
Reuses the exact two-GPU assertions and tolerances, including independent optimizer
trajectories and exact checkpoint replay. No pytest entry point is registered.
"""

import sys
from pathlib import Path

from tests.functional_tests.models.inkling2.test_pipeline_cuda import Case, _worker

CASES = {
    "fp32-pp2-dp2-flex": Case(None, dp=2, flex=True),
    "fp32-pp2-ep2-flex": Case(None, dp=2, ep=2, flex=True),
    "master-bf16-pp2-ep2-flex": Case(None, bf16=True, ac=True, fused=True, dp=2, ep=2, ep_reference=True, flex=True),
    "master-bf16-pp2-ep2-interleaved-flex": Case(
        (0, 3, 6, 10, 13), bf16=True, ac=True, fused=True, interleaved=True, dp=2, ep=2, ep_reference=True, flex=True
    ),
    "fp32-pp2-dp2": Case(None, dp=2),
    "fp32-pp2-ep2": Case(None, dp=2, ep=2),
    "master-bf16-pp2-ep2": Case(None, bf16=True, ac=True, fused=True, dp=2, ep=2, ep_reference=True),
    "master-bf16-pp2-ep2-interleaved": Case(
        (0, 3, 6, 10, 13), bf16=True, ac=True, fused=True, interleaved=True, dp=2, ep=2, ep_reference=True
    ),
}

if __name__ == "__main__":
    case_name, output = sys.argv[1:]
    _worker(case_name, Path(output), CASES[case_name])
