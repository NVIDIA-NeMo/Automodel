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

"""2-GPU functional tests for model expansion under tensor and data parallelism.

The checks live in ``run_expansion_parallel.py`` next to this file, which also runs under
``torchrun`` on its own -- the training containers do not all carry pytest. This module
spawns the same checks so they are collected with the rest of the suite; keeping one
implementation means the two entry points cannot drift.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))

from run_expansion_parallel import CHECKS, WORLD_SIZE, spawn_worker  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(torch.cuda.device_count() < WORLD_SIZE, reason=f"requires {WORLD_SIZE} GPUs"),
]


@pytest.mark.parametrize("mode", list(CHECKS))
def test_expansion_under_parallelism(mode, tmp_path):
    """Function preservation, weight distribution and gradient correctness on 2 ranks."""
    mp.spawn(spawn_worker, args=(WORLD_SIZE, str(tmp_path / "dist_init"), mode), nprocs=WORLD_SIZE, join=True)
