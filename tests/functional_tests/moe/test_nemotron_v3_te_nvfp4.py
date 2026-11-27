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

"""Pytest wrapper for the 2-GPU NVFP4 precision-layout check of a small Nemotron 3 model."""

import importlib.util

import pytest
import torch

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "moe"
TE_NVFP4_PRECISION_LAYOUT = "L2_Nemotron_V3_TE_NVFP4_Precision_Layout.sh"


def _two_nvfp4_gpus() -> bool:
    """Whether Transformer Engine is installed and two GPUs with NVFP4 kernels (Blackwell) are visible."""
    if importlib.util.find_spec("transformer_engine") is None or torch.cuda.device_count() < 2:
        return False
    return all(torch.cuda.get_device_capability(index) >= (10, 0) for index in range(2))


@pytest.mark.skipif(not _two_nvfp4_gpus(), reason="requires Transformer Engine and two GPUs with NVFP4 kernels")
class TestNemotronV3TENvfp4PrecisionLayout:
    def test_every_gemm_module_runs_the_precision_the_layout_gives_it(self):
        run_test_script(TEST_FOLDER, TE_NVFP4_PRECISION_LAYOUT)
