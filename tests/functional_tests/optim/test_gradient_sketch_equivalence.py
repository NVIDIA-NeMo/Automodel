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

"""Functional tests for the gradient_sketch optimizer: FSDP2 and fused-expert equivalence."""

import pytest
import torch

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "optim"
GRADIENT_SKETCH_FSDP_TEST_FILENAME = "L2_Optim_GradientSketch_FSDP_Equivalence.sh"
GRADIENT_SKETCH_EXPERT_TEST_FILENAME = "L2_Optim_GradientSketch_Expert_Equivalence.sh"
FSDP_REQUIRED_GPUS = 2
EXPERT_REQUIRED_GPUS = 4


class TestGradientSketchEquivalence:
    """Test suite for gradient_sketch equivalence under distributed layouts."""

    @pytest.mark.skipif(
        torch.cuda.device_count() < FSDP_REQUIRED_GPUS,
        reason="requires 2 GPUs for FSDP2 gradient_sketch equivalence",
    )
    @pytest.mark.timeout(600)
    def test_gradient_sketch_fsdp_equivalence(self):
        """Test gradient_sketch matches the dense reference under FSDP2 on 2 GPUs."""
        run_test_script(TEST_FOLDER, GRADIENT_SKETCH_FSDP_TEST_FILENAME)

    @pytest.mark.skipif(
        torch.cuda.device_count() < EXPERT_REQUIRED_GPUS,
        reason="requires 4 GPUs for fused-expert gradient_sketch equivalence",
    )
    @pytest.mark.timeout(600)
    def test_gradient_sketch_expert_equivalence(self):
        """Test gradient_sketch matches the dense reference on fused expert weights on 4 GPUs."""
        run_test_script(TEST_FOLDER, GRADIENT_SKETCH_EXPERT_TEST_FILENAME)
