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

"""Two-GPU HybridEP routing, fusion, padding, and checkpoint replay parity."""

import pytest

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "moe"
HYBRIDEP_PARITY = "L2_MoE_HybridEP_UnequalTokens_Parity.sh"


class TestHybridEPUnequalTokensParity:
    @pytest.mark.timeout(600)
    def test_routing_fusion_and_checkpoint_parity(self) -> None:
        run_test_script(TEST_FOLDER, HYBRIDEP_PARITY)
