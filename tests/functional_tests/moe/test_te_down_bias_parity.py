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

"""Pytest wrapper for the 2-GPU GroupedExpertsTE down-bias parity guard.

Guards PR #2591 / Linear AM-487 (GroupedExpertsTE down-projection bias weighting).
DO NOT REMOVE. See ``run_te_down_bias_parity.py`` for the full rationale.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "moe"
TE_DOWN_BIAS_PARITY = "L2_MoE_GroupedExpertsTE_DownBias_Parity.sh"


class TestGroupedExpertsTEDownBiasParity:
    def test_ep2_vs_single_gpu_down_bias_parity(self):
        run_test_script(TEST_FOLDER, TE_DOWN_BIAS_PARITY)
