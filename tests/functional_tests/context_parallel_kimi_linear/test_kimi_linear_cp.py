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

"""Functional test for Kimi Linear context parallelism.

Kept in its own suite (and therefore its own CI job) because the KDA path
compiles a large set of FLA Triton kernels, which alone takes longer than the
headroom left in the shared ``context_parallel`` job.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "context_parallel_kimi_linear"
CP_KIMI_LINEAR_TEST_FILENAME = "L2_CP_KimiLinear_Test.sh"


class TestKimiLinearContextParallel:
    """Test suite for context parallelism on the Kimi Linear hybrid model."""

    def test_cp_kimi_linear(self):
        """Test the Kimi Linear hybrid (KDA linear attention + MLA) with CP=1 vs CP=2."""
        run_test_script(TEST_FOLDER, CP_KIMI_LINEAR_TEST_FILENAME)
