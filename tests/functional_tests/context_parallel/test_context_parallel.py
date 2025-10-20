# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Functional tests for context parallelism on attention layers.

These tests validate that attention layers produce identical forward outputs
and gradients when using different context parallel sizes with packed sequences.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "context_parallel"
CP_QWEN3_MOE_ATTENTION_TEST_FILENAME = "L2_CP_Qwen3MoE_Attention_Test.sh"
CP_DEEPSEEK_V3_MLA_TEST_FILENAME = "L2_CP_DeepSeekV3_MLA_Test.sh"


class TestContextParallelAttention:
    """Test suite for context parallel attention layers."""

    def test_cp_qwen3_moe_attention(self):
        """Test Qwen3MoeAttention layer with CP=1 vs CP=2."""
        run_test_script(TEST_FOLDER, CP_QWEN3_MOE_ATTENTION_TEST_FILENAME)

    def test_cp_deepseek_v3_mla(self):
        """Test DeepSeek V3 MLA layer with CP=1 vs CP=2."""
        run_test_script(TEST_FOLDER, CP_DEEPSEEK_V3_MLA_TEST_FILENAME)
