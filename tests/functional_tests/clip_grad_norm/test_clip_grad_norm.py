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

"""Functional tests for gradient clipping with various parallelism configurations.

These tests validate that _clip_grad_norm_impl works correctly with
TP, PP, EP, and combinations thereof.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "clip_grad_norm"
TEST_FILENAME = "L2_ClipGradNorm_Test.sh"


class TestClipGradNorm:
    """Test suite for gradient clipping with various parallelism strategies."""

    def test_clip_grad_norm_all_configs(self):
        """Test gradient clipping with all parallelism configurations."""
        run_test_script(TEST_FOLDER, TEST_FILENAME)
