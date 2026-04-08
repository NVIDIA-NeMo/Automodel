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

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "llm_pretrain_and_kd/loss/"
TEST_FILENAME = "L2_FusedLinearCrossEntropy_HiddenStates_Minified.sh"


class TestFusedLinearCrossEntropyHiddenStatesMinified:
    def test_fused_linear_ce_hidden_states_minified(self):
        run_test_script(TEST_FOLDER, TEST_FILENAME)

