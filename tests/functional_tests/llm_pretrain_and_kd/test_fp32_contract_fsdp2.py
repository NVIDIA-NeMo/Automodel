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

TEST_FOLDER = "llm_pretrain_and_kd"
FP32_CONTRACT_FSDP2_FILENAME = "L2_Fp32ContractFSDP2.sh"


class TestFp32ContractFSDP2:
    def test_fp32_contract_matches_unsharded_reference(self):
        run_test_script(TEST_FOLDER, FP32_CONTRACT_FSDP2_FILENAME)
