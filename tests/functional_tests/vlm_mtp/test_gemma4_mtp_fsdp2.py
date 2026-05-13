# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Pytest wrapper for the Gemma4 + MTP 2-GPU FSDP2 functional test.

The actual workload is launched via ``torchrun`` from
``L2_Gemma4_MTP_FSDP2_2GPU.sh``; this thin pytest wrapper exists so the test
is discoverable by the standard ``pytest tests/functional_tests/`` runner
and obeys the project's run_test_script convention.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "vlm_mtp"
GEMMA4_MTP_FSDP2_2GPU_FILENAME = "L2_Gemma4_MTP_FSDP2_2GPU.sh"


class TestGemma4MTPFSDP2:
    def test_gemma4_mtp_fsdp2_2gpu(self):
        run_test_script(TEST_FOLDER, GEMMA4_MTP_FSDP2_2GPU_FILENAME)
