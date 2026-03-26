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

import shutil

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "checkpoint_robustness"


class TestCheckpointRobustness:
    """Checkpoint save/load robustness tests (SFT + PEFT + cross-TP).

    Checkpoints are written to /adasif/checkpoints/ and kept for vLLM tests.
    """

    def test_checkpoint_robustness_llama3_2_3b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Llama3_2_3B.sh")

    def test_checkpoint_robustness_gpt_oss_20b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_GPT_OSS_20B.sh")


class TestVLLMDeploy:
    """vLLM deployment tests. Must run after TestCheckpointRobustness (uses their checkpoints)."""

    def test_vllm_deploy_llama3_2_3b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Llama3_2_3B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_llama_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_llama_peft", ignore_errors=True)

    def test_vllm_deploy_gpt_oss_20b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_GPT_OSS_20B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_gptoss_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_gptoss_peft", ignore_errors=True)
