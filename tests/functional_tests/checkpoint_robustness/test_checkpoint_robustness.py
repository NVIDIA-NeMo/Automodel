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

TEST_FOLDER = "checkpoint_robustness"


class TestCheckpointRobustness:
    """Checkpoint save/load robustness tests (SFT + PEFT + cross-TP).

    Checkpoints are written to /adasif/checkpoints/.
    """

    def test_checkpoint_robustness_llama3_2_3b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Llama3_2_3B.sh")

    def test_checkpoint_robustness_gpt_oss_20b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_GPT_OSS_20B.sh")

    def test_checkpoint_robustness_nemotron_nano_v3(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Nemotron_Nano_V3.sh")

    def test_checkpoint_robustness_nemotron_flash_1b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Nemotron_Flash_1B.sh")

    def test_checkpoint_robustness_gemma_3_270m(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Gemma_3_270m.sh")

    def test_checkpoint_robustness_phi_4(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Phi_4.sh")

    def test_checkpoint_robustness_nemotron_nano_v2_9b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Nemotron_Nano_V2_9B.sh")

    def test_checkpoint_robustness_baichuan_2_7b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Baichuan_2_7B.sh")

    def test_checkpoint_robustness_qwen2_5_7b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Qwen2_5_7B.sh")

    def test_checkpoint_robustness_qwen3_moe_30b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Qwen3_MoE_30B.sh")

    def test_checkpoint_robustness_nemotron_super_120b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Nemotron_Super_120B.sh")

    def test_checkpoint_robustness_llama3_3_super_49b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Llama3_3_Super_49B.sh")

    def test_checkpoint_robustness_mistral3_3b(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Mistral3_3B.sh")

    def test_checkpoint_robustness_nemotron_nano_8b_v1(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Nemotron_Nano_8B_V1.sh")

    def test_checkpoint_robustness_embed_1b_v2(self):
        run_test_script(TEST_FOLDER, "L2_Checkpoint_Robustness_Embed_1B_V2.sh")
