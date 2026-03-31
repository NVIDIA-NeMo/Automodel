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

    def test_vllm_deploy_nemotron_nano_v3(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Nemotron_Nano_V3.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_nano_v3_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_nano_v3_peft", ignore_errors=True)

    def test_vllm_deploy_nemotron_flash_1b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Nemotron_Flash_1B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_nemflash1b_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_nemflash1b_peft", ignore_errors=True)

    def test_vllm_deploy_gemma_3_270m(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Gemma_3_270m.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_gemma3_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_gemma3_peft", ignore_errors=True)

    def test_vllm_deploy_phi_4(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Phi_4.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_phi4_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_phi4_peft", ignore_errors=True)

    def test_vllm_deploy_nemotron_nano_v2_9b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Nemotron_Nano_V2_9B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_nanov2_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_nanov2_peft", ignore_errors=True)

    def test_vllm_deploy_baichuan_2_7b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Baichuan_2_7B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_baichuan_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_baichuan_peft", ignore_errors=True)

    def test_vllm_deploy_qwen2_5_7b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Qwen2_5_7B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_qwen25_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_qwen25_peft", ignore_errors=True)

    def test_vllm_deploy_qwen3_moe_30b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Qwen3_MoE_30B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_qwen3moe_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_qwen3moe_peft", ignore_errors=True)

    def test_vllm_deploy_nemotron_super_120b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Nemotron_Super_120B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_super120b_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_super120b_peft", ignore_errors=True)

    def test_vllm_deploy_llama3_3_super_49b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Llama3_3_Super_49B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_super49b_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_super49b_peft", ignore_errors=True)

    def test_vllm_deploy_mistral3_3b(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Mistral3_3B.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_mistral3_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_mistral3_peft", ignore_errors=True)

    def test_vllm_deploy_nemotron_nano_8b_v1(self):
        try:
            run_test_script(TEST_FOLDER, "L2_vLLM_Deploy_Nemotron_Nano_8B_V1.sh")
        finally:
            shutil.rmtree("/adasif/checkpoints/robustness_nano8bv1_sft", ignore_errors=True)
            shutil.rmtree("/adasif/checkpoints/robustness_nano8bv1_peft", ignore_errors=True)
