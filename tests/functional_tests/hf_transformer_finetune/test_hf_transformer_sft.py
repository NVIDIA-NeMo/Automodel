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

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "hf_transformer_finetune"
HF_TRANSFORMER_SFT_FILENAME = "L2_HF_Transformer_SFT.sh"
HF_TRANSFORMER_SFT_NVFSDP_FILENAME = "L2_HF_Transformer_SFT_nvfsdp.sh"
HF_TRANSFORMER_PEFT_FILENAME = "L2_HF_Transformer_PEFT.sh"
HF_TRANSFORMER_PEFT_NVFSDP_FILENAME = "L2_HF_Transformer_PEFT_nvfsdp.sh"


class TestHFTransformerFinetune:
    def test_hf_transformer_sft(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_FILENAME)

    def test_hf_transformer_sft_nvfsdp(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_NVFSDP_FILENAME)

    def test_hf_transformer_peft(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_FILENAME)

    def test_hf_transformer_peft_nvfsdp(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_NVFSDP_FILENAME)
