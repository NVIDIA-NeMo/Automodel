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

import os
import subprocess

HF_TRANSFORMER_SFT_FILENAME = "L2_HF_Transformer_SFT.sh"
HF_TRANSFORMER_SFT_NVFSDP_FILENAME = "L2_HF_Transformer_SFT_nvfsdp.sh"


class TestHFTransformerSFT:
    def test_hf_transformer_sft(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_file_path = os.path.join(dir_path, HF_TRANSFORMER_SFT_FILENAME)
        test_cmd = ["bash", test_file_path]
        subprocess.run(test_cmd, check=True)

    def test_hf_transformer_sft_nvfsdp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_file_path = os.path.join(dir_path, HF_TRANSFORMER_SFT_NVFSDP_FILENAME)
        test_cmd = ["bash", test_file_path]
        subprocess.run(test_cmd, check=True)
