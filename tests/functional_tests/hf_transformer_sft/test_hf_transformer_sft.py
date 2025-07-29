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
import pytest

HF_TRANSFORMER_SFT_FILENAME = "L2_HF_Transformer_SFT.sh"
HF_TRANSFORMER_SFT_NVFSDP_FILENAME = "L2_HF_Transformer_SFT.sh"


class HFTransformerSFT:
    """
    Test running HuggingFAce transformer SFT
    """

    @pytest.mark.run_only_on("GPU")
    def test_hf_transformer_sft(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_file_path = os.join(dir_path, HF_TRANSFORMER_SFT_FILENAME)
        with open(test_file_path, 'r') as file:
            test_cmd = file.read() 

        result = subprocess.run(
            test_cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
        )
        # Write output to file for debugging
        with open(output_file, "w") as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
   
    @pytest.mark.run_only_on("GPU")
    def test_hf_transformer_sft_nvfsdp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_file_path = os.join(dir_path, HF_TRANSFORMER_SFT_NVFSDP_FILENAME)
        with open(test_file_path, 'r') as file:
            test_cmd = file.read() 

        result = subprocess.run(
            test_cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
        )
        # Write output to file for debugging
        with open(output_file, "w") as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)
