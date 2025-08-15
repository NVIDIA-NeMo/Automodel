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

import pytest

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "hf_transformer_vlm"
HF_TRANSFORMER_VLM_FSDP2_TP2_FILENAME = "L2_HF_Transformer_VLM_FSDP2_TP2.sh"
HF_TRANSFORMER_VLM_MegatronFSDP_TP2_FILENAME = "L2_HF_Transformer_VLM_MegatronFSDP_TP2.sh"
HF_TRANSFORMER_VLM_FUSED_CE_SFT_FILENAME = "L2_HF_Transformer_VLM_Fused_CE_SFT.sh"
HF_TRANSFORMER_VLM_PEFT_FILENAME = "L2_HF_Transformer_VLM_PEFT.sh"
HF_TRANSFORMER_VLM_SFT_FILENAME = "L2_HF_Transformer_VLM_SFT.sh"
HF_TRANSFORMER_VLM_SFT_MegatronFSDP_FILENAME = "L2_HF_Transformer_VLM_SFT_megatronfsdp.sh"


class TestHFTransformerVLM:
    @pytest.mark.pleasefixme
    def test_hf_transformer_vlm_fsdp2_tp2(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_VLM_FSDP2_TP2_FILENAME)

    @pytest.mark.pleasefixme
    def test_hf_transformer_vlm_megatronfsdp_tp2(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_VLM_MegatronFSDP_TP2_FILENAME)

    def test_hf_transformer_vlm_fused_ce_sft(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_VLM_FUSED_CE_SFT_FILENAME)

    def test_hf_transformer_vlm_peft(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_VLM_PEFT_FILENAME)

    def test_hf_transformer_vlm_sft(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_VLM_SFT_FILENAME)

    @pytest.mark.pleasefixme
    def test_hf_transformer_vlm_sft_megatronfsdp(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_VLM_SFT_MegatronFSDP_FILENAME)
