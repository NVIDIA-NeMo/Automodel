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

TEST_FOLDER = "hf_transformer_llm"
HF_TRANSFORMER_LLM_FSDP2_TP2_FILENAME = "L2_HF_Transformer_LLM_FSDP2_TP2.sh"
HF_TRANSFORMER_LLM_FSDP2_TP2_HF_TPPLAN_FILENAME = "L2_HF_Transformer_LLM_FSDP2_TP2_HF_TPPLAN.sh"
HF_TRANSFORMER_LLM_MegatronFSDP_TP2_FILENAME = "L2_HF_Transformer_LLM_MegatronFSDP_TP2.sh"
HF_TRANSFORMER_LLM_MegatronFSDP_TP2_HF_TPPLAN_FILENAME = "L2_HF_Transformer_LLM_MegatronFSDP_TP2_HF_TPPLAN.sh"
HF_TRANSFORMER_LLM_DDP_FILENAME = "L2_HF_Transformer_LLM_DDP.sh"



class TestHFTransformerLLM:
    @pytest.mark.pleasefixme
    def test_hf_transformer_llm_fsdp2_tp2(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_FSDP2_TP2_FILENAME)
    
    @pytest.mark.pleasefixme
    def test_hf_transformer_llm_fsdp2_tp2_hf_tpplan(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_FSDP2_TP2_HF_TPPLAN_FILENAME)

    @pytest.mark.pleasefixme
    def test_hf_transformer_llm_megatron_fsdp_tp2(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_MegatronFSDP_TP2_FILENAME)
    
    @pytest.mark.pleasefixme
    def test_hf_transformer_llm_megatron_fsdp_tp2_hf_tpplan(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_MegatronFSDP_TP2_HF_TPPLAN_FILENAME)

    def test_hf_transformer_llm_ddp(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_DDP_FILENAME)