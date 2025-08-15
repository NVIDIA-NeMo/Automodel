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

TEST_FOLDER = "hf_transformer"
HF_TRANSFORMER_FUSED_CE_SFT_FILENAME = "L2_HF_Transformer_Fused_CE_SFT.sh"
HF_TRANSFORMER_PACKED_SEQUENCE_FILENAME = "L2_HF_Transformer_Packed_Sequence.sh"
HF_TRANSFORMER_LLM_META_FILENAME = "L2_HF_Transformer_LLM_Meta.sh"

class TestHFTransformer:
    def test_hf_transformer_fused_ce_sft(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_FUSED_CE_SFT_FILENAME)

    def test_hf_transformer_packed_sequence(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_PACKED_SEQUENCE_FILENAME)

    def test_hf_transformer_llm_meta(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_META_FILENAME)
