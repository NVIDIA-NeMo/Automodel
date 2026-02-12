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

TEST_FOLDER = "llm_pretrain_and_kd"
TP_OUTPUT_PARITY_MINIFIED = "L2_TP_Output_Parity_Minified.sh"
TP_OUTPUT_PARITY_ALL_CAUSAL_LM = "L2_TP_Output_Parity_All_Causal_LM.sh"


class TestTensorParallelOutputParityMinified:
    # def test_tp2_vs_tp1_logits_kl_div_minified_models(self):
    #     run_test_script(TEST_FOLDER, TP_OUTPUT_PARITY_MINIFIED)

    def test_tp2_vs_tp1_logits_all_causal_lm_architectures(self):
        """TP=2 parity for all AutoModelForCausalLM architectures (HF + NeMo)."""
        run_test_script(TEST_FOLDER, TP_OUTPUT_PARITY_ALL_CAUSAL_LM)

