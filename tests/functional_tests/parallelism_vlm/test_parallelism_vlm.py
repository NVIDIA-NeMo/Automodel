# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""VLM parallelism parity tests (Gemma4 and Mistral3.5).

Split from ``parallelism/test_parallelism.py`` so the VLM parity runs can be scheduled
in a different CI job than the MoE and pipeline parity runs. The scripts, proxy
configs, and ``compare_parallel_parity.py`` stay in ``tests/functional_tests/parallelism``;
see that module for what each parity test checks.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "parallelism"
GEMMA4_PP2_PARITY_FILENAME = "L2_Parallelism_VLM_Gemma4_PP2_Parity.sh"
GEMMA4_TP2_PARITY_FILENAME = "L2_Parallelism_VLM_Gemma4_TP2_Parity.sh"
GEMMA4_KV_SHARED_AC_FILENAME = "L2_Parallelism_VLM_Gemma4_KVShared_AC.sh"
MISTRAL3P5_PP2_PARITY_FILENAME = "L2_Parallelism_VLM_Mistral3p5_PP2_Parity.sh"
MISTRAL3P5_TP2_PARITY_FILENAME = "L2_Parallelism_VLM_Mistral3p5_TP2_Parity.sh"


class TestVLMParallelismParity:
    def test_gemma4_pp2_parity(self):
        run_test_script(TEST_FOLDER, GEMMA4_PP2_PARITY_FILENAME)

    def test_gemma4_tp2_parity(self):
        run_test_script(TEST_FOLDER, GEMMA4_TP2_PARITY_FILENAME)

    def test_gemma4_kv_shared_activation_checkpointing(self):
        run_test_script(TEST_FOLDER, GEMMA4_KV_SHARED_AC_FILENAME)

    def test_mistral3p5_pp2_parity(self):
        run_test_script(TEST_FOLDER, MISTRAL3P5_PP2_PARITY_FILENAME)

    def test_mistral3p5_tp2_parity(self):
        run_test_script(TEST_FOLDER, MISTRAL3P5_TP2_PARITY_FILENAME)
