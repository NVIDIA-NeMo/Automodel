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

"""Parallelism parity tests for production recipe topologies.

Each script runs a randomly-initialized proxy of a real recipe twice with the
same seed and data order, changing exactly one parallelism axis between the two
runs, and asserts both follow the same loss and gradient-norm trajectory.
Divergence means the sharding changed the computation.

The PP and TP tests compare a parallel run against a single-rank baseline. The
EP test compares two 2-rank runs that differ only in ``ep_size``, so the data
sharding and FSDP wrapping match and the bound can be tighter.

Covers the gap from PR #2983 (commit 00f40419).
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "parallelism"
GEMMA4_PP2_PARITY_FILENAME = "L2_Parallelism_VLM_Gemma4_PP2_Parity.sh"
GEMMA4_TP2_PARITY_FILENAME = "L2_Parallelism_VLM_Gemma4_TP2_Parity.sh"
GEMMA4_KV_SHARED_AC_FILENAME = "L2_Parallelism_VLM_Gemma4_KVShared_AC.sh"
PP_GRAD_ACCUM_PARITY_FILENAME = "L2_Parallelism_PP_Grad_Accum_Parity.sh"
DEEPSEEK_V4_PP2_PARITY_FILENAME = "L2_Parallelism_DeepSeekV4_PP2_Parity.sh"
DEEPSEEK_V4_EP2_PARITY_FILENAME = "L2_Parallelism_DeepSeekV4_EP2_Parity.sh"


class TestParallelismParity:
    def test_gemma4_pp2_parity(self):
        run_test_script(TEST_FOLDER, GEMMA4_PP2_PARITY_FILENAME)

    def test_gemma4_tp2_parity(self):
        run_test_script(TEST_FOLDER, GEMMA4_TP2_PARITY_FILENAME)

    def test_gemma4_kv_shared_activation_checkpointing(self):
        run_test_script(TEST_FOLDER, GEMMA4_KV_SHARED_AC_FILENAME)

    def test_pp_grad_accum_parity(self):
        run_test_script(TEST_FOLDER, PP_GRAD_ACCUM_PARITY_FILENAME)

    def test_deepseek_v4_pp2_parity(self):
        run_test_script(TEST_FOLDER, DEEPSEEK_V4_PP2_PARITY_FILENAME)

    def test_deepseek_v4_ep2_parity(self):
        run_test_script(TEST_FOLDER, DEEPSEEK_V4_EP2_PARITY_FILENAME)
