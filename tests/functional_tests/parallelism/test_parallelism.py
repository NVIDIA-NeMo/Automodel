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

"""Parallel-vs-single-rank parity tests for production recipe topologies.

Each script runs a randomly-initialized proxy of a real recipe twice with the
same seed and data order -- once on a single rank, once with one parallelism
axis enabled -- and asserts both follow the same loss and gradient-norm
trajectory. Divergence means the sharding changed the computation.

These cover the gap that let commit 00f40419 reach main: every pre-existing PP
test in GitHub CI ran stock HuggingFace Mixtral through the generic pipeline
path at ``tp_size=1``, so none of them failed while the real gemma4_31b and
mistral3p5 recipes broke in nemo-ci.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "parallelism"
GEMMA4_PP2_PARITY_FILENAME = "L2_Parallelism_VLM_Gemma4_PP2_Parity.sh"
GEMMA4_TP2_PARITY_FILENAME = "L2_Parallelism_VLM_Gemma4_TP2_Parity.sh"


class TestParallelismParity:
    def test_gemma4_pp2_parity(self):
        run_test_script(TEST_FOLDER, GEMMA4_PP2_PARITY_FILENAME)

    def test_gemma4_tp2_parity(self):
        run_test_script(TEST_FOLDER, GEMMA4_TP2_PARITY_FILENAME)
