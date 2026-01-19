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

TEST_FOLDER = "loss/"
TE_PARALLEL_CE_DTENSOR_TP2_FILENAME = "L2_TEParallelCrossEntropy_DTENSOR_TP2.sh"


class TestTEParallelCrossEntropyDTensor:
    def test_te_parallel_cross_entropy_dtensor_tp2(self):
        run_test_script(TEST_FOLDER, TE_PARALLEL_CE_DTENSOR_TP2_FILENAME)

