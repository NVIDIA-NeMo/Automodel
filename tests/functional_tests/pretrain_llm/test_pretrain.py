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
import shutil

TEST_FOLDER = "pretrain_llm"

class TestPretrainLLM:
    def test_pretrain(self):
        try:
            run_test_script(TEST_FOLDER, "L2_Pretrain.sh")
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_pretrain_hf(self):
        try:
            run_test_script(TEST_FOLDER, "L2_Pretrain_HF.sh")
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_pretrain_moonlight_16b_te_2l(self):
        try:
            run_test_script(TEST_FOLDER, "L2_Pretrain_Moonlight_16B_TE_2L.sh")
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)
