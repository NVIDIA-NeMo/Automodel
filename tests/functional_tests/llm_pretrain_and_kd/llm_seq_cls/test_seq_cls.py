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

import shutil

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "llm_seq_cls"
SEQ_CLS_MOCK_FILENAME = "L2_Seq_Cls_Mock.sh"


class TestSequenceClassification:
    def test_seq_cls_mock(self):
        try:
            run_test_script(TEST_FOLDER, SEQ_CLS_MOCK_FILENAME)
        finally:
            shutil.rmtree("checkpoints_seq_cls_test/", ignore_errors=True)
