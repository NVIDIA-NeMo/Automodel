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

TEST_FOLDER = "retrieval"
CROSS_ENCODER_FINETUNE_FILENAME = "L2_Cross_Encoder_Finetune.sh"
RETRIEVAL_DATALOADER_CHECKPOINT_FILENAME = "L2_Retrieval_Dataloader_Checkpoint.sh"


class TestCrossEncoderFinetune:
    def test_cross_encoder_finetune(self):
        try:
            run_test_script(TEST_FOLDER, CROSS_ENCODER_FINETUNE_FILENAME)
        finally:
            shutil.rmtree("/workspace/output/cross_encoder_inline/", ignore_errors=True)


class TestRetrievalDataloaderCheckpoint:
    def test_retrieval_dataloader_checkpoint(self):
        run_test_script(TEST_FOLDER, RETRIEVAL_DATALOADER_CHECKPOINT_FILENAME)
