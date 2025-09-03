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

TEST_FOLDER = "data"
MEGATRON_BLENDED_DATASET_CHECKPOINT_TEST_FILENAME = "L2_Megatron_BlendedDataset_Checkpoint.sh" 
MEGATRON_SINGLE_DATASET_CHECKPOINT_TEST_FILENAME = "L2_Megatron_SingleDataset_Checkpoint.sh" 
MEGATRON_DP_SHARDING_TEST_FILENAME = "L2_Megatron_DP_Sharding_Test.sh"
MEGATRON_TP_SHARDING_TEST_FILENAME = "L2_Megatron_TP_Sharding_Test.sh"
MEGATRON_PREPROCESS_DATA_TEST_FILENAME = "L2_Megatron_Preprocess_Data.sh"

class TestDatasets:
    def test_megatron_blended_dataset_checkpoint(self):
        run_test_script(TEST_FOLDER, MEGATRON_BLENDED_DATASET_CHECKPOINT_TEST_FILENAME)

    def test_megatron_single_dataset_checkpoint(self):
        run_test_script(TEST_FOLDER, MEGATRON_SINGLE_DATASET_CHECKPOINT_TEST_FILENAME)

    def test_megatron_dp_sharding(self):
        run_test_script(TEST_FOLDER, MEGATRON_DP_SHARDING_TEST_FILENAME)

    def test_megatron_tp_sharding(self):
        run_test_script(TEST_FOLDER, MEGATRON_TP_SHARDING_TEST_FILENAME)

    def test_megatron_preprocess_data(self):
        run_test_script(TEST_FOLDER, MEGATRON_PREPROCESS_DATA_TEST_FILENAME)