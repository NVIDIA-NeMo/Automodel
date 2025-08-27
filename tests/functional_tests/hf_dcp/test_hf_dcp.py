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

TEST_FOLDER = "hf_dcp"
DCP_FSDP2_CHECKPOINT_FILENAME = "L2_DCP_FSDP2_Checkpoint.sh"
DCP_VLM_FSDP2_CHECKPOINT_FILENAME = "L2_DCP_VLM_FSDP2_Checkpoint.sh"
HF_DCP_FSDP2_CHECKPOINT_FILENAME = "L2_HF_DCP_FSDP2_Checkpoint.sh"
HF_DCP_VLM_FSDP2_CHECKPOINT_FILENAME = "L2_HF_DCP_VLM_FSDP2_Checkpoint.sh"
HF_DCP_PP2_CHECKPOINT_FILENAME = "L2_DCP_PP2_Checkpoint.sh"
import shutil


class TestHFDCP:
    def test_dcp_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, DCP_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_dcp_vlm_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, DCP_VLM_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_dcp_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_DCP_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_dcp_vlm_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_DCP_VLM_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_dcp_pp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_DCP_PP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)
