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

TEST_FOLDER = "hf_transformer_finetune"
HF_TRANSFORMER_SFT_FILENAME = "L2_HF_Transformer_SFT.sh"
HF_TRANSFORMER_SFT_MegatronFSDP_FILENAME = "L2_HF_Transformer_SFT_MegatronFSDP.sh"
HF_TRANSFORMER_PEFT_FILENAME = "L2_HF_Transformer_PEFT.sh"
HF_TRANSFORMER_PEFT_MegatronFSDP_FILENAME = "L2_HF_Transformer_PEFT_MegatronFSDP.sh"
HF_TRANSFORMER_PEFT_NO_TOKENIZER_FILENAME = "L2_HF_Transformer_PEFT_no_tokenizer.sh"
HF_TRANSFORMER_SFT_NO_LOGITS_FILENAME = "L2_HF_Transformer_SFT_no_logits.sh"
HF_TRANSFORMER_QWEN3_MOE_CUSTOM_FILENAME = "L2_HF_Transformer_Qwen3_MoE_custom.sh"

class TestHFTransformerFinetune:
    def test_hf_transformer_sft(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_sft_megatron_fsdp(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_MegatronFSDP_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_peft(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_peft_megatron_fsdp(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_MegatronFSDP_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_peft_no_tokenizer(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_NO_TOKENIZER_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_sft_no_logits(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_NO_LOGITS_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_qwen3_moe_sdpa(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_QWEN3_MOE_CUSTOM_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)
