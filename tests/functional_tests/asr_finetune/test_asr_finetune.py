# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

TEST_FOLDER = "asr_finetune"
ASR_WHISPER_SMALL_LIBRISPEECH_FILENAME = "L2_ASR_Whisper_Small_LibriSpeech.sh"
ASR_PARAKEET_CTC_LIBRISPEECH_FILENAME = "L2_ASR_Parakeet_CTC_LibriSpeech.sh"
ASR_WHISPER_SMALL_LIBRISPEECH_PEFT_FILENAME = "L2_ASR_Whisper_Small_LibriSpeech_PEFT.sh"
ASR_PARAKEET_CTC_LIBRISPEECH_PEFT_FILENAME = "L2_ASR_Parakeet_CTC_LibriSpeech_PEFT.sh"


class TestASRFinetune:
    """End-to-end functional tests for ASR training."""

    def test_asr_whisper_small_librispeech(self):
        """Test Whisper Small finetuning on LibriSpeech dataset.

        Behavior: Training script should complete successfully without errors.
        """
        try:
            run_test_script(TEST_FOLDER, ASR_WHISPER_SMALL_LIBRISPEECH_FILENAME)
        finally:
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_asr_parakeet_ctc_librispeech(self):
        """Test Parakeet CTC finetuning on LibriSpeech dataset.

        Behavior: Training script should complete successfully without errors.
        """
        try:
            run_test_script(TEST_FOLDER, ASR_PARAKEET_CTC_LIBRISPEECH_FILENAME)
        finally:
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_asr_whisper_small_librispeech_peft(self):
        """Test Whisper Small LoRA finetuning on LibriSpeech.

        Verifies that PEFT training completes successfully with frozen base model
        and trainable LoRA adapter weights. Checkpoint should contain only adapter
        parameters.

        Behavior: Training script should complete successfully without errors.
        """
        try:
            run_test_script(TEST_FOLDER, ASR_WHISPER_SMALL_LIBRISPEECH_PEFT_FILENAME)
        finally:
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_asr_parakeet_ctc_librispeech_peft(self):
        """Test Parakeet CTC LoRA finetuning on LibriSpeech.

        Verifies that PEFT training works with CTC loss computation and
        encoder-only architecture with frozen base model.

        Behavior: Training script should complete successfully without errors.
        """
        try:
            run_test_script(TEST_FOLDER, ASR_PARAKEET_CTC_LIBRISPEECH_PEFT_FILENAME)
        finally:
            shutil.rmtree("checkpoints/", ignore_errors=True)
