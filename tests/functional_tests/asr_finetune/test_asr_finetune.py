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

TEST_FOLDER = "asr_finetune"
ASR_WHISPER_SMALL_LIBRISPEECH_FILENAME = "L2_ASR_Whisper_Small_LibriSpeech.sh"
ASR_PARAKEET_CTC_LIBRISPEECH_FILENAME = "L2_ASR_Parakeet_CTC_LibriSpeech.sh"


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
