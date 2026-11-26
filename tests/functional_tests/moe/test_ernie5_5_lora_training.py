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

"""Pytest wrapper for the 2-GPU ERNIE 5.5 LoRA training check with expert parallelism."""

import pytest

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "moe"
ERNIE5_5_EP2_LORA = "L2_Ernie5_5_MoE_EP2_LoRA.sh"


class TestErnie5_5EP2LoRATraining:
    # A run takes about two minutes. The script limits its own run to 600 seconds, and the launcher then takes about
    # 30 seconds to stop the ranks. This bound only ends the test when the script does not return; it does not stop
    # the launcher or the ranks.
    @pytest.mark.timeout(900)
    def test_trains_adapters_only_and_saves_them_with_hybridep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ERNIE5_5_DISPATCHER", "hybridep")
        run_test_script(TEST_FOLDER, ERNIE5_5_EP2_LORA)
