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

"""Pytest wrapper for the 2-GPU ERNIE 5.5 packed context-parallel training check."""

import pytest

from tests.functional_tests.moe.test_ernie5_5_ep_training import DSA_BACKEND_VARIABLE, _cudnn_dsa_skip_reason
from tests.utils.test_utils import run_test_script

TEST_FOLDER = "context_parallel"
ERNIE5_5_PACKED_CP2 = "L2_CP_Ernie5_5_Packed_Test.sh"


class TestErnie5_5PackedCP2Training:
    def test_cp2_matches_cp1_with_eager_attention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DSA_BACKEND_VARIABLE, "eager")
        run_test_script(TEST_FOLDER, ERNIE5_5_PACKED_CP2)

    def test_cp2_matches_cp1_with_cudnn_attention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reason = _cudnn_dsa_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        monkeypatch.setenv(DSA_BACKEND_VARIABLE, "cudnn")
        run_test_script(TEST_FOLDER, ERNIE5_5_PACKED_CP2)
