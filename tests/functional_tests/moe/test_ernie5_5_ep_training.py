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

"""Pytest wrapper for the 2-GPU ERNIE 5.5 expert-parallel training check."""

import pytest
import torch

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "moe"
ERNIE5_5_EP2 = "L2_Ernie5_5_MoE_EP2.sh"
DSA_BACKEND_VARIABLE = "ERNIE5_5_DSA_BACKEND"


def _cudnn_dsa_skip_reason() -> str | None:
    """Return why the cuDNN DSA leg cannot run on this machine, or None when it can."""
    if torch.cuda.device_count() < 2:
        return "requires two CUDA GPUs"
    if any(torch.cuda.get_device_capability(index)[0] != 10 for index in range(2)):
        return "the cuDNN DSA path is validated on SM100-class GPUs (GB200) only"
    from nemo_automodel.components.models.common.cudnn_sparse_attention import is_cudnn_sparse_attention_available
    from nemo_automodel.components.models.glm_moe_dsa.kernels.cudnn_dsa import is_cudnn_dsa_available

    if not (is_cudnn_sparse_attention_available() and is_cudnn_dsa_available()):
        return "requires nvidia-cudnn-frontend[cutedsl] and FlashMLA"
    return None


class TestErnie5_5EP2Training:
    def test_trains_with_eager_dsa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DSA_BACKEND_VARIABLE, "eager")
        run_test_script(TEST_FOLDER, ERNIE5_5_EP2)

    def test_trains_with_cudnn_dsa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reason = _cudnn_dsa_skip_reason()
        if reason is not None:
            pytest.skip(reason)
        monkeypatch.setenv(DSA_BACKEND_VARIABLE, "cudnn")
        run_test_script(TEST_FOLDER, ERNIE5_5_EP2)
