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
"""``eval_safe_sdpa_kernel`` must exclude the cuDNN fused-MHA SDPA backend.

Regression for NVBugs 6293238: the cuDNN fused-MHA backend can fail
``mha_graph.execute(...).is_good() == false`` on the validation-forward shape,
so the eval guard must disable it while keeping flash / mem-efficient / math.
The ``torch.backends.cuda`` SDPA toggles are global flags and do not require a
GPU, so this runs on CPU-only CI.
"""

from torch.backends.cuda import (
    cudnn_sdp_enabled,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
)

from nemo_automodel._transformers.kernel_patches import eval_safe_sdpa_kernel


def test_eval_safe_sdpa_kernel_disables_cudnn():
    """Inside the guard cuDNN SDPA is off and the safe backends stay on."""
    with eval_safe_sdpa_kernel():
        assert cudnn_sdp_enabled() is False
        assert flash_sdp_enabled() is True
        assert mem_efficient_sdp_enabled() is True
        assert math_sdp_enabled() is True


def test_eval_safe_sdpa_kernel_restores_on_exit():
    """The context restores the prior cuDNN flag after exiting."""
    before = cudnn_sdp_enabled()
    with eval_safe_sdpa_kernel():
        assert cudnn_sdp_enabled() is False
    assert cudnn_sdp_enabled() is before
