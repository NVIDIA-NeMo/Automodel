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

import pytest
import torch

from nemo_automodel.components.distributed import init_utils


def test_initialize_distributed_reports_cuda_device_when_already_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-initialized process group with visible CUDA reports a ``torch.device``.

    The device used to be the bare index, so GPU 0 was falsy and
    ``dist_env.device or torch.device("cpu")`` silently picked the CPU.
    """
    monkeypatch.setattr(init_utils.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(init_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(init_utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(init_utils.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(init_utils.torch.distributed, "get_world_size", lambda: 1)

    info = init_utils.initialize_distributed("nccl")

    assert info.device == torch.device("cuda", 0)
    assert (info.device or torch.device("cpu")).type == "cuda"
