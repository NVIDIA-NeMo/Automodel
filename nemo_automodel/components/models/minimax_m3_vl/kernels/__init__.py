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

"""Model-private MSA kernels, loaded lazily by _msa; no eager CuTe imports."""

from functools import lru_cache

import torch


@lru_cache
def sm_capability(device: torch.device) -> tuple[int, int]:
    """Return the memoized CUDA compute capability of ``device``.

    ``torch.cuda.get_device_capability`` costs about 1.6 us and one MSA backward calls it eight
    times: the SM100 guard plus four compile-cache keys, three of which are reached twice. A
    device's capability cannot change, so look it up once per device.

    Args:
        device: CUDA device taken from a tensor, which always carries an explicit index.

    Returns:
        The ``(major, minor)`` compute capability of that device.
    """
    return torch.cuda.get_device_capability(device)


def require_sm100(device: torch.device) -> None:
    """Reject any device the MSA kernels are not built for, before compiling or launching one.

    Args:
        device: CUDA device the caller is about to run MSA kernels on.

    Raises:
        NotImplementedError: If ``device`` is not SM100.
    """
    capability = sm_capability(device)
    if capability != (10, 0):
        raise NotImplementedError(
            "MiniMax M3 MSA first supports SM100 (compute capability 10.0) only; got compute capability "
            f"{capability[0]}.{capability[1]} on {device}. Use sparse_attn='generic' on this GPU."
        )
