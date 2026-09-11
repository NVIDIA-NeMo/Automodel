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

"""Model-private MSA kernels, bound lazily by ``msa_bindings.kernels``; no eager CuTe imports.

The SM100 kernel modules are CuTe DSL source and call ``require_cute_dsl`` before their first DSL
import, so a host without the msa extra sees ``UnavailableError`` instead of ``ModuleNotFoundError``.
"""

from functools import lru_cache

import torch

from nemo_automodel.shared.import_utils import UnavailableError, safe_import

MSA_KERNEL_IMPORT_ERROR = (
    "BackendConfig.sparse_attn='msa' backward requires nvidia-cutlass-dsl==4.6.2 and cuda-bindings from the "
    "msa optional dependency, which ship Linux wheels only. Install the project with uv sync --extra msa on a "
    "CUDA SM100 system."
)


@lru_cache
def sm_capability(device: torch.device) -> tuple[int, int]:
    """Return the memoized CUDA compute capability of ``device``.

    ``torch.cuda.get_device_capability`` costs about 1.6 us and ``require_sm100`` runs on every MSA
    forward; a device's capability cannot change, so look it up once per device.

    Args:
        device: CUDA device taken from a tensor, which always carries an explicit index.

    Returns:
        The ``(major, minor)`` compute capability of that device.
    """
    return torch.cuda.get_device_capability(device)


@lru_cache
def sm_count(device: torch.device) -> int:
    """Return the memoized streaming-multiprocessor count of ``device``; it sizes the backward CTA walk.

    Args:
        device: CUDA device taken from a tensor, which always carries an explicit index.

    Returns:
        The number of streaming multiprocessors of that device.
    """
    return torch.cuda.get_device_properties(device).multi_processor_count


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


def require_cute_dsl() -> None:
    """Refuse to bind the CuTe DSL in a kernel module on a host that cannot import it.

    The SM100 kernel modules are CuTe DSL source: their decorators, annotations and module constants
    need ``cutlass`` and ``cuda.bindings`` while the module loads, and the CI import walker imports
    every module of the package on hosts without the msa extra. Each kernel module calls this before
    its first DSL import so absence surfaces as ``UnavailableError``, the signal both the walker and
    ``msa_bindings.kernels`` understand. It is the software twin of ``require_sm100``: one gates
    the toolchain at import, the other the device at launch.

    Raises:
        UnavailableError: If ``cutlass`` or ``cuda.bindings.driver`` cannot be imported.
    """
    for module in ("cutlass", "cuda.bindings.driver"):
        available, _ = safe_import(module, msg=MSA_KERNEL_IMPORT_ERROR)
        if not available:
            raise UnavailableError(MSA_KERNEL_IMPORT_ERROR)
