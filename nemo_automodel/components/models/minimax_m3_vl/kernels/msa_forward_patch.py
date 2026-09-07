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

"""Compatibility for official MiniMax-AI/MSA at 80434d7f67877c6570ca19cac444b84bc9855dac.

Two unrelated kinds of patch live here, and they must not be confused:

* ``_patch_msa_fmax`` changes **numerical behaviour** -- it rebinds a scalar helper so the CuTe DSL
  4.6.2 binding is used instead of one that only exists under an older CUDA flavour.
* ``_patch_msa_jit_gencode`` changes **build behaviour** only -- it drops a device target the local
  nvcc cannot parse. It never touches what the compiled kernels compute.

Both are removed once the pinned MSA revision carries the fix upstream.
"""

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from nemo_automodel.shared.import_utils import safe_import


def _patch_msa_fmax(sparse_module: ModuleType) -> None:
    """Patch only the loaded MSA-owned utils before JIT; preserve fp32, third operand and loc/ip."""
    available, utils = safe_import("src.common.utils")
    expected_path = Path(sparse_module.__file__).resolve().parent / "cute/src/common/utils.py"
    if not available or Path(utils.__file__).resolve() != expected_path:
        raise ImportError(
            "MSA compatibility patch requires its own src.common.utils; check for a conflicting src package"
        )

    @utils.dsl_user_op
    def fmax(
        a: float | utils.Float32, b: float | utils.Float32, c: float | utils.Float32 | None = None, *, loc=None, ip=None
    ) -> utils.Float32:
        """Emit the two- or three-input scalar fp32 maximum using the 4.6.2 binding."""
        return utils.Float32(
            utils.nvvm.fmax(
                utils.Float32(a).ir_value(loc=loc, ip=ip),
                utils.Float32(b).ir_value(loc=loc, ip=ip),
                c=utils.Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )

    utils.fmax = fmax


# sm_103a needs nvcc 12.9; MSA hard-codes both SM100 targets in a joined string with no module-level
# constant to override (jit.py:200-201), so an older toolkit fails the whole compilation.
_SM103A_TARGET = "-gencode=arch=compute_103a,code=sm_103a"
_MIN_SM103A_NVCC = (12, 9)


def _nvcc_release(cuda_home: str) -> tuple[int, int]:
    """Return the (major, minor) release of the nvcc that MSA's JIT will invoke."""
    output = subprocess.run(  # noqa: S603
        [os.path.join(cuda_home, "bin", "nvcc"), "--version"], capture_output=True, text=True, check=True
    ).stdout
    major, minor = re.search(r"release (\d+)\.(\d+)", output).groups()
    return int(major), int(minor)


def _patch_msa_jit_gencode(jit_module: ModuleType) -> None:
    """Drop the sm_103a target from MSA's own JIT flags when the local nvcc cannot parse it.

    Args:
        jit_module: The loaded ``fmha_sm100.jit`` module whose ``_get_nvcc_flags`` is wrapped.

    Raises:
        ImportError: If the module is not MSA's own ``jit``, so a name collision cannot silently
            patch someone else's compiler flags.
    """
    if getattr(jit_module, "__name__", None) != "fmha_sm100.jit" or not hasattr(jit_module, "_get_nvcc_flags"):
        raise ImportError("MSA compatibility patch requires MSA's own fmha_sm100.jit module")
    original = jit_module._get_nvcc_flags

    # Probed lazily: MSA reaches these flags only when a variant has to be built, so a container
    # with a warm JIT cache and no nvcc must still be able to import and run.
    @lru_cache(maxsize=1)
    def _needs_patch() -> bool:
        return _nvcc_release(jit_module._get_cuda_home()) < _MIN_SM103A_NVCC

    def _get_nvcc_flags(cache_dir: str, fmha: bool = True) -> str:
        flags = original(cache_dir, fmha)
        return flags.replace(_SM103A_TARGET, "") if _needs_patch() else flags

    jit_module._get_nvcc_flags = _get_nvcc_flags
