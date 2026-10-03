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

"""Bindings to official MiniMax-AI/MSA (the revision the ``msa`` extra pins) and to the local SM100 backward.

``kernels()`` is the one place MSA reaches outside this package. It binds the four official entry
points and the local backward launcher on first use, applies the two compatibility patches the pin
needs, and caches the bundle for the process. Nothing is imported while this module loads, so it
imports on any host; a host without the msa extra learns so from ``UnavailableError`` at the first
call and never earlier, which is the contract the CI import walker checks.

Two unrelated kinds of patch live here, and they must not be confused:

* ``_patch_fmax`` changes **numerical behaviour** -- it rebinds a scalar helper so the CuTe DSL
  4.6.2 binding is used instead of one that only exists under an older CUDA flavour.
* ``_patch_jit_gencode`` changes **build behaviour** only -- it drops a device target the local
  nvcc cannot parse. It never touches what the compiled kernels compute.

Both are removed once the pinned MSA revision carries the fix upstream.
"""

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from nemo_automodel.components.models.minimax_m3_vl.kernels import MSA_KERNEL_IMPORT_ERROR
from nemo_automodel.shared.import_utils import UnavailableError, safe_import, safe_import_from

_MSA_IMPORT_ERROR = (
    "BackendConfig.sparse_attn='msa' requires the fixed fmha-sm100 optional dependency. "
    "Install the project with uv sync --extra msa on a CUDA SM100 system; the MSA revision "
    "must be compatible with nvidia-cutlass-dsl==4.6.2."
)
_BACKWARD_MODULE = "nemo_automodel.components.models.minimax_m3_vl.kernels.msa_backward_sm100"


@dataclass(frozen=True, slots=True)
class MSAKernels:
    """The launchers MSA calls: the official CSR builder, flat forward, scorer and planner, and the local backward."""

    build_k2q_csr: Callable[..., Any]
    sparse_atten_func: Callable[..., Any]
    fmha_sm100: Callable[..., Any]
    fmha_sm100_plan: Callable[..., Any]
    run_backward: Callable[..., Any]


@lru_cache(maxsize=1)
def kernels() -> MSAKernels:
    """Bind the MSA launchers once per process, patching the official package on the way.

    The official package and the CuTe DSL are imported here and nowhere earlier; the backward module
    raises ``UnavailableError`` from ``require_cute_dsl`` itself when the DSL is missing. A failed bind
    is not cached, so a later call retries.

    Returns:
        The five launchers.

    Raises:
        UnavailableError: If the msa extra (official MSA, nvidia-cutlass-dsl and cuda-bindings) is not
            installed, naming ``uv sync --extra msa``.
        ImportError: If a conflicting ``src`` package or a foreign ``fmha_sm100.jit`` shadows the module a
            patch must own.
    """
    has_sparse, sparse = safe_import("fmha_sm100.sparse", msg=_MSA_IMPORT_ERROR)
    has_jit, jit = safe_import("fmha_sm100.jit", msg=_MSA_IMPORT_ERROR)
    has_csr, build_k2q_csr = safe_import_from("fmha_sm100.sparse", "build_k2q_csr", msg=_MSA_IMPORT_ERROR)
    has_attn, sparse_atten_func = safe_import_from("fmha_sm100.sparse", "sparse_atten_func", msg=_MSA_IMPORT_ERROR)
    has_score, fmha_sm100 = safe_import_from("fmha_sm100", "fmha_sm100", msg=_MSA_IMPORT_ERROR)
    has_plan, fmha_sm100_plan = safe_import_from("fmha_sm100", "fmha_sm100_plan", msg=_MSA_IMPORT_ERROR)
    if not (has_sparse and has_jit and has_csr and has_attn and has_score and has_plan):
        raise UnavailableError(_MSA_IMPORT_ERROR)
    has_backward, run_backward = safe_import_from(_BACKWARD_MODULE, "run_backward", msg=MSA_KERNEL_IMPORT_ERROR)
    if not has_backward:
        raise UnavailableError(MSA_KERNEL_IMPORT_ERROR)
    _patch_fmax(sparse)
    _patch_jit_gencode(jit)
    return MSAKernels(build_k2q_csr, sparse_atten_func, fmha_sm100, fmha_sm100_plan, run_backward)


def _patch_fmax(sparse_module: ModuleType) -> None:
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
    output = subprocess.run(
        [os.path.join(cuda_home, "bin", "nvcc"), "--version"], capture_output=True, text=True, check=True
    ).stdout
    major, minor = re.search(r"release (\d+)\.(\d+)", output).groups()
    return int(major), int(minor)


def _patch_jit_gencode(jit_module: ModuleType) -> None:
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
