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

CUTLASS DSL 4.6.2 infers the result type of ``nvvm.fmax`` for both CUDA
flavors, but MSA passes an explicit type on CUDA 12.9. QuACK 0.6.4 needs
the same DSL version; its compatibility change is the dependency pin only.
Remove this patch when the pinned official MSA uses the inferred-result call.
"""

from pathlib import Path
from types import ModuleType

from nemo_automodel.shared.import_utils import safe_import


def _patch_msa_fmax(sparse_module: ModuleType) -> None:
    """Install the scalar fix after loading MSA and before its first JIT compile.

    MSA loads its CuTe sources as top-level ``src`` modules. Verify ownership
    before replacing the helper shared by its softmax consumers. The replacement
    lives for the process lifetime, like the existing cached forward resolver;
    neither the installed source files nor CUTLASS globals are modified.

    Args:
        sparse_module: The loaded ``fmha_sm100.sparse`` module.

    Raises:
        ImportError: If MSA's helper is missing or a foreign ``src`` module shadows it.
    """
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
