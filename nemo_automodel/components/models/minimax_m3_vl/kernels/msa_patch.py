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

"""Compatibility for official MiniMax-AI/MSA at 80434d7f67877c6570ca19cac444b84bc9855dac."""

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
