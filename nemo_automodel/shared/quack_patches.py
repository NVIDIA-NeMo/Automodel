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
"""quack-kernels compatibility patches for CuTe DSL kernels written against quack-kernels 0.6.4."""

from __future__ import annotations

import logging

from nemo_automodel.shared.import_utils import safe_import, safe_import_from

_logger = logging.getLogger(__name__)


def restore_quack_sub_packed_f32x2() -> None:
    """Bind ``quack.activation.sub_packed_f32x2`` when the installed quack-kernels lacks it.

    quack-kernels 0.6.5 removed the helper; the CuTe DSL ships the same packed fp32x2 subtraction
    (round to nearest even unless ``rnd`` is given) as ``cutlass.cute.arch.sub_packed_f32x2``.
    ffpa-attn 0.2.4 (its SM100 head-dim-512 forward) and the MSA revision the ``msa`` extra pins
    still call the quack name while their kernels are traced, so call this before they compile.
    It changes nothing when quack-kernels provides the helper or when quack or the DSL is missing.
    """
    has_activation, activation = safe_import("quack.activation")
    if not has_activation or hasattr(activation, "sub_packed_f32x2"):
        return
    has_helper, sub_packed_f32x2 = safe_import_from("cutlass.cute.arch", "sub_packed_f32x2")
    if not has_helper:
        return
    activation.sub_packed_f32x2 = sub_packed_f32x2
    _logger.debug("Bound quack.activation.sub_packed_f32x2 to cutlass.cute.arch.sub_packed_f32x2")
