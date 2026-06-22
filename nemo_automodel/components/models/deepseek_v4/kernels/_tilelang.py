# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""TileLang import shim for vendored DeepSeek V4 kernels."""

from __future__ import annotations

from functools import wraps
from typing import Any

from nemo_automodel.shared.import_utils import UnavailableError

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError as _e:
    HAS_TILELANG = False
    _MSG = f"tilelang is required for DeepSeek V4 TileLang kernels: {_e}"

    def _unavailable_kernel(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            raise UnavailableError(_MSG)

        return wrapper

    def _phony_jit(*args: Any, **kwargs: Any):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _unavailable_kernel(args[0])

        def decorate(fn):
            return _unavailable_kernel(fn)

        return decorate

    def _identity_decorator(fn):
        return fn

    def _next_power_of_2(value: int) -> int:
        return 1 << (value - 1).bit_length()

    def _cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    class _PassConfigKey:
        TL_DISABLE_TMA_LOWER = "TL_DISABLE_TMA_LOWER"
        TL_DISABLE_WARP_SPECIALIZED = "TL_DISABLE_WARP_SPECIALIZED"
        TL_ENABLE_FAST_MATH = "TL_ENABLE_FAST_MATH"

    class _Math:
        next_power_of_2 = staticmethod(_next_power_of_2)

    class _PhonyTileLang:
        PassConfigKey = _PassConfigKey
        cdiv = staticmethod(_cdiv)
        jit = staticmethod(_phony_jit)
        math = _Math()

    class _PhonyLanguage:
        bfloat16 = "bfloat16"
        float = "float"
        float32 = "float32"
        int32 = "int32"
        prim_func = staticmethod(_identity_decorator)

    tilelang = _PhonyTileLang()
    T = _PhonyLanguage()
