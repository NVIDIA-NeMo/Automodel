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

"""Tests for ``_backports/hf_utils.DTYPE_MAP`` FP8-scale extensions.

DeepSeek V4 (and other quantized HF checkpoints) emit FP8-scale tensors with
dtype tags ``F8_E5M2`` and ``F8_E8M0`` that the upstream
``safetensors.torch._TYPES`` map does not yet recognize.  The in-tree backport
extends ``DTYPE_MAP`` so the in-tree HF storage reader can decode them.
"""

import pytest
import torch

from nemo_automodel.components.checkpoint._backports.hf_utils import DTYPE_MAP


@pytest.mark.parametrize(
    "key, expected_dtype",
    [
        ("F8_E4M3", torch.float8_e4m3fn),
        ("F8_E5M2", torch.float8_e5m2),
        ("F8_E8M0", torch.float8_e8m0fnu),
    ],
)
def test_dtype_map_contains_fp8_dtypes(key, expected_dtype):
    """All three FP8 variants emitted by DSV4-style checkpoints must be mapped."""
    assert key in DTYPE_MAP, f"{key} missing from DTYPE_MAP"
    assert DTYPE_MAP[key] is expected_dtype


def test_dtype_map_preserves_existing_entries():
    """Adding FP8 entries must not break the existing standard-dtype mappings."""
    expected = {
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "BF16": torch.bfloat16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
    }
    for key, dtype in expected.items():
        assert DTYPE_MAP.get(key) is dtype, f"DTYPE_MAP[{key!r}] regressed"
