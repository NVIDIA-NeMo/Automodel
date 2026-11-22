# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Packed MoE text metadata must not replace vision-frame boundaries."""

import pytest
import torch

from tests.unit_tests.models.qwen3_5.test_packed_vision import check_packed_media_parity

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("media", ["image", "video", "both", "text"])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pack_sizes,hybrid", [((2,), False), ((2, 1), True)])
def test_packed_moe_vision(
    media: str, precomputed: bool, dtype: torch.dtype, pack_sizes: tuple[int, ...], hybrid: bool
) -> None:
    pytest.importorskip("flash_attn")
    if hybrid:
        pytest.importorskip("fla.ops.gated_delta_rule")
        pytest.importorskip("causal_conv1d")
    check_packed_media_parity(
        media, precomputed, device="cuda", dtype=dtype, pack_sizes=pack_sizes, hybrid=hybrid, moe=True
    )
