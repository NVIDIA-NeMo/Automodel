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

"""FP32 mixing paths: the block residual and the short convolution ignore an enclosing BF16 autocast."""

import pytest
import torch

from nemo_automodel.components.models.inkling2.text import (
    Inkling2BlockAttentionResidual,
    Inkling2ShortConvolution,
)

pytestmark = pytest.mark.cpu_only


def test_mixing_stays_fp32_under_bf16_autocast():
    """The single-GPU FSDP2 path wraps the forward in BF16 autocast; the residual must not follow it."""
    torch.manual_seed(0)
    module = Inkling2BlockAttentionResidual(64, num_queries=3)
    with torch.no_grad():
        module.query.normal_()
    block_states = torch.randn(1, 4, 2, 64, dtype=torch.bfloat16)
    partial = torch.randn(1, 4, 64, dtype=torch.bfloat16)
    plain = module(block_states, partial)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = module(block_states, partial)
    for a, b in zip(plain, cast):
        assert b.dtype is torch.float32
        torch.testing.assert_close(b, a, rtol=0, atol=0)


def test_short_convolution_stays_fp32_under_bf16_autocast():
    torch.manual_seed(0)
    conv = Inkling2ShortConvolution(32, 4)
    conv.init_weights(0.5)
    hidden = torch.randn(2, 9, 32, dtype=torch.bfloat16)
    mask = torch.ones(2, 9)
    plain = conv(hidden, mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = conv(hidden, mask)
    torch.testing.assert_close(cast, plain, rtol=0, atol=0)
