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

"""FP32 mixing paths: einsum block residual equals SDPA; residual and short conv ignore BF16 autocast."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.inkling2.text import (
    Inkling2BlockAttentionResidual,
    Inkling2ShortConvolution,
    _unweighted_rms_norm,
)

pytestmark = pytest.mark.cpu_only


def _sdpa_reference(block_states: torch.Tensor, partial: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Depth attention as SDPA with the hidden axis as head_dim.

    Args:
        block_states: Tensor of shape [batch, sequence, blocks, hidden].
        partial: Tensor of shape [batch, sequence, hidden].
        query: Tensor of shape [queries, hidden].

    Returns:
        Tensor of shape [batch, sequence, queries, hidden].
    """
    stacked = _unweighted_rms_norm(torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float(), 1e-6)
    hidden = stacked.shape[-1]
    queries = query.float().view(-1, hidden).expand(*stacked.shape[:-2], -1, -1)
    return F.scaled_dot_product_attention(queries, stacked, stacked, scale=hidden**-0.5)


@pytest.mark.parametrize(("num_queries", "blocks"), [(1, 0), (1, 3), (3, 2)])
def test_einsum_mixing_matches_sdpa_forward_and_backward(num_queries, blocks):
    torch.manual_seed(0)
    module = Inkling2BlockAttentionResidual(64, num_queries=num_queries)
    with torch.no_grad():
        module.query.normal_()
    block_states = torch.randn(2, 5, blocks, 64, dtype=torch.float64).float().requires_grad_()
    partial = torch.randn(2, 5, 64).requires_grad_()
    upstream = torch.randn(2, 5, num_queries, 64)

    got = torch.stack(module(block_states, partial), dim=-2)
    got.backward(upstream)
    grads = [module.query.grad.clone(), block_states.grad.clone(), partial.grad.clone()]
    for tensor in (module.query, block_states, partial):
        tensor.grad = None
    want = _sdpa_reference(block_states, partial, module.query)
    want.backward(upstream)

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    for got_grad, want_grad in zip(grads, (module.query.grad, block_states.grad, partial.grad)):
        torch.testing.assert_close(got_grad, want_grad, rtol=1e-5, atol=1e-6)


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
