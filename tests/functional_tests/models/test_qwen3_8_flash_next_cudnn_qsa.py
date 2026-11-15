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

"""Actual cuDNN QSA forward/backward against independent dense FP64 attention."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    qsa_gqa_attention,
    select_qsa_cuda_kernel,
    select_qsa_token_ids,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="cuDNN QSA integration requires Blackwell SM100/SM103",
)


def _routes(lengths: tuple[int, ...], batch: int = 1, budget: int = 8) -> torch.Tensor:
    total = sum(lengths)
    result = torch.full((batch, total, budget + 3), -1, device="cuda", dtype=torch.int32)
    start = 0
    for length in lengths:
        # Actual model selector: dense prefix, score-selected complete blocks,
        # forced causal tails, and non-four-aligned packed document starts.
        query = torch.randn(batch, length, 4, 8, device="cuda")
        key = torch.randn(batch, length // 4, 1, 8, device="cuda")
        ids = select_qsa_token_ids(
            query, key, torch.full((batch,), length, device="cuda"), token_budget=budget, compress_ratio=4
        )
        result[:, start : start + length, : ids.shape[-1]] = torch.where(ids >= 0, ids + start, -1)
        start += length
    return result


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, routes: torch.Tensor, scale: float) -> torch.Tensor:
    """Evaluate independent dense attention.

    Args:
        q: FP64 query [batch, sequence, query_heads, head_dim].
        k: FP64 key [batch, sequence, kv_heads, head_dim].
        v: FP64 value with the same layout as k.
        routes: Integer token IDs [batch, sequence, selected_tokens].
        scale: Attention score multiplier.

    Returns:
        FP64 output with q's layout.
    """
    batch, sequence = q.shape[:2]
    mask = torch.zeros(batch, sequence, sequence, device=q.device, dtype=torch.bool)
    # Independent route-set oracle, including empty rows.
    for b in range(batch):
        for row in range(sequence):
            ids = routes[b, row]
            mask[b, row, ids[(ids >= 0) & (ids < sequence)].long()] = True
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask[:, None],
        enable_gqa=True,
        scale=scale,
    ).transpose(1, 2)


def _assert_parity(actual: torch.Tensor, reference: torch.Tensor) -> None:
    """Check finite values and relative error.

    Args:
        actual: Output or gradient tensor with arbitrary leading dimensions.
        reference: FP64 reference tensor with the same shape and axis order.
    """
    assert actual.shape == reference.shape and torch.isfinite(actual).all()
    # Same independent-FP64 BF16 bound as the existing FA4 attention test.
    # FP64 is the numerical acceptance reference; Flex comparisons are diagnostic.
    error = (actual.double() - reference).norm()
    assert error <= 0.004 * reference.norm() + 1e-8, (float(error), float(reference.norm()))


@pytest.mark.parametrize(
    "heads,kv_heads,batch,lengths",
    [
        (24, 2, 2, (3, 13, 17)),
        (12, 1, 1, (1, 7, 21)),
    ],
)
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_forward_backward_packed_and_empty(
    heads: int, kv_heads: int, batch: int, lengths: tuple[int, ...], use_checkpoint: bool
) -> None:
    torch.manual_seed(173)
    sequence = sum(lengths)
    q = torch.randn(batch, sequence, heads, 512, device="cuda", dtype=torch.bfloat16)[..., ::2].requires_grad_()
    k = torch.randn(batch, sequence, kv_heads, 512, device="cuda", dtype=torch.bfloat16)[..., ::2].requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    routes = _routes(lengths, batch)
    routes[:, 0] = -1
    routes[:, -2:] = -1  # right-tail padding
    before = routes.clone()
    scale = 0.125

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Call the selected backend.

        Args:
            q: BF16 query [batch, sequence, query_heads, 256].
            k: BF16 key [batch, sequence, kv_heads, 256].
            v: BF16 value with k's layout.

        Returns:
            BF16 output with q's layout.
        """
        return qsa_gqa_attention(q, k, v, routes, cuda_kernel=select_qsa_cuda_kernel("cudnn"), softmax_scale=scale)

    out = checkpoint(attention, q, k, v, use_reentrant=False) if use_checkpoint else attention(q, k, v)
    dy = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), dy)
    ref_inputs = [t.detach().double().requires_grad_() for t in (q, k, v)]
    ref = _reference(*ref_inputs, routes, scale)
    refs = torch.autograd.grad(ref, ref_inputs, dy.double())
    for a, b in zip((out, *grads), (ref, *refs)):
        _assert_parity(a, b)
    assert torch.count_nonzero(out[:, 0]) == torch.count_nonzero(grads[0][:, 0]) == 0
    assert torch.count_nonzero(out[:, -2:]) == 0
    torch.testing.assert_close(routes, before, rtol=0, atol=0)


def test_all_empty_and_nondefault_stream() -> None:
    torch.manual_seed(19)
    inputs = [torch.randn(1, 9, h, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (24, 2, 2)]
    routes = torch.full((1, 9, 11), -1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = qsa_gqa_attention(*inputs, routes, cuda_kernel=select_qsa_cuda_kernel("cudnn"))
        grads = torch.autograd.grad(out, inputs, torch.randn_like(out))
    torch.cuda.current_stream().wait_stream(stream)
    for tensor in (out, *grads):
        assert torch.isfinite(tensor).all() and torch.count_nonzero(tensor) == 0


def test_deterministic_backward_rejected() -> None:
    inputs = [torch.randn(1, 4, h, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (12, 1, 1)]
    routes = _routes((4,))
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="not deterministic"):
            qsa_gqa_attention(*inputs, routes, cuda_kernel=select_qsa_cuda_kernel("cudnn"))
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
