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

"""Exercise the shared interleaved kernel through packed MLA and V4.1 callers."""

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import rope_triton
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb,
    apply_rotary_emb_qk,
    freqs_cis_from_position_ids,
)
from nemo_automodel.components.models.deepseek_v41 import attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not rope_triton.HAVE_TRITON, reason="packed Triton RoPE requires CUDA and Triton"
)


def _document_rotation(values: torch.Tensor, positions: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Apply the original eager complex arithmetic to one independent document.

    Args:
        values: Tensor of shape [tokens, heads, channels]; only the final 64 channels rotate.
        positions: Tensor of shape [tokens] containing this document's positions.
        inverse: Whether to conjugate the complex frequencies.

    Returns:
        Tensor of shape [tokens, heads, channels], preserving the input dtype.
    """
    inv_freq = 10000.0 ** (-torch.arange(0, 64, 2, device=values.device).float() / 64)
    angles = positions.float().unsqueeze(-1) * inv_freq
    frequencies = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)
    if inverse:
        frequencies = frequencies.conj()
    pairs = torch.view_as_complex(values[..., -64:].float().unflatten(-1, (-1, 2)).contiguous())
    product = pairs * frequencies
    # Separate component views avoid view_as_real's backward storage-offset
    # restriction when an odd prefix leaves the upstream slice unaligned.
    rotated = torch.stack((product.real, product.imag), dim=-1).flatten(-2).to(values.dtype)
    return torch.cat((values[..., :-64], rotated), dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("batched_positions", [False, True])
def test_packed_mla_matches_independent_documents(dtype: torch.dtype, padded: bool, batched_positions: bool) -> None:
    """Packing must preserve per-document phases, shared-key gradients, and padding storage."""
    torch.manual_seed(4346)
    lengths = (1, 7, 3, 18)
    extents = tuple((length + 3) // 4 * 4 if padded else length for length in lengths)
    positions = torch.cat(
        [torch.cat((torch.arange(length), torch.zeros(extent - length))) for length, extent in zip(lengths, extents)]
    ).to(device="cuda", dtype=torch.int64)
    tokens = sum(extents)
    # MLA Q is a trailing slice of a wider projection; K is shared across heads.
    q = torch.randn(tokens, 4, 192, device="cuda", dtype=dtype)[..., -64:].requires_grad_()
    k = torch.randn(tokens, 128, device="cuda", dtype=dtype)[..., -64:].requires_grad_()
    original_q, original_k = q.detach().clone(), k.detach().clone()
    inv_freq = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    frequencies = freqs_cis_from_position_ids(
        positions.unsqueeze(0) if batched_positions else positions, inv_freq, qkv_format="thd", for_fused_rope=False
    )
    cu_seqlens = torch.tensor((0, *torch.tensor(extents).cumsum(0).tolist()), device="cuda", dtype=torch.int32)
    with patch.object(rope_triton, "_apply_rope_torch", side_effect=AssertionError("must exercise the fused kernel")):
        actual_q, actual_k = apply_rotary_emb_qk(
            q, k.unsqueeze(1), frequencies, format="thd", rope_fusion=False, cu_seqlens=cu_seqlens
        )
        # Also cover the headless-key normalization used by direct helper callers.
        headless_k = apply_rotary_emb(k, frequencies, qkv_format="thd", unsqueeze_dim=2)
    torch.testing.assert_close(headless_k, actual_k.squeeze(1), rtol=0, atol=0)
    actual = (actual_q, actual_k.expand(-1, 4, -1))

    expected_q, expected_k = [], []
    for q_doc, k_doc, length in zip(q.split(extents), k.split(extents), lengths):
        doc_positions = torch.arange(length, device="cuda")
        expected_q.append(torch.cat((_document_rotation(q_doc[:length], doc_positions), q_doc[length:]), dim=0))
        expected_k.append(
            torch.cat(
                (_document_rotation(k_doc[:length].unsqueeze(1), doc_positions).squeeze(1), k_doc[length:]), dim=0
            )
        )
    expected = (torch.cat(expected_q), torch.cat(expected_k).unsqueeze(1).expand(-1, 4, -1))
    # Include padding in the random upstream gradient: it must remain an identity,
    # and expansion of the shared K must accumulate exactly the eager gradient.
    upstream = tuple(torch.randn_like(output) for output in actual)
    actual_grads = torch.autograd.grad(actual, (q, k), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
    torch.testing.assert_close(k, original_k, rtol=0, atol=0)
    assert actual_q.data_ptr() != q.data_ptr() and headless_k.data_ptr() != k.data_ptr()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("headed", [False, True])
def test_v41_packed_trailing_rotation_preserves_prefix(
    dtype: torch.dtype, inverse: bool, headed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relocated V4.1 kernel preserves packed Q/KV and inverse-output behavior."""
    torch.manual_seed(41)
    lengths = (1, 7, 19)
    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths])
    # An odd prefix tests the kernel's pair offset; channel stride tests its copy.
    shape = (1, sum(lengths), 3, 258) if headed else (1, sum(lengths), 258)
    values = torch.randn(*shape, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    inv_freq = 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
    angles = (positions.float().unsqueeze(-1) * inv_freq).unsqueeze(0)
    monkeypatch.setattr(attention, "_ROPE_IMPL", "triton")
    with patch.object(rope_triton, "_apply_rope_torch", side_effect=AssertionError("must exercise the fused kernel")):
        actual = attention._apply_rope(values, angles, inverse=inverse)
    docs = values[0] if headed else values[0].unsqueeze(1)
    expected = torch.cat(
        [
            _document_rotation(document, torch.arange(length, device="cuda"), inverse=inverse)
            for document, length in zip(docs.split(lengths), lengths)
        ]
    )
    expected = expected.unsqueeze(0) if headed else expected.squeeze(1).unsqueeze(0)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, values, upstream)[0]
    expected_grad = torch.autograd.grad(expected, values, upstream)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., :-64], values[..., :-64], rtol=0, atol=0)
    assert actual.is_contiguous() and actual.data_ptr() != values.data_ptr()
