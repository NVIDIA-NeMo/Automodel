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

"""Blackwell numerical coverage for native packed FlashAttention-4."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.shared.packed_sequence import get_unpad_data


def _packed_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Evaluate packed causal attention independently with PyTorch SDPA.

    Args:
        q: Query tensor of shape [batch, sequence, heads, head_dim].
        k: Key tensor of shape [batch, sequence, heads, head_dim].
        v: Value tensor of shape [batch, sequence, heads, head_dim].
        attention_mask: Indexed document mask of shape [batch, sequence].
        scale: Attention score scale.

    Returns:
        Tensor of shape [batch, sequence, heads, head_dim], with zeros at
        padding positions.
    """
    output = torch.zeros_like(q)
    for batch_idx in range(attention_mask.shape[0]):
        for document_id in range(1, int(attention_mask[batch_idx].max().item()) + 1):
            positions = torch.nonzero(attention_mask[batch_idx] == document_id, as_tuple=False).flatten()
            q_document = q[batch_idx, positions].transpose(0, 1).unsqueeze(0)
            k_document = k[batch_idx, positions].transpose(0, 1).unsqueeze(0)
            v_document = v[batch_idx, positions].transpose(0, 1).unsqueeze(0)
            document_output = F.scaled_dot_product_attention(
                q_document,
                k_document,
                v_document,
                is_causal=True,
                scale=scale,
            )
            output[batch_idx, positions] = document_output.squeeze(0).transpose(0, 1)
    return output


def test_native_fa4_packed_forward_backward_matches_sdpa() -> None:
    """Native packed FA4 matches independent SDPA outputs and input gradients."""
    if not torch.cuda.is_available():
        pytest.skip("FlashAttention-4 parity requires a CUDA device")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FlashAttention-4 parity requires a Blackwell SM100+ GPU")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 64
    scale = head_dim**-0.5
    attention_mask = torch.tensor(
        [[1] * 32 + [2] * 48 + [0] * 16, [1] * 24 + [2] * 24 + [3] * 48],
        device=device,
    )
    indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask)

    torch.manual_seed(1234)
    q = torch.randn(2, 96, 4, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 96, 4, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 96, 4, head_dim, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    _, fa4 = initialize_attn_module_and_func(
        attn_impl="fa4",
        num_attention_heads=4,
        num_qk_channels=head_dim,
        num_v_channels=head_dim,
        softmax_scale=scale,
    )
    packed_q, packed_k, packed_v, fa4_kwargs = preprocess_args_and_kwargs_for_attn(
        q,
        k,
        v,
        attention_mask,
        "fa4",
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        _fa4_unpad_indices=indices,
    )
    output = fa4(packed_q, packed_k, packed_v, **fa4_kwargs)
    reference = _packed_sdpa_reference(q_ref, k_ref, v_ref, attention_mask, scale=scale)

    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)
    output_weight = torch.randn_like(output)
    (output * output_weight).sum().backward()
    (reference * output_weight).sum().backward()
    torch.testing.assert_close(q.grad, q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=5e-2, rtol=5e-2)
