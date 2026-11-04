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
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.datasets.packing import build_packed_sequence_metadata
from nemo_automodel.components.models.common.packing import flatten_packed_sequence_metadata


def _packed_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scale: float,
    window_size: int,
) -> torch.Tensor:
    """Evaluate packed causal attention independently with PyTorch SDPA.

    Args:
        q: Query tensor of shape [batch, sequence, heads, qk_head_dim].
        k: Key tensor of shape [batch, sequence, kv_heads, qk_head_dim].
        v: Value tensor of shape [batch, sequence, kv_heads, v_head_dim].
        attention_mask: Indexed document mask of shape [batch, sequence].
        scale: Attention score scale.
        window_size: Visible keys including the current token, or -1 for unbounded.

    Returns:
        Tensor of shape [batch, sequence, heads, v_head_dim], with zeros at
        padding positions.
    """
    output = v.new_zeros((*q.shape[:-1], v.shape[-1]))
    for batch_idx in range(attention_mask.shape[0]):
        for document_id in range(1, int(attention_mask[batch_idx].max().item()) + 1):
            positions = torch.nonzero(attention_mask[batch_idx] == document_id, as_tuple=False).flatten()
            q_document = q[batch_idx, positions].transpose(0, 1).unsqueeze(0)
            k_document = k[batch_idx, positions].transpose(0, 1).unsqueeze(0)
            v_document = v[batch_idx, positions].transpose(0, 1).unsqueeze(0)
            local_positions = torch.arange(positions.numel(), device=q.device)
            allowed = local_positions[None, :] <= local_positions[:, None]
            if window_size > 0:
                allowed &= local_positions[None, :] > local_positions[:, None] - window_size
            with sdpa_kernel(SDPBackend.MATH):
                document_output = F.scaled_dot_product_attention(
                    q_document,
                    k_document,
                    v_document,
                    attn_mask=allowed,
                    scale=scale,
                    enable_gqa=True,
                )
            output[batch_idx, positions] = document_output.squeeze(0).transpose(0, 1)
    return output


@pytest.mark.parametrize("qk_head_dim,v_head_dim,kv_heads", [(192, 128, 4), (256, 256, 2)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("window_size", [-1, 16])
def test_native_fa4_forward_backward_matches_sdpa(
    qk_head_dim: int, v_head_dim: int, kv_heads: int, packed: bool, window_size: int
) -> None:
    """Dense/packed and full/local FA4 match SDPA outputs and input gradients."""
    if not torch.cuda.is_available():
        pytest.skip("FlashAttention-4 parity requires a CUDA device")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FlashAttention-4 parity requires a Blackwell SM100+ GPU")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    scale = qk_head_dim**-0.5
    attention_mask = torch.tensor(
        [[1] * 32 + [2] * 48 + [0] * 16, [1] * 24 + [2] * 24 + [3] * 48],
        device=device,
    )
    if not packed:
        attention_mask = torch.ones_like(attention_mask)

    torch.manual_seed(1234)
    q = torch.randn(2, 96, 4, qk_head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 96, kv_heads, qk_head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 96, kv_heads, v_head_dim, device=device, dtype=dtype, requires_grad=True)
    # Use a float32 math reference, independently of GPU fused-kernel dispatch.
    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()

    _, fa4 = initialize_attn_module_and_func(
        attn_impl="fa4",
        num_attention_heads=4,
        num_qk_channels=qk_head_dim,
        num_v_channels=v_head_dim,
        softmax_scale=scale,
    )
    metadata = {}
    if packed:
        packing_metadata = build_packed_sequence_metadata(attention_mask)
        packed_token_indices, cu_seqlens = flatten_packed_sequence_metadata(
            packing_metadata["packed_token_indices"],
            packing_metadata["cu_seqlens"],
            batch_size=2,
            sequence_length=96,
        )
        metadata = dict(
            packed_token_indices=packed_token_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=packing_metadata["max_seqlen"],
        )
    packed_q, packed_k, packed_v, fa4_kwargs = preprocess_args_and_kwargs_for_attn(
        q, k, v, attention_mask if packed else None, "fa4", window_size=(window_size, 0), **metadata
    )
    if qk_head_dim == v_head_dim == 256 and window_size > 0:
        # The pinned SM100/SM110 HD256 kernel explicitly rejects local attention.
        # Turn this into parity coverage when the upstream pin supports it.
        with pytest.raises(ValueError, match="head_dim=256 does not support local attention"):
            fa4(packed_q, packed_k, packed_v, **fa4_kwargs)
        return
    output = fa4(packed_q, packed_k, packed_v, **fa4_kwargs)
    reference = _packed_sdpa_reference(q_ref, k_ref, v_ref, attention_mask, scale=scale, window_size=window_size)

    # bf16 kernels use different tiled reduction orders; output and gradient
    # tolerances allow rounding differences while exposing window/layout errors.
    assert torch.isfinite(output).all(), "FA4 output contains nonfinite values"
    torch.testing.assert_close(output.float(), reference, atol=3e-2, rtol=3e-2)
    output_weight = torch.randn_like(output)
    (output * output_weight).sum().backward()
    (reference * output_weight).sum().backward()
    for name, actual, expected in (("q", q.grad, q_ref.grad), ("k", k.grad, k_ref.grad), ("v", v.grad, v_ref.grad)):
        assert torch.isfinite(actual).all(), f"FA4 {name} gradient contains nonfinite values"
        assert torch.isfinite(expected).all(), f"SDPA reference {name} gradient contains nonfinite values"
        torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)
