# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the fused Triton soft cross-entropy kernel."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.triton.soft_cross_entropy import HAVE_TRITON, fused_soft_cross_entropy

pytestmark = [
    pytest.mark.skipif(not HAVE_TRITON, reason="Triton not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

# (batch, seq_len, vocab) shapes exercised by the forward/backward parity tests.
_PARITY_SHAPES = [
    (1, 128, 1024),
    (1, 256, 8192),
    (1, 1024, 32000),
    (1, 2048, 32000),
    (1, 4096, 32000),
    (1, 8192, 32000),
    (1, 16384, 32000),
    (1, 32768, 32000),
]


def _pytorch_reference(logits, target_probs, position_mask):
    """Reference PyTorch implementation."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token_loss = -(target_probs.float() * log_probs).sum(dim=-1)
    valid_mask = position_mask.squeeze(-1).float()
    valid_count = valid_mask.sum().clamp_min(1.0)
    return (per_token_loss * valid_mask).sum() / valid_count


def _make_inputs(batch, seq_len, vocab, dtype=torch.bfloat16, mask_ratio=0.8, device="cuda"):
    """Generate random test inputs."""
    logits = torch.randn(batch, seq_len, vocab, dtype=dtype, device=device)
    target_logits = torch.randn(batch, seq_len, vocab, dtype=torch.float32, device=device)
    target_probs = torch.softmax(target_logits, dim=-1).to(dtype)
    mask = (torch.rand(batch, seq_len, device=device) < mask_ratio).unsqueeze(-1).float()
    return logits, target_probs, mask


@pytest.mark.parametrize("batch,seq_len,vocab", _PARITY_SHAPES)
def test_forward_matches_pytorch(batch, seq_len, vocab):
    """Triton forward output matches PyTorch reference within tolerance."""
    logits, target_probs, mask = _make_inputs(batch, seq_len, vocab)

    ref_loss = _pytorch_reference(logits, target_probs, mask)
    triton_loss = fused_soft_cross_entropy(logits, target_probs, mask)

    torch.testing.assert_close(triton_loss, ref_loss, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("batch,seq_len,vocab", _PARITY_SHAPES)
def test_backward_matches_pytorch(batch, seq_len, vocab):
    """Triton backward gradients match PyTorch autograd."""
    logits_ref, target_probs, mask = _make_inputs(batch, seq_len, vocab)
    logits_ref.requires_grad_(True)

    logits_tri = logits_ref.detach().clone().requires_grad_(True)

    ref_loss = _pytorch_reference(logits_ref, target_probs, mask)
    ref_loss.backward()

    triton_loss = fused_soft_cross_entropy(logits_tri, target_probs, mask)
    triton_loss.backward()

    torch.testing.assert_close(logits_tri.grad, logits_ref.grad, rtol=1e-3, atol=1e-3)


def test_all_masked():
    """All positions masked produces zero loss and zero gradients."""
    logits, target_probs, _ = _make_inputs(1, 64, 1024)
    mask = torch.zeros(1, 64, 1, device="cuda")
    logits.requires_grad_(True)

    loss = fused_soft_cross_entropy(logits, target_probs, mask)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad.abs().max().item() == 0.0


def test_single_valid_position():
    """Single valid position produces correct loss."""
    logits, target_probs, _ = _make_inputs(1, 64, 1024)
    mask = torch.zeros(1, 64, 1, device="cuda")
    mask[0, 7, 0] = 1.0

    ref_loss = _pytorch_reference(logits, target_probs, mask)
    triton_loss = fused_soft_cross_entropy(logits, target_probs, mask)

    torch.testing.assert_close(triton_loss, ref_loss, rtol=1e-4, atol=1e-4)


def test_fp32_target_probs():
    """Works with fp32 target_probs (live training path)."""
    logits = torch.randn(1, 256, 8192, dtype=torch.bfloat16, device="cuda")
    target_probs = torch.softmax(torch.randn(1, 256, 8192, device="cuda"), dim=-1)  # fp32
    mask = torch.ones(1, 256, 1, device="cuda")

    ref_loss = _pytorch_reference(logits, target_probs, mask)
    triton_loss = fused_soft_cross_entropy(logits, target_probs, mask)

    torch.testing.assert_close(triton_loss, ref_loss, rtol=1e-4, atol=1e-4)


def test_one_hot_target():
    """One-hot targets (degenerate case) match standard CE."""
    B, S, V = 1, 128, 1024
    logits = torch.randn(B, S, V, dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(0, V, (B, S), device="cuda")
    target_probs = F.one_hot(labels, V).float()
    mask = torch.ones(B, S, 1, device="cuda")

    ref_loss = _pytorch_reference(logits, target_probs, mask)
    triton_loss = fused_soft_cross_entropy(logits, target_probs, mask)

    torch.testing.assert_close(triton_loss, ref_loss, rtol=1e-4, atol=1e-4)
