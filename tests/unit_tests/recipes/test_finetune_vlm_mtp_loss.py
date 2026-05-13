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

"""Tests for the VLM recipe's MTP auxiliary-loss helper.

Verifies that ``calculate_mtp_loss``:

* sums per-depth cross-entropies and applies the requested scaling factor;
* masks the trailing ``k+1`` rolled-positions per depth (so end-of-sequence
  padding never leaks gradients);
* produces gradients only on the inputs (hidden states + lm_head), confirming
  the autograd graph is intact.

CPU-only; uses a tiny LM head module and the project's
``MaskedCrossEntropy`` loss to mirror real recipe behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes.vlm.finetune import calculate_mtp_loss


class _TinyLM(nn.Module):
    """Minimal stand-in exposing get_output_embeddings -> shared LM head."""

    def __init__(self, hidden: int = 8, vocab: int = 16):
        super().__init__()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


def test_calculate_mtp_loss_runs_and_scales():
    torch.manual_seed(0)
    B, S, H, V = 2, 6, 8, 16
    D = 2

    model = _TinyLM(hidden=H, vocab=V)
    loss_fn = MaskedCrossEntropy()

    mtp_per_depth_h = [torch.randn(B, S, H, requires_grad=True) for _ in range(D)]
    labels = torch.randint(0, V, (B, S))
    num_label_tokens = int((labels != -100).sum().item())

    loss_05 = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=mtp_per_depth_h,
        labels=labels,
        model=model,
        scaling_factor=0.5,
        num_label_tokens=num_label_tokens,
    )
    loss_10 = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=mtp_per_depth_h,
        labels=labels,
        model=model,
        scaling_factor=1.0,
        num_label_tokens=num_label_tokens,
    )

    assert torch.isfinite(loss_05)
    assert torch.isfinite(loss_10)
    # Doubling the scaling factor must double the loss exactly.
    assert torch.allclose(loss_10, loss_05 * 2.0, rtol=1e-5, atol=1e-6)


def test_calculate_mtp_loss_grads_flow_to_hidden_states():
    torch.manual_seed(1)
    B, S, H, V = 1, 5, 8, 16

    model = _TinyLM(hidden=H, vocab=V)
    loss_fn = MaskedCrossEntropy()

    h0 = torch.randn(B, S, H, requires_grad=True)
    labels = torch.randint(0, V, (B, S))
    num_label_tokens = int((labels != -100).sum().item())

    loss = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=[h0],
        labels=labels,
        model=model,
        scaling_factor=0.1,
        num_label_tokens=num_label_tokens,
    )
    loss.backward()
    assert h0.grad is not None
    # lm_head also receives gradient because logits = lm_head(h_k).
    assert model.lm_head.weight.grad is not None


def test_calculate_mtp_loss_masks_tail_positions():
    """Depth k masks the last (k+1) positions; the loss must not depend on
    the labels at those positions."""
    torch.manual_seed(2)
    B, S, H, V = 1, 6, 8, 16

    model = _TinyLM(hidden=H, vocab=V)
    loss_fn = MaskedCrossEntropy()

    h0 = torch.randn(B, S, H)
    labels_a = torch.randint(0, V, (B, S))
    labels_b = labels_a.clone()
    # Mutate ONLY the last position — depth 0 masks the last 1 position after
    # rolling left by 1, so the loss for depth 0 must be invariant to this.
    labels_b[..., -1] = (labels_b[..., -1] + 7) % V
    num_label_tokens_a = int((labels_a != -100).sum().item())
    num_label_tokens_b = int((labels_b != -100).sum().item())
    assert num_label_tokens_a == num_label_tokens_b

    la = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=[h0],
        labels=labels_a,
        model=model,
        scaling_factor=1.0,
        num_label_tokens=num_label_tokens_a,
    )
    lb = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=[h0],
        labels=labels_b,
        model=model,
        scaling_factor=1.0,
        num_label_tokens=num_label_tokens_b,
    )
    assert torch.allclose(la, lb, rtol=1e-5, atol=1e-6)
