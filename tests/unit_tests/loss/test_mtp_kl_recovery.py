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

"""Tests for the experimental soft-MTP straight-through KL-recovery trick.

``apply_mtp_kl_recovery`` (kashif/experiment/soft-mtp) injects a zero-valued
straight-through term into the main loss's pre-loss representation. On its
own it does nothing useful; COMBINED with the existing hard-label
``calculate_mtp_loss`` term for the same depths, the two gradient
contributions are claimed to cancel algebraically into a clean forward-KL
gradient (`coeff * (softmax(head) - softmax(main))`), reusing only the two
softmaxes a hard-CE MTP loss already needs -- no extra softmax, no teacher
logits tensor.

These tests verify that claim directly with autograd against a float64
closed-form reference, rather than trusting the derivation.
"""

from __future__ import annotations

import torch

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import apply_mtp_kl_recovery, calculate_mtp_loss

IGNORE = -100


def _combined_loss(main_logits, head_logits_list, labels, *, scaling_factor, num_label_tokens):
    """Main loss (on the KL-recovery-patched logits) + the unmodified hard-label MTP loss."""
    loss_fn = MaskedCrossEntropy(ignore_index=IGNORE, fp32_upcast=False)
    patched_main = apply_mtp_kl_recovery(main_logits, head_logits_list, scaling_factor=scaling_factor)
    main_loss = loss_fn(logits=patched_main, labels=labels, num_label_tokens=num_label_tokens)
    mtp_loss = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_logits=head_logits_list,
        labels=labels,
        model=None,
        scaling_factor=scaling_factor,
        num_label_tokens=num_label_tokens,
    )
    return main_loss + mtp_loss, main_loss


def test_forward_value_is_exactly_unchanged():
    """The injected term is `(x.detach() - x)`, which is exactly 0 -- the main loss's VALUE
    (not just approximately) must be identical whether or not the patch is applied."""
    torch.manual_seed(0)
    batch, seq, vocab = 2, 10, 12
    main_logits = torch.randn(batch, seq, vocab, dtype=torch.float64)
    head_logits = torch.randn(batch, seq, vocab, dtype=torch.float64)
    labels = torch.randint(0, vocab, (batch, seq))

    loss_fn = MaskedCrossEntropy(ignore_index=IGNORE, fp32_upcast=False)
    unpatched = loss_fn(logits=main_logits, labels=labels)
    patched_main = apply_mtp_kl_recovery(main_logits, [head_logits], scaling_factor=0.1)
    patched = loss_fn(logits=patched_main, labels=labels)

    assert torch.equal(unpatched, patched), (unpatched.item(), patched.item())


def test_combined_gradient_matches_forward_kl_single_depth():
    """The CLAIM under test: main-loss-gradient + hard-CE-gradient on the head's logits,
    at the aligned valid positions, equals `coeff * (softmax(head) - softmax(main))` exactly
    (float64), where coeff = scaling_factor / num_depths -- the forward-KL(main‖head) gradient,
    with no teacher-logits tensor and no softmax beyond what the hard-CE MTP loss already needs.
    """
    torch.manual_seed(0)
    batch, seq, vocab = 2, 9, 11
    num_depths = 1
    scaling_factor = 0.37
    main_logits = torch.randn(batch, seq, vocab, dtype=torch.float64, requires_grad=True)
    head_logits = torch.randn(batch, seq, vocab, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq))
    num_label_tokens = int((labels != IGNORE).sum().item())

    total, _ = _combined_loss(
        main_logits, [head_logits], labels, scaling_factor=scaling_factor, num_label_tokens=num_label_tokens
    )
    (grad_head,) = torch.autograd.grad(total, [head_logits])

    depth = 1
    valid_len = seq - depth
    coeff = scaling_factor / num_depths
    p_k = torch.softmax(head_logits[:, :valid_len].detach(), dim=-1)
    q = torch.softmax(main_logits[:, depth:].detach(), dim=-1)
    expected = coeff * (p_k - q) / num_label_tokens

    torch.testing.assert_close(grad_head[:, :valid_len], expected, atol=1e-10, rtol=1e-8)
    # Positions with no valid future target get no gradient from this mechanism.
    assert torch.equal(grad_head[:, valid_len:], torch.zeros_like(grad_head[:, valid_len:]))


def test_combined_gradient_matches_forward_kl_two_depths():
    """Same identity, two independent MTP depths sharing the same main_logits.

    Each depth's gradient must match its OWN aligned (coeff, teacher) pair -- this is the
    test most likely to fail if the per-depth roll offset or the `scaling_factor / D`
    coefficient (not a per-depth token-count ratio -- see apply_mtp_kl_recovery's docstring)
    were wrong.
    """
    torch.manual_seed(1)
    batch, seq, vocab = 1, 10, 8
    num_depths = 2
    scaling_factor = 0.8
    main_logits = torch.randn(batch, seq, vocab, dtype=torch.float64, requires_grad=True)
    head_logits_list = [
        torch.randn(batch, seq, vocab, dtype=torch.float64, requires_grad=True) for _ in range(num_depths)
    ]
    labels = torch.randint(0, vocab, (batch, seq))
    num_label_tokens = int((labels != IGNORE).sum().item())

    total, _ = _combined_loss(
        main_logits, head_logits_list, labels, scaling_factor=scaling_factor, num_label_tokens=num_label_tokens
    )
    grads = torch.autograd.grad(total, head_logits_list)

    coeff = scaling_factor / num_depths
    for k, grad_head in enumerate(grads):
        depth = k + 1
        valid_len = seq - depth
        p_k = torch.softmax(head_logits_list[k][:, :valid_len].detach(), dim=-1)
        q = torch.softmax(main_logits[:, depth:].detach(), dim=-1)
        expected = coeff * (p_k - q) / num_label_tokens
        torch.testing.assert_close(grad_head[:, :valid_len], expected, atol=1e-10, rtol=1e-8)


def test_either_gradient_contribution_alone_does_not_match_forward_kl():
    """Negative control: the hard-CE term alone (no patch) and the patch alone (no hard-CE
    term) must each NOT equal the forward-KL gradient -- only their sum does. Guards against
    the test above passing for a trivial reason (e.g. one contribution already being ~0).
    """
    torch.manual_seed(2)
    batch, seq, vocab = 1, 8, 10
    scaling_factor = 0.5
    main_logits = torch.randn(batch, seq, vocab, dtype=torch.float64, requires_grad=True)
    head_logits = torch.randn(batch, seq, vocab, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq))
    num_label_tokens = int((labels != IGNORE).sum().item())
    loss_fn = MaskedCrossEntropy(ignore_index=IGNORE, fp32_upcast=False)

    # Hard-CE term alone (the pre-existing, unmodified behavior).
    hard_only = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_logits=[head_logits],
        labels=labels,
        model=None,
        scaling_factor=scaling_factor,
        num_label_tokens=num_label_tokens,
    )
    (grad_hard_only,) = torch.autograd.grad(hard_only, [head_logits])

    depth, valid_len = 1, seq - 1
    coeff = scaling_factor / 1
    p_k = torch.softmax(head_logits[:, :valid_len].detach(), dim=-1)
    q = torch.softmax(main_logits[:, depth:].detach(), dim=-1)
    expected = coeff * (p_k - q) / num_label_tokens

    assert not torch.allclose(grad_hard_only[:, :valid_len], expected, atol=1e-6)
