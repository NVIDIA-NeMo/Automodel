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

"""Numerical contracts for Section 2.5, independent of the dense training layout."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.speculative.gated_mtp import GatedMTPConfig, GatedMTPOutput

pytestmark = pytest.mark.cpu_only


def _reference_loss(
    output: GatedMTPOutput,
    sampler_logits: list[torch.Tensor],
    labels: torch.Tensor,
    head: torch.nn.Linear,
) -> torch.Tensor:
    """Enumerate supervised predictions and group consistency terms by clean target.

    Args:
        output: Clean logits, Tensor of shape [batch, sequence, vocab]; clean hidden
            states and per-depth mask states, each a Tensor of shape [batch, sequence, hidden].
        sampler_logits: Clean and per-depth logits, each a Tensor of shape [batch, sequence, vocab].
        labels: Tensor of shape [batch, sequence], with -100 at ignored positions.
        head: Frozen projection from hidden to vocabulary.

    Returns:
        Per-sample objective, Tensor of shape [batch].
    """
    losses = []
    for row in range(labels.shape[0]):
        ce_terms = []
        matches = {}
        for anchor in range(labels.shape[1]):
            if labels[row, anchor] == -100:
                continue
            ce_terms.append(
                F.cross_entropy(output.logits[row, anchor].unsqueeze(0), labels[row, anchor].unsqueeze(0))
                + F.cross_entropy(sampler_logits[0][row, anchor].unsqueeze(0), labels[row, anchor].unsqueeze(0))
            )
            for depth, hidden in enumerate(output.mtp_per_depth_h, start=1):
                target = anchor + depth
                if target >= labels.shape[1] or labels[row, target] == -100:
                    continue
                ce_terms.append(
                    F.cross_entropy(head(hidden[row, anchor]).unsqueeze(0), labels[row, target].unsqueeze(0))
                    + F.cross_entropy(sampler_logits[depth][row, anchor].unsqueeze(0), labels[row, target].unsqueeze(0))
                )
                matches.setdefault(target, []).append(
                    F.mse_loss(hidden[row, anchor], output.hidden_states[row, target].detach())
                )
        ce = torch.stack(ce_terms).mean()
        consistency = sum(torch.stack(terms).mean() for terms in matches.values())
        consistency = consistency / (labels[row] != -100).sum()
        losses.append(ce + consistency)
    return torch.stack(losses)


@pytest.mark.parametrize("depths", [1, 3, 6])
@pytest.mark.parametrize("ignore_index", [-100, -1])
def test_paper_loss_values_and_gradients_match_enumerated_reference(depths: int, ignore_index: int) -> None:
    """Unequal supervision, missing anchors, and short tails must preserve both means."""
    torch.manual_seed(7)
    batch, sequence, hidden, vocab = 2, 5, 4, 7
    layout = GatedMTPConfig(mask_token_ids=list(range(depths)), objective="paper", scaling_factor=0.7).build()
    head = torch.nn.Linear(hidden, vocab, bias=False).requires_grad_(False)
    clean = torch.randn(batch, sequence, hidden, requires_grad=True)
    logits = torch.randn(batch, sequence, vocab, requires_grad=True)
    masks = [torch.randn(batch, sequence, hidden, requires_grad=True) for _ in range(depths)]
    sampler = [torch.randn(batch, sequence, vocab, requires_grad=True) for _ in range(depths + 1)]
    output = GatedMTPOutput(logits, clean, masks, None, 0.7)
    labels = torch.tensor([[-100, 2, 3, 1, -100], [1, 4, -100, 2, 3]])

    actual = layout.paper_loss(output, sampler_logits=sampler, lm_head=head, labels=labels, ignore_index=ignore_index)
    expected = 0.7 * _reference_loss(output, sampler, labels, head)
    torch.testing.assert_close(actual, expected)
    leaves = [clean, logits, *masks, *sampler]
    actual_grads = torch.autograd.grad(actual.sum(), leaves, retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad(expected.sum(), leaves, allow_unused=True)
    assert actual_grads[0] is None  # LCM must not train the clean target.
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None or torch.count_nonzero(actual_grad) == 0
        else:
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
    # A mask after an unsupervised anchor must not receive CE or LCM gradients.
    torch.testing.assert_close(actual_grads[2][0, 0], torch.zeros(hidden))


@pytest.mark.parametrize("sequence", [1, 5])
def test_paper_loss_empty_supervision_has_zero_finite_gradients(sequence: int) -> None:
    """Every sampler depth stays connected even when all its labels are ignored."""
    torch.manual_seed(11)
    layout = GatedMTPConfig(mask_token_ids=[5, 6], objective="paper").build()
    head = torch.nn.Linear(4, 7, bias=False).requires_grad_(False)
    logits = torch.randn(2, sequence, 7, requires_grad=True)
    masks = [torch.randn(2, sequence, 4, requires_grad=True) for _ in range(2)]
    sampler = [torch.randn(2, sequence, 7, requires_grad=True) for _ in range(3)]
    output = GatedMTPOutput(logits, torch.randn(2, sequence, 4), masks, None, 1.0)
    loss = layout.paper_loss(
        output,
        sampler_logits=sampler,
        lm_head=head,
        labels=torch.full((2, sequence), -100),
        ignore_index=-100,
    )
    torch.testing.assert_close(loss, torch.zeros(2))
    loss.sum().backward()
    for value in [logits, *sampler]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) == 0
