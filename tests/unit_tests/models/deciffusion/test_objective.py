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

"""Decision objective: hard/soft cross-entropy and gold CE + KL, values, gradients and failure modes."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deciffusion.objective import decision_loss


def test_hard_label_loss_is_cross_entropy_with_softmax_gradient():
    logits = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    loss = decision_loss(logits, 2)
    torch.testing.assert_close(loss, F.cross_entropy(logits[None], torch.tensor([2])))
    loss.backward()
    torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - F.one_hot(torch.tensor(2), 3).float())


def test_soft_label_loss_is_soft_cross_entropy_and_ignores_gold():
    logits = torch.tensor([0.1, 0.2, -0.3], requires_grad=True)
    soft = torch.tensor([0.7, 0.3, 0.0])
    loss = decision_loss(logits, 1, soft)
    torch.testing.assert_close(loss, -(soft * logits.detach().log_softmax(-1)).sum())
    loss.backward()
    torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - soft)


def test_gold_ce_plus_kl():
    logits = torch.tensor([1.0, 0.0, -1.0])
    soft = torch.tensor([0.5, 0.5, 0.0])
    log_probs = logits.log_softmax(-1)
    kl = F.kl_div(log_probs, soft, reduction="sum")
    torch.testing.assert_close(decision_loss(logits, 0, soft, kl_weight=0.5), -log_probs[0] + 0.5 * kl)
    torch.testing.assert_close(decision_loss(logits, 0, None, kl_weight=0.5), -log_probs[0])


def test_bf16_logits_are_scored_in_fp32():
    logits = torch.tensor([30.0, -30.0, 0.0], dtype=torch.bfloat16)
    loss = decision_loss(logits, 1)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"gold": 3}, "index"),
        ({"gold": 0, "soft": torch.ones(2) / 2}, "align"),
        ({"gold": 0, "kl_weight": -1.0}, "non-negative"),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, error):
    with pytest.raises(ValueError, match=error):
        decision_loss(torch.zeros(3), **kwargs)
