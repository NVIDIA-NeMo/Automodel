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

import math

import torch

from nemo_automodel.components.loss.decision import (
    group_masked_bidirectional_infonce,
    rlcr_loss,
)


def test_group_masked_bidirectional_infonce_prefers_aligned_pairs():
    states = torch.eye(4)
    actions = torch.eye(4)
    scale = torch.tensor(math.log(10.0))

    aligned = group_masked_bidirectional_infonce(states, actions, scale)
    permuted = group_masked_bidirectional_infonce(states, actions.roll(1, 0), scale)
    grouped = group_masked_bidirectional_infonce(
        states,
        actions,
        scale,
        group_ids=torch.tensor([0, 0, 1, 1]),
    )

    assert aligned < permuted
    assert torch.isfinite(grouped)


def test_rlcr_loss_has_finite_gradient():
    logits = torch.tensor([[1.5, 0.0, -0.5], [0.2, 0.3, 0.1]], requires_grad=True)
    targets = torch.tensor([0, 2])
    output = rlcr_loss(
        logits,
        targets,
        group_size=32,
        exploration_std=0.4,
        policy_weight=0.25,
        supervised_weight=0.5,
        generator=torch.Generator().manual_seed(7),
    )

    output.loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert output.brier >= 0
    assert -1 <= output.mean_reward <= 1
