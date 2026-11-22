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


def test_group_mask_is_equivalent_to_removing_only_false_negative_candidates():
    states = torch.tensor([[1, 2, 0], [3, -1, 1], [0, 1, 2], [1, 0, -1]], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([[2, 1, 0], [1, -1, 2], [0, 3, 1], [2, 0, -1]], dtype=torch.float64, requires_grad=True)
    for groups in ([0, 0, 1, 2], [0, 0, 0, 0]):
        terms = []
        for source, candidates in ((states, actions), (actions, states)):
            for row in range(4):
                allowed = [index for index in range(4) if index == row or groups[index] != groups[row]]
                scores = 2 * torch.nn.functional.cosine_similarity(source[row], candidates[allowed], dim=-1)
                terms.append(torch.nn.functional.cross_entropy(scores[None], torch.tensor([allowed.index(row)])))
        expected = torch.stack(terms).mean()
        actual = group_masked_bidirectional_infonce(
            states, actions, torch.tensor(math.log(2.0), dtype=torch.float64), torch.tensor(groups)
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_grad, expected_grad in zip(
            torch.autograd.grad(actual, (states, actions)),
            torch.autograd.grad(expected, (states, actions)),
            strict=True,
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_rlcr_matches_scalar_reward_and_analytic_score_function_gradient():
    values = [[0.2, -0.1], [-0.3, 0.7]]
    targets = [0, 0]
    std, policy_weight, supervised_weight, group = 0.5, 0.25, 0.75, 3
    noise = torch.randn(2, group, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(7)).tolist()
    expected_policy, expected_supervised, expected_reward, expected_brier = 0.0, 0.0, 0.0, 0.0
    expected_gradient = []
    for base, target, draws in zip(values, targets, noise, strict=True):
        proposals = [[value + std * delta for value, delta in zip(base, draw, strict=True)] for draw in draws]
        rewards = []
        for proposal in proposals:
            masses = [math.exp(value) for value in proposal]
            probabilities = [mass / sum(masses) for mass in masses]
            choice = max(range(2), key=probabilities.__getitem__)
            correct = float(choice == target)
            rewards.append(correct - (probabilities[choice] - correct) ** 2)
        baseline = sum(rewards) / group
        advantages = [reward - baseline for reward in rewards]
        location_scores = [
            -sum((sample - value) ** 2 for sample, value in zip(p, base)) / (2 * std**2) for p in proposals
        ]
        expected_policy -= sum(a * score for a, score in zip(advantages, location_scores)) / (2 * group)
        masses = [math.exp(value) for value in base]
        probabilities = [mass / sum(masses) for mass in masses]
        expected_supervised -= math.log(probabilities[target]) / 2
        expected_reward += baseline / 2
        expected_brier += sum((p - float(index == target)) ** 2 for index, p in enumerate(probabilities)) / 2
        expected_gradient.append(
            [
                -policy_weight
                * sum(a * (p[index] - base[index]) for a, p in zip(advantages, proposals))
                / (2 * group * std**2)
                + supervised_weight * (probabilities[index] - float(index == target)) / 2
                for index in range(2)
            ]
        )
    logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    output = rlcr_loss(
        logits,
        torch.tensor(targets),
        group_size=group,
        exploration_std=std,
        policy_weight=policy_weight,
        supervised_weight=supervised_weight,
        generator=torch.Generator().manual_seed(7),
    )
    expected = [
        policy_weight * expected_policy + supervised_weight * expected_supervised,
        expected_policy,
        expected_supervised,
        expected_reward,
        expected_brier,
    ]
    for actual, value in zip(
        (output.loss, output.policy_loss, output.supervised_loss, output.mean_reward, output.brier),
        expected,
        strict=True,
    ):
        torch.testing.assert_close(actual, torch.tensor(value, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    output.loss.backward()
    torch.testing.assert_close(
        logits.grad, torch.tensor(expected_gradient, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
