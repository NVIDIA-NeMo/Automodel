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

"""Independent Gaussian directional derivatives with fixed numerical bounds.

The relative allowances permit rounding variation while rejecting the
BF16 backward-error regression exercised by the corruption controls. Finite differences use both
fixed neighboring steps and a small absolute floor for cancellation directions.
Both fixed neighboring perturbation sizes must satisfy the bound.
"""

import json
import math

import pytest
import torch
from torch.nn import functional as F

from tests.unit_tests.models.ernie5_5._mtp_reference_loss import ernie5_5_mtp_loss
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import (
    OWNED_PARAMETERS,
    _mtp_definition,
    _nonidentity_model,
    batch,
    model,
)

pytestmark = pytest.mark.cpu_only


def measure_definition(dtype: str) -> dict:
    """Record every named gradient against the independent layer definition."""
    actual, reference = _nonidentity_model(dtype), model(dtype)
    reference.load_state_dict(actual.state_dict())
    tokens, labels = batch()
    observed = actual(tokens).mtp_per_depth_logits[0]
    expected, _ = _mtp_definition(reference, tokens)
    ernie5_5_mtp_loss(observed, labels).loss.backward()
    (0.1 * F.cross_entropy(expected[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten())).backward()
    result = {}
    for name, parameter in actual.named_parameters():
        left, right = parameter.grad, reference.get_parameter(name).grad
        row = {"actual_none": left is None, "reference_none": right is None}
        if left is not None and right is not None:
            x, y = left.float(), right.float()
            row.update(
                reference_norm=float(y.norm()),
                actual_norm=float(x.norm()),
                absolute_l2=float((x - y).norm()),
                relative_l2=float((x - y).norm() / y.norm().clamp_min(1e-30)),
            )
        result[name] = row
    return result


def measure_directions() -> list[dict]:
    """Record both fixed steps in directions independent of the gradient."""
    candidate = _nonidentity_model()
    body = candidate.model.layers[-1]
    with torch.no_grad():
        for hc in [body.attn_hc, body.ffn_hc]:
            hc.base[:4].copy_(torch.tensor([-0.5, 0.4, 0.2, -0.3]))
            hc.base[4:].copy_(torch.tensor([8.0, -8.0, -8.0, 8.0]))
        candidate.model.multimax.multimax_ranges.copy_(torch.tensor([1.0, -1.0, 1.0, -1.0]))
        candidate.model.multimax.multimax_ts.copy_(torch.tensor([0.3, -0.2, 0.25, 0.15]))
    generator = torch.Generator().manual_seed(8341)
    streams = torch.randn(1, 8, 2, 16, generator=generator)
    tokens, labels = batch()
    targets = torch.tensor([[3, 4, 5, 6, 7, 8, -100, -100]])

    def objective() -> torch.Tensor:
        logits = candidate._forward_mtp(streams, candidate.model.attention_inputs(tokens), input_ids=tokens)
        # FP64 reduction resolves a slope without FP32 CE reduction noise.
        loss = 0.1 * F.cross_entropy(logits.double().flatten(0, 1), targets.flatten())
        return loss

    candidate.zero_grad(set_to_none=True)
    initial_loss = float(objective().detach())
    objective().backward()
    names = sorted(OWNED_PARAMETERS) + ["model.layers.1.moe.fc2_latent_proj.weight"]
    result = []
    for index, name in enumerate(names):
        parameter = candidate.get_parameter(name)
        assert parameter.grad is not None, name
        base, gradient = parameter.detach().clone(), parameter.grad.detach().clone()
        for direction_index in [0] if name == "mtp.e_proj.weight" else range(2):
            generator.manual_seed(9301 + 13 * index + direction_index)
            direction = torch.randn(parameter.shape, generator=generator)
            direction /= direction.norm()
            analytic = float((gradient.double() * direction.double()).sum())
            estimates = []
            try:
                for step in [0.03, 0.01]:
                    with torch.no_grad():
                        parameter.copy_(base + step * direction)
                        plus = float(objective())
                        parameter.copy_(base - step * direction)
                        minus = float(objective())
                    numeric = (plus - minus) / (2 * step)
                    estimates.append(
                        {
                            "step": step,
                            "numeric": numeric,
                            "analytic": analytic,
                            "absolute_error": abs(numeric - analytic),
                            "relative_error": abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-30),
                        }
                    )
            finally:
                with torch.no_grad():
                    parameter.copy_(base)
            result.append(
                {
                    "name": name,
                    "direction": direction_index,
                    "seed": 9301 + 13 * index + direction_index,
                    "gradient_norm": float(gradient.norm()),
                    "estimates": estimates,
                }
            )
    assert float(objective().detach()) == initial_loss
    return result


@pytest.mark.parametrize("dtype,relative_bound", [("float32", 1e-4), ("bfloat16", 0.002)])
def test_all_named_definition_gradients(dtype: str, relative_bound: float) -> None:
    measurements = measure_definition(dtype)
    failures = []
    nonzero = set()
    for name, row in measurements.items():
        if row["actual_none"] != row["reference_none"]:
            failures.append((name, "missing contribution", row))
        elif not row["reference_none"]:
            if not all(
                math.isfinite(row[key])
                for key in [
                    "reference_norm",
                    "actual_norm",
                    "absolute_l2",
                    "relative_l2",
                ]
            ):
                failures.append((name, "nonfinite gradient metric", row))
                continue
            if row["reference_norm"]:
                nonzero.add(name)
                if row["relative_l2"] > relative_bound:
                    failures.append((name, "relative gradient error", row))
            elif row["absolute_l2"] != 0:
                failures.append((name, "expected exact zero", row))
    assert "model.layers.0.mlp.down_proj.weight" in nonzero
    assert "model.layers.1.moe.fc2_latent_proj.weight" in nonzero
    assert len([name for name in nonzero if name.startswith("mtp.")]) == 8
    assert not failures, json.dumps(failures, indent=2)


def test_all_owned_and_shared_seeded_directional_derivatives() -> None:
    measurements = measure_directions()
    assert len(measurements) == 17
    failures = []
    for row in measurements:
        assert math.isfinite(row["gradient_norm"]) and row["gradient_norm"] > 0, row["name"]
        assert [item["step"] for item in row["estimates"]] == [0.03, 0.01]
        # Keep one fixed independent e_proj direction; require every retained
        # slope to resolve rather than allowing an uninformative passing slot.
        absolute_floor = 8e-7 if row["name"] == "mtp.e_proj.weight" else 3e-7
        for estimate in row["estimates"]:
            bound = absolute_floor + 0.01 * abs(estimate["analytic"])
            assert abs(estimate["analytic"]) > 2 * bound, (
                row,
                "unresolved direction",
                bound,
            )
            if (
                not all(math.isfinite(estimate[key]) for key in ["numeric", "analytic", "absolute_error"])
                or estimate["absolute_error"] > bound
            ):
                failures.append({**row, "failed_step": estimate, "bound": bound})
    assert not failures, json.dumps(failures, indent=2)
