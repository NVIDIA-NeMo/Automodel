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

"""Muown layout, state, grouping and per-expert update regressions."""

import copy

import pytest
import torch
from torch import nn

pytest.importorskip("dion")

from nemo_automodel.components.optim.muown import Muown, _nonzero_magnitude
from nemo_automodel.components.optim.optimizer import MuownConfig, build_optimizer_config


@pytest.fixture(autouse=True)
def eager_optimizer():
    # Compilation has a separate GPU smoke test; keep CPU unit tests inexpensive.
    with torch._dynamo.config.patch(disable=True):
        yield


@pytest.mark.parametrize("shape", [(3, 8, 12), (2, 24, 8)])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("decay", [0.0, 0.1])
def test_batched_experts_match_independent_neurons(shape, transposed, nesterov, decay):
    torch.manual_seed(17)
    initial = torch.randn(shape)
    weight = nn.Parameter(initial.mT.contiguous() if transposed else initial.clone())
    experts = [nn.Parameter(value.clone()) for value in initial]
    optimizer = Muown([{"params": [weight], "matrix_transposed": transposed}], nesterov=nesterov, weight_decay=decay)
    reference = Muown(experts, nesterov=nesterov, weight_decay=decay)
    for _ in range(4):
        grad = torch.randn_like(initial)
        weight.grad = grad.mT.contiguous() if transposed else grad.clone()
        original_grad = weight.grad
        for expert, expert_grad in zip(experts, grad):
            expert.grad = expert_grad.clone()
        optimizer.step()
        reference.step()
        assert weight.grad is original_grad
        torch.testing.assert_close(original_grad, grad.mT if transposed else grad, rtol=0, atol=0)
        actual = weight.mT if transposed else weight
        torch.testing.assert_close(actual, torch.stack(experts), rtol=1e-5, atol=1e-5)
    state = optimizer.state[weight]
    axis = -2 if transposed else -1
    expected_shape = list(weight.shape)
    expected_shape[axis] = 1
    assert state["g"].shape == tuple(expected_shape)
    assert state["muown_step"] == 4


def test_resume_missing_grad_and_zero_rows():
    torch.manual_seed(9)
    weight = nn.Parameter(torch.randn(8, 12))
    unused = nn.Parameter(torch.zeros(8, 12))
    optimizer = Muown([weight, unused], weight_decay=0.1)
    for _ in range(3):
        weight.grad = torch.randn_like(weight)
        optimizer.step()
    assert optimizer.state[unused]["muown_step"] == 0
    torch.testing.assert_close(unused, torch.zeros_like(unused), rtol=0, atol=0)
    copies = [nn.Parameter(p.detach().clone()) for p in [weight, unused]]
    resumed = Muown(copies, weight_decay=0.1)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for original, restored in zip([weight, unused], copies):
        original.grad = torch.randn_like(original)
        restored.grad = original.grad.clone()
    optimizer.step()
    resumed.step()
    for original, restored in zip([weight, unused], copies):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
        assert torch.isfinite(original).all()


def test_negative_magnitude_is_not_clamped_positive():
    weight = nn.Parameter(torch.randn(8, 12))
    optimizer = Muown([weight], lr=0)
    optimizer.state[weight]["g"].neg_()
    before = weight.detach().clone()
    weight.grad = torch.randn_like(weight)
    optimizer.step()
    torch.testing.assert_close(weight, before, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"mu": 1.0}, {"betas": (0.9, 1.0)}, {"epsilon": 0}, {"ns_steps": 0}])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        Muown([nn.Parameter(torch.randn(8, 12))], **kwargs)


def test_typed_config_layout_and_scalar_groups():
    class Experts(nn.Module):
        _nemo_transposed_matrix_parameters = ("projection",)

        def __init__(self):
            super().__init__()
            self.projection = nn.Parameter(torch.randn(4, 8, 12))
            self.proj_bias = nn.Parameter(torch.randn(4, 12))

    model = nn.ModuleDict(
        {
            "experts": Experts(),
            "linear": nn.Linear(8, 12),
            "embed": nn.Embedding(12, 8),
            "lm_head": nn.Linear(8, 12, bias=False),
        }
    )
    for target in ("muown", MuownConfig, Muown):
        config = build_optimizer_config(target, {"lr": 1e-3, "scalar_lr": 1e-4})
        assert isinstance(config, MuownConfig)
        optimizers = config.build(model)
        assert len(optimizers) == 1
        groups = {id(p): group for group in optimizers[0].param_groups for p in group["params"]}
        assert groups[id(model["experts"].projection)]["matrix_transposed"]
        assert groups[id(model["experts"].projection)]["algorithm"] == "muon"
        assert not groups[id(model["linear"].weight)]["matrix_transposed"]
        for parameter in [
            model["experts"].proj_bias,
            model["linear"].bias,
            model["embed"].weight,
            model["lm_head"].weight,
        ]:
            assert groups[id(parameter)]["algorithm"] == "adamw"
        assert groups[id(model["experts"].proj_bias)]["lr"] == 1e-4


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_magnitude_guard_preserves_sign_and_dtype(dtype):
    magnitude = torch.tensor([[-2.0], [0.0], [2.0]], dtype=dtype)
    guarded = _nonzero_magnitude(magnitude, 0.25)
    assert guarded.dtype == dtype
    torch.testing.assert_close(guarded, torch.tensor([[-2.0], [0.25], [2.0]], dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(magnitude, torch.tensor([[-2.0], [0.0], [2.0]], dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("update_scale", [0.0, 0.5, 1.0, 2.0])
def test_direction_update_does_not_round_scaled_update_to_bfloat16(monkeypatch, update_scale):
    # The output/input ratio cancels the learned magnitude and row norm. It
    # must retain the FP32 learning-rate product even when NS returns BF16.
    import nemo_automodel.components.optim.muown as implementation

    monkeypatch.setattr(
        implementation,
        "_newton_schulz",
        lambda *args, **kwargs: torch.tensor([[0.0, 0.75]], dtype=torch.bfloat16),
    )
    weight = nn.Parameter(torch.tensor([[1.0, 0.0]]))
    optimizer = Muown([weight], lr=0.0033, mu=0.0, nesterov=False, muon_update_scale=update_scale)
    weight.grad = torch.tensor([[0.25, 1.0]])
    optimizer.step()
    expected_ratio = -0.0033 * update_scale * (0.2 * 2**0.5) * 0.75
    assert abs((weight[0, 1] / weight[0, 0]).item() - expected_ratio) < 1e-9
    # At the first step Adam reduces a positive scalar magnitude by lr,
    # independently of the direction multiplier (including zero).
    torch.testing.assert_close(optimizer.state[weight]["g"], torch.tensor([[1.0 - 0.0033]]), rtol=0, atol=1e-7)


def test_adamw_fallback_matches_torch_with_missing_grad_and_resume():
    torch.manual_seed(101)
    params = [nn.Parameter(torch.randn(8, 12)), nn.Parameter(torch.randn(12))]
    reference_params = [nn.Parameter(param.detach().clone()) for param in params]
    optimizer = Muown([{"params": params, "algorithm": "adamw"}], lr=0.002, weight_decay=0.1)
    reference = torch.optim.AdamW(reference_params, lr=0.002, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    for step in range(6):
        for index, (param, expected) in enumerate(zip(params, reference_params)):
            grad = None if index == 1 and step < 2 else torch.randn_like(param)
            param.grad = grad
            expected.grad = grad.clone() if grad is not None else None
        optimizer.step()
        reference.step()
        for param, expected in zip(params, reference_params):
            torch.testing.assert_close(param, expected, atol=0, rtol=0)
            if expected in reference.state and reference.state[expected]:
                actual_state = optimizer.state[param]
                expected_state = reference.state[expected]
                for actual_key, expected_key in [
                    ("momentum", "exp_avg"),
                    ("variance", "exp_avg_sq"),
                    ("step", "step"),
                ]:
                    torch.testing.assert_close(actual_state[actual_key], expected_state[expected_key], atol=0, rtol=0)
        if step == 2:
            saved = copy.deepcopy(optimizer.state_dict())
            optimizer = Muown([{"params": params, "algorithm": "adamw"}], lr=0.002, weight_decay=0.1)
            optimizer.load_state_dict(saved)


def test_zero_gradient_zero_row_stays_finite():
    weight = nn.Parameter(torch.zeros(8, 12))
    optimizer = Muown([weight])
    weight.grad = torch.zeros_like(weight)
    optimizer.step()
    torch.testing.assert_close(weight, torch.zeros_like(weight), atol=0, rtol=0)
    for value in optimizer.state[weight].values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()


@pytest.mark.parametrize("scale", [-0.1, float("nan"), float("inf")])
def test_invalid_direction_scale(scale):
    with pytest.raises(ValueError, match="muon_update_scale"):
        Muown([nn.Parameter(torch.ones(2, 2))], muon_update_scale=scale)
