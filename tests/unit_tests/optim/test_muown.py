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
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("dion")

from nemo_automodel.components.optim.muown import Muown, _nonzero_magnitude
from nemo_automodel.components.optim.optimizer import (
    AdamWConfig,
    LRSchedulerConfig,
    MuownConfig,
    ParamGroupOverride,
    build_optimizer_config,
)


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


def test_unsupported_dion_revision_is_rejected(monkeypatch):
    # A Dion release without the private Muon hooks Muown overrides must fail loudly instead of running plain Muon.
    from nemo_automodel.components.optim import muown

    monkeypatch.setattr(muown, "_DION_MUON_HOOKS", (*muown._DION_MUON_HOOKS, "_hook_removed_upstream"))
    with pytest.raises(ImportError, match="_hook_removed_upstream"):
        Muown([nn.Parameter(torch.randn(4, 4))])


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


@pytest.mark.parametrize("update_scale", [0.0, 0.5, 1.0])
def test_router_override_matches_adamw_and_preserves_matrix_groups(update_scale):
    torch.manual_seed(113)
    model = nn.ModuleDict(
        {
            "attn": nn.Linear(8, 12, bias=False),
            "router": nn.Linear(8, 4, bias=False),
            "embed": nn.Embedding(16, 8),
            "lm_head": nn.Linear(8, 16, bias=False),
        }
    )
    config = MuownConfig(
        lr=0.003,
        scalar_lr=0.001,
        embed_lr=0.0002,
        lm_head_lr=0.0003,
        weight_decay=0.1,
        scalar_betas=(0.8, 0.9),
        scalar_eps=1e-6,
        muon_update_scale=update_scale,
        adamw_param_patterns=[r"^router\.weight$"],
        param_group_overrides=[
            {"pattern": r"^router\.weight$", "lr_mult": 0.5, "wd_mult": 0.2},
            # First LR/WD match wins; algorithm selection is independent.
            {"pattern": "weight", "lr_mult": 1.0},
        ],
    )
    opt = config.build(model)[0]
    groups = {id(p): group for group in opt.param_groups for p in group["params"]}
    assert len(groups) == len(list(model.parameters()))
    assert groups[id(model["attn"].weight)]["algorithm"] == "muon"
    router_group = groups[id(model["router"].weight)]
    assert router_group["algorithm"] == "adamw"
    assert "g" not in opt.state[model["router"].weight]
    expected_router = nn.Parameter(model["router"].weight.detach().clone())
    reference = torch.optim.AdamW(
        [expected_router], lr=0.0005, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.02, fused=True
    )
    # Also exercise the native scheduler: scalar LR must not become matrix LR.
    schedule = LRSchedulerConfig(lr_warmup_steps=2, init_lr=0.0, min_lr=0.0003).build(
        opt, SimpleNamespace(epoch_len=10, num_epochs=1, max_steps=10)
    )[0]
    for step in range(5):
        if step:
            schedule.step(1)
        base_lr = schedule.get_lr({})
        assert router_group["lr"] == pytest.approx(base_lr / 6)
        assert router_group["weight_decay"] == pytest.approx(0.02)
        assert groups[id(model["embed"].weight)]["lr"] == pytest.approx(base_lr / 15)
        assert groups[id(model["lm_head"].weight)]["lr"] == pytest.approx(base_lr / 10)
        assert groups[id(model["embed"].weight)]["weight_decay"] == 0.0
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        expected_router.grad = model["router"].weight.grad.clone()
        reference.param_groups[0]["lr"] = base_lr / 6
        opt.step()
        reference.step()
        torch.testing.assert_close(model["router"].weight, expected_router, rtol=0, atol=0)


def test_adamw_patterns_rejected_by_standard_optimizer():
    with pytest.raises(TypeError, match="adamw_param_patterns"):
        AdamWConfig(adamw_param_patterns=["weight"])


def test_algorithm_override_is_not_a_generic_group_option():
    with pytest.raises(TypeError, match="algorithm"):
        ParamGroupOverride(pattern="router", algorithm="adamw")


def test_override_preserves_transposed_experts_and_ignores_frozen_parameters(caplog):
    class Experts(nn.Module):
        _nemo_transposed_matrix_parameters = ("projection",)

        def __init__(self):
            super().__init__()
            self.projection = nn.Parameter(torch.randn(2, 8, 12))

    model = nn.ModuleDict({"experts": Experts(), "router": nn.Linear(8, 4, bias=False)})
    model["router"].requires_grad_(False)
    config = MuownConfig(
        adamw_param_patterns=["router"],
        param_group_overrides=[{"pattern": "experts", "lr_mult": 0.5}],
    )
    optimizer = config.build(model)[0]
    groups = [g for g in optimizer.param_groups if g["params"]]
    assert len(groups) == 1
    assert groups[0]["matrix_transposed"]
    assert groups[0]["algorithm"] == "muon"
    assert groups[0]["params"][0] is model["experts"].projection
    assert groups[0]["lr"] == pytest.approx(config.lr * 0.5)
    assert "matched no parameters" in caplog.text


def test_all_matrix_parameters_can_use_adamw_without_changing_scheduler_base():
    model = nn.Linear(8, 4, bias=False)
    config = MuownConfig(
        lr=0.003,
        scalar_lr=0.001,
        weight_decay=0.1,
        adamw_param_patterns=["weight"],
        param_group_overrides=[{"pattern": "weight", "lr_mult": 0.5, "wd_mult": 0.0}],
    )
    optimizer = config.build(model)[0]
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.0005
    schedule = LRSchedulerConfig(lr_warmup_steps=0, lr_decay_style="constant").build(
        optimizer, SimpleNamespace(epoch_len=10, num_epochs=1, max_steps=10)
    )[0]
    assert schedule.max_lr == 0.003
    assert optimizer.param_groups[0]["lr"] == 0.0005
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_optional_router_rule_excludes_expert_gate_proj():
    config = MuownConfig(
        adamw_param_patterns=[r"(^|\.)(gate|router)\.weight$"],
        scalar_lr=1e-4,
    )
    model = nn.ModuleDict(
        {
            "mlp": nn.ModuleDict(
                {
                    "gate": nn.Linear(8, 4, bias=False),
                    "gate_proj": nn.Linear(8, 16, bias=False),
                }
            ),
        }
    )
    optimizer = config.build(model)[0]
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    assert optimizer.muon_update_scale == 1.0
    assert groups[id(model["mlp"]["gate"].weight)]["algorithm"] == "adamw"
    assert groups[id(model["mlp"]["gate"].weight)]["lr"] == config.scalar_lr
    assert groups[id(model["mlp"]["gate_proj"].weight)]["algorithm"] == "muon"


@pytest.mark.parametrize("scalar_opt", ["adamw", "lion"])
@pytest.mark.parametrize("with_overrides", [False, True])
def test_adamw_patterns_work_independently_of_lr_overrides(scalar_opt, with_overrides):
    torch.manual_seed(117)
    model = nn.ModuleDict(
        {
            "attn": nn.Linear(8, 12),
            "router": nn.Linear(8, 4, bias=False),
            "embed": nn.Embedding(16, 8),
            "lm_head": nn.Linear(8, 16, bias=False),
        }
    )
    multiplier = 0.5 if with_overrides else 1.0
    config = MuownConfig(
        lr=3e-4,
        weight_decay=0.1,
        scalar_opt=scalar_opt,
        scalar_lr=1e-4,
        scalar_betas=(0.8, 0.9),
        scalar_eps=1e-6,
        embed_lr=2e-5,
        lm_head_lr=3e-5,
        # Overlapping patterns select a parameter once. Existing auxiliary
        # groups keep their LR/WD even when explicitly selected for AdamW.
        adamw_param_patterns=["router", r"router\.weight$", "embed", "lm_head"],
        param_group_overrides=[{"pattern": "weight", "lr_mult": 0.5}] if with_overrides else [],
    )
    opt = config.build(model)[0]
    parameters = [p for group in opt.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters}) == len(list(model.parameters()))
    groups = {id(p): group for group in opt.param_groups for p in group["params"]}
    assert groups[id(model["attn"].weight)]["algorithm"] == "muon"
    assert groups[id(model["attn"].bias)]["algorithm"] == scalar_opt
    assert groups[id(model["router"].weight)]["lr"] == pytest.approx(1e-4 * multiplier)
    for name, lr in [("embed", 2e-5), ("lm_head", 3e-5)]:
        group = groups[id(model[name].weight)]
        assert group["algorithm"] == "adamw"
        assert group["lr"] == pytest.approx(lr * multiplier)
        assert group["weight_decay"] == 0.0

    expected = nn.Parameter(model["router"].weight.detach().clone())
    reference = torch.optim.AdamW(
        [expected], lr=1e-4 * multiplier, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.1, fused=True
    )
    for step in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        expected.grad = model["router"].weight.grad.clone()
        opt.step()
        reference.step()
        torch.testing.assert_close(model["router"].weight, expected, rtol=0, atol=0)
        if step == 1:
            state = copy.deepcopy(opt.state_dict())
            opt = config.build(model)[0]
            opt.load_state_dict(state)


def test_adamw_patterns_split_transposed_expert_groups():
    class Experts(nn.Module):
        _nemo_transposed_matrix_parameters = ("selected", "remaining")

        def __init__(self):
            super().__init__()
            self.selected = nn.Parameter(torch.randn(2, 8, 12))
            self.remaining = nn.Parameter(torch.randn(2, 8, 12))

    torch.manual_seed(118)
    model = Experts()
    config = MuownConfig(adamw_param_patterns=["selected"], scalar_lr=1e-4)
    optimizer = config.build(model)[0]
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    assert groups[id(model.selected)]["algorithm"] == "adamw"
    assert not groups[id(model.selected)].get("matrix_transposed", False)
    assert groups[id(model.remaining)]["algorithm"] == "muon"
    assert groups[id(model.remaining)]["matrix_transposed"]
    expected = nn.Parameter(model.selected.detach().clone())
    reference = torch.optim.AdamW([expected], lr=1e-4, weight_decay=0.0, fused=True)
    for _ in range(3):
        model.selected.grad = torch.randn_like(model.selected)
        model.remaining.grad = torch.randn_like(model.remaining)
        expected.grad = model.selected.grad.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(model.selected, expected, rtol=0, atol=0)
