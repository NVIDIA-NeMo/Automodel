import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.optim.gradient_sketch.sketch import (
    SketchSettings,
    fill_projection_,
    install_sketches,
    local_weight_rows,
    make_projection,
    projection_seed,
    select_sketch_targets,
    sketch_buffer,
)


def test_settings_defaults_and_validation():
    settings = SketchSettings()
    assert settings.rank == 256 and settings.optimizer == "row_adam"
    assert settings.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    with pytest.raises(ValueError, match="rank"):
        SketchSettings(rank=0)
    with pytest.raises(ValueError, match="distribution"):
        SketchSettings(distribution="uniform")
    with pytest.raises(ValueError, match="optimizer"):
        SketchSettings(optimizer="adam")
    with pytest.raises(ValueError, match="beta2"):
        SketchSettings(beta2=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        SketchSettings(epsilon=0.0)


def test_projection_seed_ignores_wrapper_names():
    plain = projection_seed(42, 3, "model.layers.0.self_attn.q_proj")
    wrapped = projection_seed(42, 3, "model._fsdp_wrapped_module.layers.0._checkpoint_wrapped_module.self_attn.q_proj")
    assert plain == wrapped
    assert projection_seed(42, 4, "model.layers.0.self_attn.q_proj") != plain


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_make_projection_is_deterministic_and_unit_column_norm(distribution):
    a = make_projection(64, 512, 7, device="cpu", dtype=torch.float32, distribution=distribution)
    b = make_projection(64, 512, 7, device="cpu", dtype=torch.float32, distribution=distribution)
    c = make_projection(64, 512, 8, device="cpu", dtype=torch.float32, distribution=distribution)
    assert torch.equal(a, b) and not torch.equal(a, c)
    if distribution == "rademacher":
        assert torch.equal(a.abs(), torch.full_like(a, 1 / math.sqrt(64)))
    assert (a * a).sum(dim=0).mean().item() == pytest.approx(1.0, abs=0.1)


def test_fill_projection_rejects_bad_inputs():
    with pytest.raises(ValueError, match="distribution"):
        fill_projection_(torch.empty(4, 4), 1, distribution="uniform")
    with pytest.raises(ValueError, match="positive"):
        make_projection(0, 4, 1, device="cpu", dtype=torch.float32, distribution="rademacher")


def test_local_weight_rows_plain_and_3d():
    w2 = torch.zeros(6, 4)
    local, offset = local_weight_rows(w2)
    assert local is w2 and offset == 0
    w3 = torch.zeros(2, 3, 4)
    local, offset = local_weight_rows(w3)
    assert local.shape == (6, 4) and offset == 0


def _mlp():
    return nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3))


def test_select_targets_by_name_and_prefix():
    net = _mlp()
    targets = select_sketch_targets(net, SketchSettings(target_modules=("0", "2")))
    assert [t[0] for t in targets] == ["0", "2"]
    targets = select_sketch_targets(net, SketchSettings(target_modules=("0", "2"), target_prefixes=("2",)))
    assert [t[0] for t in targets] == ["2"]
    with pytest.raises(ValueError, match="matched no"):
        select_sketch_targets(net, SketchSettings(target_modules=("missing",)))
    with pytest.raises(TypeError, match="nn.Linear"):
        select_sketch_targets(net, SketchSettings(target_modules=("1",)))


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_direct_sketch_matches_projected_dense_gradient(recompute, distribution):
    torch.manual_seed(5)
    dense = _mlp()
    sketched = copy.deepcopy(dense)
    states = install_sketches(sketched, SketchSettings(rank=3, target_modules=("0",), distribution=distribution))
    x = torch.randn(6, 7, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    dense(x).square().sum().backward()
    for chunk in z.chunk(2):
        out = checkpoint(sketched, chunk, use_reentrant=False) if recompute else sketched(chunk)
        out.square().sum().backward()
    torch.testing.assert_close(states[0].sketch, dense[0].weight.grad @ states[0].projection.T)
    torch.testing.assert_close(x.grad, z.grad)
    torch.testing.assert_close(dense[2].weight.grad, sketched[2].weight.grad)
    assert sketched[0].weight.grad is None and not sketched[0].weight.requires_grad


def test_sketches_share_one_buffer_and_cast_is_persistent():
    net = _mlp()
    states = install_sketches(net, SketchSettings(rank=3, target_modules=("0", "2")))
    buffer = sketch_buffer(states)
    assert buffer.shape == (8, 3)
    assert all(s.sketch.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr() for s in states)
    cast_a = states[0].projection_as(torch.bfloat16)
    states[0].projection.fill_(1.0)
    states[0].refresh_cast()
    assert states[0].projection_as(torch.bfloat16) is cast_a and torch.all(cast_a == 1.0)


class _FusedExperts(nn.Module):
    def __init__(self, experts=2, hidden=4, inter=3):
        super().__init__()
        self.up = nn.Parameter(torch.randn(experts, inter, hidden))

    def forward(self, x):  # x: [tokens, hidden]; every expert sees every token
        return torch.einsum("th,eih->tei", x, self.up).sum(dim=1)


def test_fused_expert_parameter_is_sketched_without_dense_grad():
    torch.manual_seed(2)
    dense = _FusedExperts()
    sketched = copy.deepcopy(dense)
    states = install_sketches(sketched, SketchSettings(rank=2, target_modules=(), target_parameters=("up",)))
    x = torch.randn(5, 4)
    dense(x).square().sum().backward()
    sketched(x).square().sum().backward()
    expected = dense.up.grad.reshape(-1, 4) @ states[0].projection.T
    torch.testing.assert_close(states[0].sketch, expected)
    assert sketched.up.grad is None and states[0].swapped_weight is None
