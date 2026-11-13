# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import gc

import pytest
import torch
from torch import nn

from nemo_automodel.components.optim import gradient_sketch
from nemo_automodel.components.optim.gradient_sketch import GradientSketchOptimizer, SketchSettings


def _mlp(dtype=torch.float32):
    torch.manual_seed(3)
    return nn.Sequential(nn.Linear(7, 5), nn.Tanh(), nn.Linear(5, 3)).to(dtype)


def _build(dtype=torch.float32, rule="row_adam", weight_decay=0.1, lr=0.01):
    net = _mlp(dtype)
    native = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    opt = GradientSketchOptimizer(net, native, SketchSettings(rank=3, target_modules=("0", "2"), optimizer=rule))
    return net, opt


def _backward(net, dtype=torch.float32):
    x = torch.randn(4, 7, dtype=dtype, requires_grad=True)
    net(x).float().square().sum().backward()


def test_construction_moves_targets_out_of_native_and_shares_groups():
    net, opt = _build()
    native_params = {id(p) for g in opt.native.param_groups for p in g["params"]}
    assert id(net[0].weight) not in native_params and id(net[2].weight) not in native_params
    assert id(net[0].bias) in native_params
    assert opt.param_groups[1] is opt.native.param_groups[0]
    assert opt.state is opt.native.state
    assert opt.param_groups[0]["update_count"] == 0
    assert opt.param_groups[0]["sketch_settings"]["rank"] == 3


def test_requires_fresh_single_group_native():
    net = _mlp()
    native = torch.optim.AdamW([{"params": [net[0].weight]}, {"params": [net[2].weight]}])
    with pytest.raises(ValueError, match="one parameter group"):
        GradientSketchOptimizer(net, native, SketchSettings(rank=2, target_modules=("0",)))


def test_sgd_matches_explicit_reconstruction_and_clipping():
    torch.manual_seed(9)
    net = _mlp()
    reference = copy.deepcopy(net)
    opt = GradientSketchOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0),
        SketchSettings(rank=3, target_modules=("0",), optimizer="sgd"),
    )
    x = torch.randn(4, 7, requires_grad=True)
    reference(x).square().sum().backward()
    net(x).square().sum().backward()
    a = opt.layers[0].projection
    expected = reference[0].weight.grad @ a.T @ a
    norm2 = expected.square().sum() + sum(
        p.grad.square().sum() for n, p in reference.named_parameters() if n != "0.weight"
    )
    norm = opt.synchronize_and_clip(None, 0.1)
    assert norm.item() == pytest.approx(norm2.sqrt().item(), rel=1e-5)
    before = net[0].weight.detach().clone()
    opt.step()
    torch.testing.assert_close(net[0].weight, before - 0.01 * expected * min(1.0, 0.1 / (norm2.sqrt().item() + 1e-6)))


def test_sgd_folds_weight_decay_into_the_update():
    torch.manual_seed(9)
    net = _mlp()
    reference = copy.deepcopy(net)
    lr, wd = 0.01, 0.1
    opt = GradientSketchOptimizer(
        net,
        torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd),
        SketchSettings(rank=3, target_modules=("0",), optimizer="sgd"),
    )
    assert opt.param_groups[0]["weight_decay"] == wd
    x = torch.randn(4, 7, requires_grad=True)
    reference(x).square().sum().backward()
    net(x).square().sum().backward()
    a = opt.layers[0].projection
    expected = reference[0].weight.grad @ a.T @ a
    norm = opt.synchronize_and_clip(None, 0.1)
    clip = min(1.0, 0.1 / (norm.item() + 1e-6))
    before = net[0].weight.detach().clone()
    opt.step()
    torch.testing.assert_close(net[0].weight, (1 - lr * wd) * before - lr * clip * expected)


def test_clip_scales_non_contiguous_3d_native_grad_in_place():
    torch.manual_seed(4)
    net = _mlp()
    extra = nn.Parameter(torch.randn(3, 4, 2))
    net.register_parameter("extra", extra)
    opt = GradientSketchOptimizer(
        net, torch.optim.AdamW(net.parameters(), lr=0.01), SketchSettings(rank=3, target_modules=("0", "2"))
    )
    _backward(net)
    extra.grad = torch.randn(4, 3, 2).transpose(0, 1)
    assert not extra.grad.is_contiguous()
    grad_before = extra.grad.clone()
    norm = opt.synchronize_and_clip(None, 1e-3)
    assert norm.item() > 1e-3
    torch.testing.assert_close(extra.grad, grad_before * (1e-3 / (norm + 1e-6)))


def test_sketch_state_keys_are_parameter_fqns():
    _, opt = _build(torch.bfloat16)
    assert set(opt.sketch_state_dict()) == {"row_second_moment", "master_weight.0.weight", "master_weight.2.weight"}


def test_step_requires_synchronize_first_and_zero_grad_resets():
    net, opt = _build()
    _backward(net)
    with pytest.raises(RuntimeError, match="synchronize_and_clip"):
        opt.step()
    opt.synchronize_and_clip(None, 1.0)
    with pytest.raises(RuntimeError, match="already"):
        opt.synchronize_and_clip(None, 1.0)
    opt.step()
    assert opt.update_count == 1
    opt.zero_grad()
    assert torch.all(opt.sketches == 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_state_dict_round_trip_restores_fp32_moments_and_masters(dtype):
    net, opt = _build(dtype)
    for _ in range(2):
        opt.zero_grad()
        _backward(net, dtype)
        opt.synchronize_and_clip(None, 1.0)
        opt.step()
    saved = copy.deepcopy(opt.state_dict())
    assert set(saved) == {"state", "param_groups"}  # standard torch shape, DCP-compatible
    expected_moments = opt.row_second_moment.clone()

    net2, opt2 = _build(dtype)
    net2.load_state_dict(net.state_dict())
    native_ids = {id(p) for g in opt.native.param_groups for p in g["params"]}
    assert len(saved["state"]) == len(native_ids)  # no sketch-target entries
    opt2.load_state_dict(saved)
    opt2.load_sketch_state_dict(copy.deepcopy(opt.sketch_state_dict()))
    assert opt2.update_count == 2
    torch.testing.assert_close(opt2.row_second_moment, expected_moments)
    targets = {id(layer.weight) for layer in opt2.layers}
    assert all(id(p) not in targets for p in opt2.state)  # sketch-target state stays off optimizer.state
    for master, source in zip(opt2.masters, opt.masters, strict=True):
        if dtype == torch.bfloat16:
            assert master is not None and master.dtype == torch.float32
            torch.testing.assert_close(master, source)
        else:
            assert master is None
    assert opt2.param_groups[1] is opt2.native.param_groups[0] and opt2.state is opt2.native.state
    # native Adam state restored too
    bias = net2[0].bias
    assert "exp_avg" in opt2.state[bias]
    # identical next step on both
    for o, n in ((opt, net), (opt2, net2)):
        torch.manual_seed(11)
        o.zero_grad()
        _backward(n, dtype)
        o.synchronize_and_clip(None, 1.0)
        o.step()
    for p, q in zip(net.parameters(), net2.parameters()):
        torch.testing.assert_close(p, q)


def test_load_rejects_mismatched_settings():
    net, opt = _build()
    saved = opt.state_dict()
    net2 = _mlp()
    other = GradientSketchOptimizer(
        net2, torch.optim.AdamW(net2.parameters(), lr=0.01), SketchSettings(rank=2, target_modules=("0", "2"))
    )
    with pytest.raises(ValueError, match="settings"):
        other.load_state_dict(saved)


def test_to_device_moves_native_state_masters_and_moments():
    net, opt = _build(torch.bfloat16)
    _backward(net, torch.bfloat16)
    opt.synchronize_and_clip(None, 1.0)
    opt.step()
    moments, masters = opt.row_second_moment.clone(), [m.clone() for m in opt.masters]
    opt.to("cpu")
    torch.testing.assert_close(opt.row_second_moment, moments)
    for master, before in zip(opt.masters, masters, strict=True):
        torch.testing.assert_close(master, before)
    assert all(
        v.device.type == "cpu" for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor)
    )


def test_sketches_reduce_with_one_collective(monkeypatch):
    net, opt = _build(rule="sgd")
    monkeypatch.setattr(opt, "_check_sharded_targets", lambda: None)  # plain tensors stand in for DTensor shards
    calls = []
    monkeypatch.setattr(gradient_sketch.optimizer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gradient_sketch.optimizer.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        gradient_sketch.optimizer.dist, "all_reduce", lambda t, op=None, group=None: calls.append(tuple(t.shape))
    )
    _backward(net)
    opt.synchronize_and_clip(None, 1.0)
    # two fingerprint checks (one-time), one flat buffer, one scalar norm
    assert len(calls) == 4 and calls.count((8, 3)) == 1


def test_multi_rank_requires_sharded_targets(monkeypatch):
    net, opt = _build(rule="sgd")
    monkeypatch.setattr(gradient_sketch.optimizer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gradient_sketch.optimizer.dist, "get_world_size", lambda group: 2)
    _backward(net)
    with pytest.raises(RuntimeError, match="FSDP-sharded"):
        opt.synchronize_and_clip(None, 1.0)


@pytest.fixture
def gloo_group(tmp_path):
    import torch.distributed as dist

    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "store"), 1), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rule", ["row_adam", "sgd"])
def test_optimizer_state_wrapper_round_trip(rule, dtype, gloo_group):
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

    net, opt = _build(dtype, rule=rule)
    for _ in range(2):
        opt.zero_grad()
        _backward(net, dtype)
        opt.synchronize_and_clip(None, 1.0)
        opt.step()
    saved = copy.deepcopy(OptimizerState(net, opt).state_dict())
    assert "optim" in saved and "sketch" in saved

    net2, opt2 = _build(dtype, rule=rule)
    net2.load_state_dict(net.state_dict())
    wrapper = OptimizerState(net2, opt2)
    wrapper.state_dict()  # DCP builds the load skeleton this way before loading
    wrapper.load_state_dict(saved)
    assert opt2.update_count == 2
    if rule == "row_adam":
        torch.testing.assert_close(opt2.row_second_moment, opt.row_second_moment)
    for master, source in zip(opt2.masters, opt.masters, strict=True):
        if dtype == torch.bfloat16:
            assert master is not None and master.dtype == torch.float32
            torch.testing.assert_close(master, source)
    assert "exp_avg" in opt2.state[net2[0].bias]
    torch.testing.assert_close(opt2.state[net2[0].bias]["exp_avg"], opt.state[net[0].bias]["exp_avg"])
    for o, n in ((opt, net), (opt2, net2)):
        torch.manual_seed(11)
        o.zero_grad()
        _backward(n, dtype)
        o.synchronize_and_clip(None, 1.0)
        o.step()
    for p, q in zip(net.parameters(), net2.parameters()):
        torch.testing.assert_close(p, q)


def test_load_sketch_state_dict_rejects_missing_and_mismatched():
    net, opt = _build(torch.bfloat16)
    saved = copy.deepcopy(opt.sketch_state_dict())
    with pytest.raises(ValueError, match="gradient sketch checkpoint is missing"):
        opt.load_sketch_state_dict({k: v for k, v in saved.items() if k != "row_second_moment"})
    with pytest.raises(ValueError, match="shape mismatch"):
        opt.load_sketch_state_dict({**saved, "row_second_moment": torch.zeros(3, 1)})
    with pytest.raises(ValueError, match="unexpected"):
        opt.load_sketch_state_dict({**saved, "bogus": torch.zeros(1)})


class _FusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.randn(2, 3, 4))

    def forward(self, x):
        return torch.einsum("th,eih->tei", x, self.up).sum(dim=1)


def test_bf16_fused_expert_master_is_stored_3d_and_updated_as_rows():
    def build():
        torch.manual_seed(5)
        model = _FusedExperts().to(torch.bfloat16)
        native = torch.optim.AdamW(model.parameters(), lr=0.01)
        settings = SketchSettings(rank=2, target_modules=(), target_parameters=("up",))
        return model, GradientSketchOptimizer(model, native, settings)

    model, opt = build()
    before = model.up.detach().clone()
    model(torch.randn(5, 4, dtype=torch.bfloat16)).float().square().sum().backward()
    opt.synchronize_and_clip(None, 1.0)
    opt.step()
    (master,) = opt.masters
    assert master.shape == (2, 3, 4) and master.dtype == torch.float32
    assert not torch.equal(model.up, before)
    assert torch.equal(master.to(torch.bfloat16), model.up)

    assert set(opt.sketch_state_dict()) == {"row_second_moment", "master_weight.up"}

    model2, opt2 = build()
    opt2.load_sketch_state_dict(copy.deepcopy(opt.sketch_state_dict()))
    torch.testing.assert_close(opt2.masters[0], master)
    opt2.reset_master_weights()
    assert torch.equal(opt2.masters[0].to(torch.bfloat16), model2.up)


def test_backward_leaves_no_cyclic_garbage():
    net, opt = _build()
    _backward(net)
    gc.collect()
    gc.disable()
    try:
        for _ in range(3):
            _backward(net)
        unreachable = gc.collect()
    finally:
        gc.enable()
    assert unreachable == 0


@pytest.mark.parametrize("placements", ["replicate", "two_shards"])
def test_check_sharded_targets_rejects_replicated_or_multiply_sharded(placements, gloo_group):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    net, opt = _build(rule="sgd")
    mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("a", "b"))
    layout = [Replicate(), Shard(0)] if placements == "replicate" else [Shard(0), Shard(0)]
    row_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("c",))
    for layer in opt.layers:
        weight = distribute_tensor(layer.weight.detach(), row_mesh, [Shard(0)])
        layer.module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
    opt._check_sharded_targets()  # one row shard per target is accepted
    first = opt.layers[0]
    first.module._parameters["weight"] = nn.Parameter(
        distribute_tensor(first.weight.full_tensor(), mesh, layout), requires_grad=False
    )
    with pytest.raises(RuntimeError, match=f"sharded once along rows by FSDP; {first.name} "):
        opt._check_sharded_targets()


def test_state_holds_only_native_parameters():
    """Sketch-target state must stay out of ``optimizer.state``.

    DCP's ``_init_optim_state`` skips initialization when ``state`` is non-empty, which would
    leave a non-Adam native optimizer (TE FusedAdam, SGD with momentum) without a load skeleton.
    """
    net, opt = _build(torch.bfloat16)
    assert len(opt.state) == 0
    _backward(net, torch.bfloat16)
    opt.synchronize_and_clip(None, 1.0)
    opt.step()
    targets = {id(layer.weight) for layer in opt.layers}
    assert opt.state and all(id(p) not in targets for p in opt.state)


def test_optimizer_state_wrapper_round_trip_keeps_non_adam_native_history(gloo_group):
    """Resume through OptimizerState with a lazily-stated native optimizer that is not Adam."""
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

    def build():
        net = _mlp(torch.bfloat16)
        native = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        return net, GradientSketchOptimizer(net, native, SketchSettings(rank=3, target_modules=("0", "2")))

    net, opt = build()
    for _ in range(2):
        opt.zero_grad()
        _backward(net, torch.bfloat16)
        opt.synchronize_and_clip(None, 1.0)
        opt.step()
    saved = copy.deepcopy(OptimizerState(net, opt).state_dict())

    net2, opt2 = build()
    net2.load_state_dict(net.state_dict())
    wrapper = OptimizerState(net2, opt2)
    wrapper.state_dict()  # DCP builds the load skeleton this way before loading
    wrapper.load_state_dict(saved)
    bias = net2[0].bias
    assert "momentum_buffer" in opt2.state[bias]
    torch.testing.assert_close(opt2.state[bias]["momentum_buffer"], opt.state[net[0].bias]["momentum_buffer"])
    for o, n in ((opt, net), (opt2, net2)):
        torch.manual_seed(11)
        o.zero_grad()
        _backward(n, torch.bfloat16)
        o.synchronize_and_clip(None, 1.0)
        o.step()
    for p, q in zip(net.parameters(), net2.parameters()):
        torch.testing.assert_close(p, q)


def test_freeze_non_target_registers_no_empty_native_group_and_round_trips(gloo_group):
    """With every non-target frozen, no empty native group may reach DCP's flattened state dict."""
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

    def build():
        net = _mlp(torch.bfloat16)
        native = torch.optim.AdamW(net.parameters(), lr=0.01)
        settings = SketchSettings(rank=3, target_modules=("0", "2"), freeze_non_target=True)
        return net, GradientSketchOptimizer(net, native, settings)

    net, opt = build()
    assert len(opt.param_groups) == 1
    assert all(not p.requires_grad for p in net.parameters())
    for _ in range(2):
        opt.zero_grad()
        _backward(net, torch.bfloat16)
        opt.synchronize_and_clip(None, 1.0)
        opt.step()
    saved = copy.deepcopy(OptimizerState(net, opt).state_dict())

    net2, opt2 = build()
    net2.load_state_dict(net.state_dict())
    wrapper = OptimizerState(net2, opt2)
    wrapper.state_dict()
    wrapper.load_state_dict(saved)
    assert opt2.update_count == 2
    torch.testing.assert_close(opt2.row_second_moment, opt.row_second_moment)
    for o, n in ((opt, net), (opt2, net2)):
        torch.manual_seed(11)
        o.zero_grad()
        _backward(n, torch.bfloat16)
        o.synchronize_and_clip(None, 1.0)
        o.step()
    for p, q in zip(net.parameters(), net2.parameters()):
        torch.testing.assert_close(p, q)
