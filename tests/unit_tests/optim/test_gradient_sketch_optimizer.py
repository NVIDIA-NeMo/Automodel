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
    for layer, source in zip(opt2.layers, opt.layers, strict=True):
        assert (
            opt2.state[layer.weight]["row_second_moment"].untyped_storage().data_ptr()
            == opt2.row_second_moment.untyped_storage().data_ptr()
        )
        for value in opt2.state[layer.weight].values():
            assert value.dtype == torch.float32
        if dtype == torch.bfloat16:
            torch.testing.assert_close(
                opt2.state[layer.weight]["master_weight"], opt.state[source.weight]["master_weight"]
            )
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


def test_to_device_relinks_views():
    net, opt = _build()
    _backward(net)
    opt.synchronize_and_clip(None, 1.0)
    opt.step()
    for state in opt.state.values():  # emulate generic offload that replaces every tensor
        for key, value in state.items():
            state[key] = value.clone()
    opt.to("cpu")
    for layer in opt.layers:
        assert (
            opt.state[layer.weight]["row_second_moment"].untyped_storage().data_ptr()
            == opt.row_second_moment.untyped_storage().data_ptr()
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
    for layer, source in zip(opt2.layers, opt.layers, strict=True):
        for value in opt2.state[layer.weight].values():
            assert value.dtype == torch.float32
        if dtype == torch.bfloat16:
            torch.testing.assert_close(
                opt2.state[layer.weight]["master_weight"], opt.state[source.weight]["master_weight"]
            )
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
