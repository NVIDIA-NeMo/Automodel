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
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.optim.weight_rounding import (
    WeightRounding,
    _refresh_bf16_weights,
    _stochastic_round_bf16,
    attach_bf16_weight_rounding,
)

pytestmark = pytest.mark.cpu_only


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare local tensors of arbitrary shape [...] including signed-zero bits.

    Args:
        actual: Local tensor of shape [...], with arbitrary rank.
        expected: Local tensor with identical shape and dtype. Neither input is mutated.
    """
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _optimizer(parameter: torch.Tensor, rounding: WeightRounding) -> torch.optim.AdamW:
    """Construct a native scalar AdamW with BF16 rounding hooks.

    Args:
        parameter: FP32 tensor of arbitrary global shape [...], optionally a
            Shard/Replicate DTensor; its current values lie on the BF16 grid.
        rounding: BF16 materialization mode.

    Returns:
        Native AdamW referencing the original parameter object.
    """
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, betas=(0.8, 0.97), weight_decay=0.1, foreach=False, fused=False)
    attach_bf16_weight_rounding(optimizer, rounding=rounding, rounding_seed=7)
    return optimizer


def test_stochastic_rounding_independent_neighbors_and_rne_negative_control() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.tensor([1.001953125, -1.001953125]).repeat(200000, 1)
    result = _stochastic_round_bf16(values, generator).double()
    assert ((result[:, 0] == 1.0) | (result[:, 0] == 1.0078125)).all()
    assert ((result[:, 1] == -1.0) | (result[:, 1] == -1.0078125)).all()
    assert (result.mean(0) - values[0].double()).abs().max() < 4e-5
    assert (values.bfloat16().double().mean(0) - values[0].double()).abs().min() > 0.001


def test_rounding_special_values_noncontiguous_and_global_rng() -> None:
    values = torch.tensor([[0.0, -0.0, 1.0, -2.0], [float("inf"), -float("inf"), float("nan"), 0.5]]).t()
    before = torch.get_rng_state().clone()
    result = _stochastic_round_bf16(values, torch.Generator().manual_seed(1))
    torch.testing.assert_close(result.float(), values, rtol=0, atol=0, equal_nan=True)
    assert torch.signbit(result[1, 0])
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(TypeError):
        _stochastic_round_bf16(values.bfloat16(), torch.Generator())


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_native_adam_oracle_protected_lazy_state_and_resume(rounding: WeightRounding) -> None:
    generator = torch.Generator().manual_seed(13)
    initial = torch.randn(128, 16, generator=generator).bfloat16().float()
    parameter = torch.nn.Parameter(initial.clone())
    protected = torch.nn.Parameter(initial[:1].clone())
    unused = torch.nn.Parameter(initial[:1].clone())
    reference = torch.nn.Parameter(initial.clone())
    protected_reference = torch.nn.Parameter(protected.detach().clone())
    groups = [{"params": [parameter, unused], "quantize": True}, {"params": [protected], "quantize": False}]
    optimizer = torch.optim.AdamW(groups, lr=1e-3, betas=(0.8, 0.97), weight_decay=0.1, foreach=False, fused=False)
    attach_bf16_weight_rounding(optimizer, rounding=rounding, rounding_seed=7)
    oracle = torch.optim.AdamW(
        [reference, protected_reference], lr=1e-3, betas=(0.8, 0.97), weight_decay=0.1, foreach=False, fused=False
    )
    assert type(optimizer) is torch.optim.AdamW
    assert not optimizer.state
    for step in range(5):
        gradient = torch.randn(initial.shape, generator=generator)
        parameter.grad = gradient.clone()
        reference.grad = gradient.clone()
        protected.grad = gradient[:1].clone()
        protected_reference.grad = gradient[:1].clone()
        rng = torch.get_rng_state().clone()
        optimizer.step()
        oracle.step()
        _assert_bits(torch.get_rng_state(), rng)
        _assert_bits(optimizer.state[parameter]["master"], reference)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            _assert_bits(optimizer.state[parameter][key], oracle.state[reference][key])
        _assert_bits(protected, protected_reference)
        assert "master" not in optimizer.state[protected]
        assert unused not in optimizer.state
        _assert_bits(parameter, parameter.bfloat16().float())
        if rounding == "nearest":
            _assert_bits(parameter, reference.bfloat16().float())
        if step == 1:
            restored_parameter = torch.nn.Parameter(parameter.detach().clone())
            restored_unused = torch.nn.Parameter(unused.detach().clone())
            restored_protected = torch.nn.Parameter(protected.detach().clone())
            restored = torch.optim.AdamW(
                [
                    {"params": [restored_parameter, restored_unused], "quantize": True},
                    {"params": [restored_protected], "quantize": False},
                ],
                lr=1e-3,
                betas=(0.8, 0.97),
                weight_decay=0.1,
                foreach=False,
                fused=False,
            )
            attach_bf16_weight_rounding(restored, rounding=rounding, rounding_seed=7)
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        if step > 1:
            restored_parameter.grad = gradient.clone()
            restored_protected.grad = gradient[:1].clone()
            restored.step()
            _assert_bits(restored_parameter, parameter)
            for key in ("master", "exp_avg", "exp_avg_sq", "step"):
                _assert_bits(restored.state[restored_parameter][key], optimizer.state[parameter][key])
            assert [{k: v for k, v in g.items() if k != "params"} for g in restored.param_groups] == [
                {k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups
            ]


def test_no_gradient_refresh_preserves_master_moments_and_legacy_parameter_index() -> None:
    protected = torch.nn.Parameter(torch.ones(4))
    parameter = torch.nn.Parameter(torch.ones(8192))
    optimizer = torch.optim.AdamW(
        [{"params": [protected], "quantize": False}, {"params": [parameter]}],
        lr=0.0013,
        betas=(0.8, 0.97),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )
    attach_bf16_weight_rounding(optimizer, rounding="stochastic", rounding_seed=7)
    parameter.grad = torch.full_like(parameter, 0.25)
    optimizer.step()
    saved = {k: v.clone() for k, v in optimizer.state[parameter].items()}
    old_q = parameter.detach().clone()
    parameter.grad = None
    rng = torch.get_rng_state().clone()
    optimizer.step()
    for key, value in saved.items():
        _assert_bits(optimizer.state[parameter][key], value)
    assert [g["quantization_step"] for g in optimizer.param_groups] == [2, 2]
    assert not torch.equal(old_q, parameter)
    _assert_bits(torch.get_rng_state(), rng)
    # This explicit legacy wire contract includes the protected parameter as index 1.
    expected_seed = (7 + 2 * 1000003 + 2 * 1000000007) % (2**63 - 1)
    expected = _stochastic_round_bf16(saved["master"], torch.Generator().manual_seed(expected_seed)).float()
    _assert_bits(parameter, expected)
    wrong_seed = (7 + 1000003 + 2 * 1000000007) % (2**63 - 1)
    wrong = _stochastic_round_bf16(saved["master"], torch.Generator().manual_seed(wrong_seed)).float()
    assert not torch.equal(parameter, wrong)
    assert protected not in optimizer.state
    _assert_bits(protected, torch.ones_like(protected))


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_recomputation_uses_one_policy_realization(rounding: WeightRounding) -> None:
    model = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(model.weight.bfloat16().float())
    optimizer = _optimizer(model.weight, rounding)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    q = model.weight.detach().clone()
    seen = []

    def capture(module: torch.nn.Linear, args: tuple[torch.Tensor, ...]) -> None:
        """Observe a linear call without changing parameter storage.

        Args:
            module: Linear layer holding weight [output, input].
            args: Tuple containing the input tensor [batch, input].
        """
        seen.append(module.weight.detach().clone())

    handle = model.register_forward_pre_hook(capture)
    x = torch.randn(3, 8, requires_grad=True)
    checkpoint(model, x, use_reentrant=True).square().sum().backward()
    handle.remove()
    assert len(seen) == 2
    for value in seen:
        _assert_bits(value, q)
    _assert_bits(model.weight, q)
    assert torch.isfinite(model.weight.grad).all()


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_automodel_checkpoint_wrapper_with_unused_parameter(rounding: WeightRounding) -> None:
    from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

    model = torch.nn.Linear(8, 4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(parameter.bfloat16().float())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False, fused=False)
    attach_bf16_weight_rounding(optimizer, rounding=rounding, rounding_seed=7)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    assert model.bias not in optimizer.state
    q = [p.detach().clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    wrapper = OptimizerState(model, optimizer)
    saved = copy.deepcopy(wrapper.state_dict())
    for parameter, value in zip(model.parameters(), q):
        _assert_bits(parameter, value)
    _assert_bits(torch.get_rng_state(), rng)
    assert optimizer.param_groups[0]["quantization_step"] == 1
    assert optimizer.state[model.bias]["step"].item() == 0
    _assert_bits(optimizer.state[model.bias]["master"], model.bias)
    restored_model = copy.deepcopy(model)
    restored = torch.optim.AdamW(restored_model.parameters(), lr=1e-3, foreach=False, fused=False)
    attach_bf16_weight_rounding(restored, rounding=rounding, rounding_seed=7)
    restored_wrapper = OptimizerState(restored_model, restored)
    restored_wrapper.state_dict()  # Native skeleton construction, not an optimizer step.
    assert restored.param_groups[0]["quantization_step"] == 0
    restored_wrapper.load_state_dict(saved)
    for parameter, other in zip(model.parameters(), restored_model.parameters()):
        for key, value in optimizer.state[parameter].items():
            _assert_bits(value, restored.state[other][key])
        parameter.grad = torch.full_like(parameter, 0.2)
        other.grad = parameter.grad.clone()
    optimizer.step()
    restored.step()
    for parameter, other in zip(model.parameters(), restored_model.parameters()):
        _assert_bits(parameter, other)


def test_checkpoint_boundary_does_not_repair_nonzero_state_or_initialize_empty_state() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = _optimizer(parameter, "stochastic")
    optimizer.state_dict()
    assert not optimizer.state
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    del optimizer.state[parameter]["master"]
    with pytest.raises(ValueError, match="nonzero or incomplete"):
        optimizer.state_dict()


def test_legacy_checkpoint_metadata_restored_without_new_phase_reset() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = _optimizer(parameter, "nearest")
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.param_groups[0].update(faircast_phase=1, faircast_phase_start_global_step=5000, quantization_step=5001)
    state = copy.deepcopy(optimizer.state_dict())
    other = torch.nn.Parameter(parameter.detach().clone())
    restored = _optimizer(other, "nearest")
    restored.load_state_dict(state)
    assert restored.param_groups[0]["faircast_phase"] == 1
    assert restored.param_groups[0]["faircast_phase_start_global_step"] == 5000
    assert restored.param_groups[0]["quantization_step"] == 5001
    for key, value in optimizer.state[parameter].items():
        _assert_bits(restored.state[other][key], value)


def test_unsupported_optimizer_parameter_closure_and_restore_contracts() -> None:
    with pytest.raises(TypeError, match="exact torch.optim.AdamW"):
        attach_bf16_weight_rounding(torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.1))
    with pytest.raises(TypeError, match="FP32"):
        _optimizer(torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)), "nearest")
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = _optimizer(parameter, "nearest")
    with pytest.raises(ValueError, match="already attached"):
        attach_bf16_weight_rounding(optimizer)
    with pytest.raises(ValueError, match="closures"):
        optimizer.step(lambda: torch.tensor(1.0))
    plain = torch.optim.AdamW([torch.nn.Parameter(torch.ones(8))], foreach=False, fused=False)
    with pytest.raises(ValueError, match="plain AdamW"):
        optimizer.load_state_dict(plain.state_dict())
    for variant in ("foreach", "fused"):
        kwargs = {variant: True}
        unsupported = torch.optim.AdamW([torch.nn.Parameter(torch.ones(8))], **kwargs)
        with pytest.raises(ValueError, match="non-fused"):
            attach_bf16_weight_rounding(unsupported)


def _distributed_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}")
    try:
        mesh = init_device_mesh("cpu", (2,))
        for placement in (Shard(0), Shard(1), Replicate()):
            initial = torch.ones(32, 32)
            parameter = torch.nn.Parameter(distribute_tensor(initial, mesh, [placement]))
            reference = torch.nn.Parameter(parameter.to_local().detach().clone())
            optimizer = _optimizer(parameter, "stochastic")
            oracle = torch.optim.AdamW(
                [reference], lr=1e-3, betas=(0.8, 0.97), weight_decay=0.1, foreach=False, fused=False
            )
            for step in range(3):
                parameter.grad = distribute_tensor(torch.full_like(initial, step + 1.0), mesh, [placement])
                reference.grad = parameter.grad.to_local().clone()
                optimizer.step()
                oracle.step()
                _assert_bits(optimizer.state[parameter]["master"].to_local(), reference)
                for key in ("exp_avg", "exp_avg_sq"):
                    _assert_bits(optimizer.state[parameter][key].to_local(), oracle.state[reference][key])
                if step == 1:
                    saved = copy.deepcopy(optimizer.state_dict())
                    optimizer = _optimizer(parameter, "stochastic")
                    optimizer.load_state_dict(saved)
                theta = optimizer.state[parameter]["master"].to_local().clone()
                _refresh_bf16_weights(optimizer)
                _assert_bits(optimizer.state[parameter]["master"].to_local(), theta)
                peers = [torch.empty_like(parameter.to_local()) for _ in range(2)]
                dist.all_gather(peers, parameter.to_local())
                if isinstance(placement, Replicate):
                    _assert_bits(peers[0], peers[1])
                else:
                    assert not torch.equal(peers[0], peers[1])
            parameter.grad = DTensor.from_local(torch.ones_like(initial), mesh, [Partial()])
            with pytest.raises(ValueError, match="fully reduced"):
                optimizer.step()
        tiny = torch.nn.Parameter(distribute_tensor(torch.ones(1, 8), mesh, [Shard(0)]))
        optimizer = _optimizer(tiny, "stochastic")
        tiny.grad = distribute_tensor(torch.ones(1, 8), mesh, [Shard(0)])
        optimizer.step()
        assert optimizer.state[tiny]["master"].to_local().shape == tiny.to_local().shape
        assert optimizer.param_groups[0]["quantization_step"] == 1
    finally:
        dist.destroy_process_group()


def test_actual_two_rank_shards_replicas_empty_local_and_restore(tmp_path: Path) -> None:
    mp.spawn(_distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("rounding", [None, "nearest", "stochastic"])
def test_config_empty_global_group_only_filtered_when_rounding_enabled(rounding: WeightRounding | None) -> None:
    from nemo_automodel.components.optim.optimizer import AdamWConfig

    model = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(model.weight.bfloat16().float())
    groups = [{"params": [model.weight], "quantize": True}, {"params": [], "quantize": False}]
    optimizer = AdamWConfig(weight_rounding=rounding).build_from_param_groups(groups)
    assert len(groups) == 2 and groups[1]["params"] == []  # Caller-owned list is unchanged.
    assert len(optimizer.param_groups) == (2 if rounding is None else 1)
    assert optimizer.param_groups[0]["params"][0] is model.weight
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    assert len(scheduler.base_lrs) == len(optimizer.param_groups)
    if rounding is None:
        assert all("quantization_step" not in group for group in optimizer.param_groups)
    else:
        from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        scheduler.step()
        saved = OptimizerState(model, optimizer, scheduler).state_dict()
        assert saved
        assert optimizer.param_groups[0]["quantization_step"] == 1
