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

"""TE FusedAdam construction must keep zero-numel local shards.

FSDP2 shards every parameter along dim-0 across the shard group, so any
parameter with dim-0 smaller than the group (e.g. a LoRA ``lora_A`` weight with
fewer rows than ranks, or small vision-tower biases on a wide mesh) leaves
zero-numel local shards on the tail ranks.  Both TE FusedAdam construction
paths (the typed :class:`FusedAdamConfig` and the YAML ``_target_`` factory
escape hatch) must hand TE the same parameter list on every rank:
``state_dict()`` keys optimizer state by parameter position, so a rank that
left a parameter out would save another parameter under the same checkpoint
key.  The empty shards are kept out of TE's native multi-tensor kernel by the
empty-shard patch, which both paths apply before they construct the optimizer.
"""

import functools
import sys
import types
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.optim.optimizer import (
    FusedAdamConfig,
    OptimizerFromFactoryConfig,
)

_FUSED_ADAM_MODULE = "transformer_engine.pytorch.optimizers.fused_adam"


class _RecordingFusedAdam:
    """Stand-in for TE FusedAdam that records its constructor arguments.

    Args:
        params: Flat list of parameters or list of param-group dicts, recorded
            as-is on the class (parameters keep their original tensor/DTensor
            types and shapes; no tensor is read or modified).
    """

    last_params = None
    last_kwargs = None

    def __init__(self, params, **kwargs):
        type(self).last_params = list(params)
        type(self).last_kwargs = kwargs
        self.param_groups = []

    def _apply_scale(
        self, state_name: str, unscaled_state: torch.Tensor, scaled_state: Any, scale: torch.Tensor
    ) -> None:
        """Stand-in for the TE method that the empty-shard patch wraps; does nothing.

        Args:
            state_name: Name of the optimizer state.
            unscaled_state: Tensor with the shape of the parameter's local shard; not read.
            scaled_state: Stored state with the same shape; not written.
            scale: Tensor of shape [1]; not written.
        """


class _RecordingApplier:
    """Stand-in for TE's ``multi_tensor_applier`` that records what reaches the native kernel."""

    def __init__(self) -> None:
        self.calls: list[list[list[torch.Tensor]]] = []

    def __call__(self, op: Any, noop_flag_buffer: Any, tensor_lists: list[list[torch.Tensor]], *args: Any) -> str:
        """Record one native kernel call.

        Args:
            op: Native optimizer kernel; not called.
            noop_flag_buffer: Overflow flag buffer; not read.
            tensor_lists: Role-major lists [role][parameter]. Each entry is a
                tensor of arbitrary shape, possibly with zero elements. The
                lists are copied; no tensor is read or modified.
            *args: Scalar kernel arguments; not used.

        Returns:
            The string ``"updated"``.
        """
        self.calls.append([list(tensors) for tensors in tensor_lists])
        return "updated"


def _te_stub_modules() -> dict[str, types.ModuleType]:
    """Module stubs exposing ``_RecordingFusedAdam`` as TE's ``FusedAdam``.

    ``transformer_engine.pytorch.optimizers.fused_adam`` carries a fresh
    ``_RecordingApplier`` as ``multi_tensor_applier``, the attribute that the
    empty-shard patch wraps.
    """
    fused_adam_mod = types.ModuleType(_FUSED_ADAM_MODULE)
    fused_adam_mod.FusedAdam = _RecordingFusedAdam
    fused_adam_mod.multi_tensor_applier = _RecordingApplier()
    optimizers_mod = types.ModuleType("transformer_engine.pytorch.optimizers")
    optimizers_mod.FusedAdam = _RecordingFusedAdam
    optimizers_mod.fused_adam = fused_adam_mod
    pytorch_mod = types.ModuleType("transformer_engine.pytorch")
    pytorch_mod.optimizers = optimizers_mod
    te_mod = types.ModuleType("transformer_engine")
    te_mod.pytorch = pytorch_mod
    return {
        "transformer_engine": te_mod,
        "transformer_engine.pytorch": pytorch_mod,
        "transformer_engine.pytorch.optimizers": optimizers_mod,
        _FUSED_ADAM_MODULE: fused_adam_mod,
    }


@pytest.fixture
def stub_te_modules(monkeypatch):
    """Install fake ``transformer_engine`` modules and return them by module name.

    Makes the CPU tests independent of a TransformerEngine installation and
    lets them observe exactly which parameters reach the TE constructor and
    which tensors reach the native multi-tensor kernel.
    """
    modules = _te_stub_modules()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # The empty-shard patch wraps this method on the class; restore it after the test.
    monkeypatch.setattr(_RecordingFusedAdam, "_apply_scale", _RecordingFusedAdam._apply_scale)
    _RecordingFusedAdam.last_params = None
    _RecordingFusedAdam.last_kwargs = None
    return modules


@pytest.fixture
def stub_te_fused_adam(stub_te_modules):
    """Install the fake ``transformer_engine`` modules and return the fake ``FusedAdam``."""
    return _RecordingFusedAdam


@pytest.fixture
def single_rank_pg():
    """Single-rank gloo process group, enough to build CPU DTensors."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    yield
    dist.destroy_process_group()


def _assert_same_parameters(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> None:
    """Assert that two parameter lists hold the same objects in the same order.

    Args:
        actual: Parameters of arbitrary shape, possibly with zero elements or
            DTensors with an empty local shard; compared by identity only.
        expected: Parameters of arbitrary shape; compared by identity only.
    """
    assert len(actual) == len(expected)
    assert all(a is e for a, e in zip(actual, expected))


def _native_columns_after_patch(stub_te_modules: dict[str, types.ModuleType], native: _RecordingApplier) -> list[int]:
    """Send one empty and one nonempty column through the installed applier.

    Returns:
        Number of tensors per role that reached the fake native kernel.
    """
    applier = stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier
    columns = [[torch.empty(0), torch.ones(3)] for _ in range(4)]
    assert applier("kernel", "overflow", columns, 1e-3) == "updated"
    assert len(native.calls) == 1
    return [len(tensors) for tensors in native.calls[0]]


class TestFusedAdamConfigKeepsEmptyLocalShards:
    def test_flat_params_keep_empty(self, stub_te_fused_adam):
        empty = nn.Parameter(torch.empty(0))
        kept = nn.Parameter(torch.ones(3))

        FusedAdamConfig(lr=1e-3)._build_optimizer([empty, kept])

        _assert_same_parameters(stub_te_fused_adam.last_params, [empty, kept])
        assert stub_te_fused_adam.last_kwargs["lr"] == 1e-3

    def test_flat_params_all_empty_are_kept(self, stub_te_fused_adam):
        # A rank whose shards are all empty still holds every parameter, at the
        # same positions as the other ranks; it has no native update to run.
        empty = nn.Parameter(torch.empty(0))

        FusedAdamConfig()._build_optimizer([empty])

        _assert_same_parameters(stub_te_fused_adam.last_params, [empty])

    def test_param_groups_keep_empty_params_and_options(self, stub_te_fused_adam):
        empty = nn.Parameter(torch.empty(0))
        kept_a = nn.Parameter(torch.ones(3))
        kept_b = nn.Parameter(torch.ones(2))

        FusedAdamConfig().build_from_param_groups(
            [
                {"params": [empty, kept_a], "lr": 0.25},
                {"params": [kept_b], "lr": 0.5},
            ]
        )

        first, second = stub_te_fused_adam.last_params
        _assert_same_parameters(first["params"], [empty, kept_a])
        _assert_same_parameters(second["params"], [kept_b])
        assert first["lr"] == 0.25
        assert second["lr"] == 0.5

    def test_param_group_with_only_empty_shards_is_kept(self, stub_te_fused_adam):
        # The group stays, with its parameters and options, so param_groups has
        # the same layout on every rank (LR/WD schedulers address groups
        # positionally) and no group is empty.
        kept = nn.Parameter(torch.ones(3))
        empty = nn.Parameter(torch.empty(0))

        FusedAdamConfig().build_from_param_groups(
            [
                {"params": [kept], "lr": 0.25},
                {"params": [empty], "lr": 0.5},
            ]
        )

        first, second = stub_te_fused_adam.last_params
        _assert_same_parameters(first["params"], [kept])
        _assert_same_parameters(second["params"], [empty])
        assert first["lr"] == 0.25
        assert second["lr"] == 0.5

    def test_dtensor_with_empty_local_shard_is_kept(self, stub_te_fused_adam, single_rank_pg):
        mesh = init_device_mesh("cpu", (1,))
        # Globally non-empty (4, 3) parameter whose local dim-0 shard on this rank is empty —
        # the shape FSDP2 leaves on tail ranks when dim-0 < shard-group size.
        locally_empty = nn.Parameter(
            DTensor.from_local(torch.empty(0, 3), mesh, [Shard(0)], run_check=False, shape=(4, 3), stride=(3, 1))
        )
        locally_present = nn.Parameter(DTensor.from_local(torch.ones(2, 3), mesh, [Shard(0)], run_check=False))
        assert locally_empty.numel() == 12
        assert locally_empty.to_local().numel() == 0

        FusedAdamConfig()._build_optimizer([locally_empty, locally_present])

        _assert_same_parameters(stub_te_fused_adam.last_params, [locally_empty, locally_present])

    def test_empty_shards_do_not_reach_native_kernel(self, stub_te_modules):
        native = stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier

        FusedAdamConfig()._build_optimizer([nn.Parameter(torch.empty(0)), nn.Parameter(torch.ones(3))])

        assert _native_columns_after_patch(stub_te_modules, native) == [1, 1, 1, 1]


class _TinyModel(nn.Module):
    """Linear probe plus a zero-numel trainable parameter."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.empty = nn.Parameter(torch.empty(0))


class _OnlyEmptyTrainableModel(nn.Module):
    """Frozen linear; the sole trainable parameter has a zero-numel (local) shard."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.linear.requires_grad_(False)
        self.empty = nn.Parameter(torch.empty(0))


class TestFactoryEscapeHatchKeepsEmptyLocalShards:
    def test_te_fused_adam_factory_keeps_empty(self, stub_te_modules):
        native = stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier
        cfg = OptimizerFromFactoryConfig(factory=_RecordingFusedAdam, kwargs={"lr": 1e-3})
        model = _TinyModel()

        cfg.build(model)

        # empty + linear weight + linear bias, in named_parameters() order
        _assert_same_parameters(_RecordingFusedAdam.last_params, list(model.parameters()))
        assert len(_RecordingFusedAdam.last_params) == 3
        assert _native_columns_after_patch(stub_te_modules, native) == [1, 1, 1, 1]

    def test_partial_wrapped_te_fused_adam_factory_keeps_empty(self, stub_te_modules):
        # functools.partial wrappers must be unwrapped by the TE FusedAdam identity
        # check; without unwrapping, the empty-shard patch would not be applied and
        # the empty shard would reach the native kernel.
        native = stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier
        factory = functools.partial(_RecordingFusedAdam, bias_correction=True)
        cfg = OptimizerFromFactoryConfig(factory=factory, kwargs={"lr": 1e-3})
        model = _TinyModel()

        cfg.build(model)

        _assert_same_parameters(_RecordingFusedAdam.last_params, list(model.parameters()))
        assert _RecordingFusedAdam.last_kwargs["bias_correction"] is True
        assert _native_columns_after_patch(stub_te_modules, native) == [1, 1, 1, 1]

    def test_te_fused_adam_factory_all_params_empty_are_kept(self, stub_te_fused_adam):
        cfg = OptimizerFromFactoryConfig(factory=stub_te_fused_adam, kwargs={"lr": 1e-3})
        model = _OnlyEmptyTrainableModel()

        cfg.build(model)

        _assert_same_parameters(stub_te_fused_adam.last_params, [model.empty])

    def test_te_fused_adam_factory_keeps_empty_param_groups(self, stub_te_modules):
        native = stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier
        cfg = OptimizerFromFactoryConfig(factory=_RecordingFusedAdam, kwargs={"lr": 1e-3})
        empty = nn.Parameter(torch.empty(0))
        kept = nn.Parameter(torch.ones(3))

        cfg.build_from_param_groups(
            [
                {"params": [empty, kept], "weight_decay": 0.1},
            ]
        )

        (group,) = _RecordingFusedAdam.last_params
        _assert_same_parameters(group["params"], [empty, kept])
        assert group["weight_decay"] == 0.1
        assert _native_columns_after_patch(stub_te_modules, native) == [1, 1, 1, 1]

    def test_non_te_factory_keeps_every_parameter(self):
        # Other factories are handed every parameter as well, the empty one included.
        cfg = OptimizerFromFactoryConfig(factory=torch.optim.SGD, kwargs={"lr": 0.01})

        opt = cfg.build(_TinyModel())[0]

        assert len(opt.param_groups[0]["params"]) == 3  # linear weight + bias + empty

    def test_non_te_factory_does_not_patch_te(self, stub_te_modules):
        native = stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier
        cfg = OptimizerFromFactoryConfig(factory=torch.optim.SGD, kwargs={"lr": 0.01})

        cfg.build(_TinyModel())

        assert stub_te_modules[_FUSED_ADAM_MODULE].multi_tensor_applier is native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_adam_cuda_step_keeps_locally_empty_parameter():
    """Real TE FusedAdam keeps a zero-numel parameter and steps the other parameters.

    The zero-numel parameter keeps its position in the parameter list and in
    ``state_dict()``; the nonempty parameter is updated.
    """
    fused_adam = pytest.importorskip("transformer_engine.pytorch.optimizers.fused_adam")

    device = torch.device("cuda", 0)
    # bf16 params with fp32 master weights: the configuration wide-mesh runs use.
    empty = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16))
    kept = nn.Parameter(torch.tensor([1.0, -2.0, 3.0], device=device, dtype=torch.bfloat16))
    empty.grad = torch.empty_like(empty)
    kept.grad = torch.tensor([0.5, -0.25, 1.0], device=device, dtype=torch.bfloat16)
    before = kept.detach().clone()

    original_applier = fused_adam.multi_tensor_applier
    original_apply_scale = fused_adam.FusedAdam._apply_scale
    try:
        opt = FusedAdamConfig(lr=1e-2, master_weight_dtype="torch.float32")._build_optimizer([empty, kept])
        assert len(opt.param_groups) == 1
        _assert_same_parameters(opt.param_groups[0]["params"], [empty, kept])

        opt.step()
        torch.cuda.synchronize(device)

        assert torch.isfinite(kept).all()
        assert not torch.equal(kept, before)
        assert opt.param_groups[0]["step"] == 1
        assert all(value.numel() == 0 for value in opt.state[empty].values())
        state_dict = opt.state_dict()
        assert state_dict["param_groups"][0]["params"] == [0, 1]
        assert sorted(state_dict["state"]) == [0, 1]
        assert all(value.numel() == 3 for value in state_dict["state"][1].values())
    finally:
        fused_adam.multi_tensor_applier = original_applier
        fused_adam.FusedAdam._apply_scale = original_apply_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_adam_cuda_fp16_state_keeps_locally_empty_parameter():
    """Real TE FusedAdam with fp16 moments steps and reloads with a zero-numel parameter present.

    FusedAdam rescales fp16 moments after every step and in ``load_state_dict``
    by the largest magnitude of the state, which has no value for a
    zero-numel state.
    """
    fused_adam = pytest.importorskip("transformer_engine.pytorch.optimizers.fused_adam")

    device = torch.device("cuda", 0)
    values = torch.tensor([1.0, -2.0, 3.0], device=device, dtype=torch.bfloat16)
    gradient = torch.tensor([0.5, -0.25, 1.0], device=device, dtype=torch.bfloat16)
    original_applier = fused_adam.multi_tensor_applier
    original_apply_scale = fused_adam.FusedAdam._apply_scale
    try:

        def build() -> tuple[torch.optim.Optimizer, nn.Module]:
            model = nn.Module()
            model.empty = nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16))
            model.kept = nn.Parameter(values.clone())
            config = OptimizerFromFactoryConfig(
                factory=fused_adam.FusedAdam,
                kwargs={
                    "lr": 1e-2,
                    "master_weights": True,
                    "exp_avg_dtype": "torch.float16",
                    "exp_avg_sq_dtype": "torch.float16",
                },
            )
            return config.build(model)[0], model

        def step(opt: torch.optim.Optimizer, model: nn.Module) -> None:
            model.empty.grad = torch.empty_like(model.empty)
            model.kept.grad = gradient.clone()
            opt.step()
            torch.cuda.synchronize(device)

        opt, model = build()
        _assert_same_parameters(opt.param_groups[0]["params"], [model.empty, model.kept])
        step(opt, model)
        assert torch.isfinite(model.kept).all()
        assert not torch.equal(model.kept, values)
        assert opt.state[model.kept]["exp_avg"].dtype == torch.float16

        resumed, resumed_model = build()
        with torch.no_grad():
            resumed_model.kept.copy_(model.kept)
        resumed.load_state_dict(opt.state_dict())
        step(opt, model)
        step(resumed, resumed_model)
        # A reload rescales the fp16 moments once more, so allow one bf16 step of the parameter.
        torch.testing.assert_close(resumed_model.kept.float(), model.kept.float(), rtol=1e-2, atol=0)
        assert resumed.param_groups[0]["step"] == opt.param_groups[0]["step"] == 2
        assert all(value.numel() == 0 for value in resumed.state[resumed_model.empty].values())
    finally:
        fused_adam.multi_tensor_applier = original_applier
        fused_adam.FusedAdam._apply_scale = original_apply_scale
