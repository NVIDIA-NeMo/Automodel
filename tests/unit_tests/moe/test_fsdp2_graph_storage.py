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

from __future__ import annotations

from enum import Enum, auto
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

import nemo_automodel.components.moe.fsdp2_graph_storage as graph_storage
from nemo_automodel.components.moe.experts import GroupedExpertsTeOps
from nemo_automodel.components.moe.fsdp2_graph_storage import (
    FSDP2ExpertGraphStorage,
    FSDP2ExpertGraphStorageError,
)


class _ShardedState(Enum):
    SHARDED = auto()
    UNSHARDED = auto()


class _TrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()


class _FakeFSDPParam:
    def __init__(self, module: nn.Module, name: str, full_shape: tuple[int, ...]) -> None:
        self._module_info = SimpleNamespace(module=module, param_name=name)
        self.sharded_param = nn.Parameter(torch.zeros(full_shape[:-1] + (full_shape[-1] // 2,)))
        module.register_parameter(name, self.sharded_param)

        self.all_gather_outputs = [torch.empty(full_shape).view(-1)]
        self._unsharded_param = nn.Parameter(self.all_gather_outputs[0].view(full_shape))
        self._unsharded_inner_tensors: list[torch.Tensor] = []
        self.offload_to_cpu = False
        self.sharded_state = _ShardedState.SHARDED
        self.free_calls = 0
        self.free_unsharded_param()

    def all_gather(self, fill: float) -> None:
        output = self.all_gather_outputs[0]
        expected_bytes = output.numel() * output.element_size()
        if output.untyped_storage().size() != expected_bytes:
            output.untyped_storage().resize_(expected_bytes)
        output.fill_(fill)
        self._module_info.module._parameters[self._module_info.param_name] = self._unsharded_param
        self.sharded_state = _ShardedState.UNSHARDED

    def to_sharded(self) -> None:
        self._module_info.module._parameters[self._module_info.param_name] = self.sharded_param
        self.free_unsharded_param()
        self.sharded_state = _ShardedState.SHARDED

    def free_unsharded_param(self) -> None:
        self.free_calls += 1
        self.all_gather_outputs[0].untyped_storage().resize_(0)


class _FakeFSDPParamGroup:
    def __init__(self, fsdp_params: list[_FakeFSDPParam]) -> None:
        self.fsdp_params = fsdp_params
        self._sharded_state = _ShardedState.SHARDED
        self._training_state = _TrainingState.IDLE
        self._all_gather_result = None
        self.reshard_after_backward = True
        self._fill = 0

    def _to_unsharded(self) -> None:
        if self._sharded_state is _ShardedState.UNSHARDED:
            return
        self._fill += 1
        for fsdp_param in self.fsdp_params:
            fsdp_param.all_gather(float(self._fill))
        self._sharded_state = _ShardedState.UNSHARDED

    def _to_sharded(self) -> None:
        if self._sharded_state is _ShardedState.SHARDED:
            return
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_sharded()
        self._sharded_state = _ShardedState.SHARDED


def _make_experts() -> tuple[GroupedExpertsTeOps, _FakeFSDPParamGroup]:
    experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
    nn.Module.__init__(experts)
    experts.add_module("gate_up_linear", nn.Module())
    experts.add_module("down_linear", nn.Module())

    fsdp_params = [
        _FakeFSDPParam(experts.gate_up_linear, "_stacked_weight", (2, 8, 8)),
        _FakeFSDPParam(experts.gate_up_linear, "_stacked_bias", (2, 8)),
        _FakeFSDPParam(experts.down_linear, "_stacked_weight", (2, 8, 8)),
        _FakeFSDPParam(experts.down_linear, "_stacked_bias", (2, 8)),
    ]
    group = _FakeFSDPParamGroup(fsdp_params)

    def unshard(module: nn.Module, async_op: bool = False) -> None:
        assert module is experts
        assert not async_op
        group._to_unsharded()

    experts.unshard = MethodType(unshard, experts)
    return experts, group


def _install_fake_contract(
    monkeypatch: pytest.MonkeyPatch, group: _FakeFSDPParamGroup, *, compiled: bool = False
) -> None:
    contract = graph_storage._FSDP2Contract(
        fsdp_module_type=GroupedExpertsTeOps,
        fsdp_param_group_type=_FakeFSDPParamGroup,
        fsdp_param_type=_FakeFSDPParam,
        sharded_state_type=_ShardedState,
        compiled_autograd_enabled=lambda: compiled,
        state_for_module=lambda module: SimpleNamespace(_fsdp_param_group=group),
    )
    monkeypatch.setattr(graph_storage, "_load_fsdp2_contract", lambda: contract)


def test_retains_stable_storage_across_logical_reshard_and_future_all_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    experts, group = _make_experts()
    _install_fake_contract(monkeypatch, group)
    original_reshard_after_backward = group.reshard_after_backward

    handle = FSDP2ExpertGraphStorage(experts)
    handle.prepare_before_capture()
    expected_bytes = sum(
        param.all_gather_outputs[0].numel() * param.all_gather_outputs[0].element_size() for param in group.fsdp_params
    )
    captured_ptrs = [param._unsharded_param.data_ptr() for param in group.fsdp_params]
    assert handle.retained_bytes == expected_bytes
    assert handle.is_active

    handle.finish_capture()
    assert group._sharded_state is _ShardedState.SHARDED
    assert all(param.all_gather_outputs[0].untyped_storage().size() > 0 for param in group.fsdp_params)
    assert all(
        param._module_info.module._parameters[param._module_info.param_name] is param.sharded_param
        for param in group.fsdp_params
    )

    group._to_unsharded()
    group._to_sharded()
    handle.validate_stable()
    assert [param._unsharded_param.data_ptr() for param in group.fsdp_params] == captured_ptrs
    assert group.reshard_after_backward is original_reshard_after_backward

    handle.reset()
    assert handle.retained_bytes == 0
    assert not handle.is_active
    assert all(param.all_gather_outputs[0].untyped_storage().size() == 0 for param in group.fsdp_params)
    assert all("free_unsharded_param" not in vars(param) for param in group.fsdp_params)
    assert group.reshard_after_backward is original_reshard_after_backward


def test_prepare_rolls_back_when_tensor_extension_breaks_private_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    experts, group = _make_experts()
    group.fsdp_params[0]._unsharded_inner_tensors = [torch.empty(1)]
    _install_fake_contract(monkeypatch, group)
    handle = FSDP2ExpertGraphStorage(experts)

    with pytest.raises(FSDP2ExpertGraphStorageError, match="post-all-gather tensor extensions"):
        handle.prepare_before_capture()

    assert not handle.is_active
    assert group._sharded_state is _ShardedState.SHARDED
    assert all(param.all_gather_outputs[0].untyped_storage().size() == 0 for param in group.fsdp_params)
    assert all("free_unsharded_param" not in vars(param) for param in group.fsdp_params)


def test_prepare_accepts_existing_fsdp_pre_forward_unshard(monkeypatch: pytest.MonkeyPatch) -> None:
    experts, group = _make_experts()
    group._to_unsharded()
    _install_fake_contract(monkeypatch, group)
    captured_ptrs = [param._unsharded_param.data_ptr() for param in group.fsdp_params]

    handle = FSDP2ExpertGraphStorage(experts)
    handle.prepare_before_capture()
    assert [param._unsharded_param.data_ptr() for param in group.fsdp_params] == captured_ptrs

    handle.finish_capture()
    handle.reset()


def test_rejects_compiled_autograd_without_unsharding(monkeypatch: pytest.MonkeyPatch) -> None:
    experts, group = _make_experts()
    _install_fake_contract(monkeypatch, group, compiled=True)
    handle = FSDP2ExpertGraphStorage(experts)

    with pytest.raises(FSDP2ExpertGraphStorageError, match="only supports eager autograd"):
        handle.prepare_before_capture()

    assert group._sharded_state is _ShardedState.SHARDED
    assert all(param.all_gather_outputs[0].untyped_storage().size() == 0 for param in group.fsdp_params)


def test_reset_fails_closed_until_fsdp_step_is_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    experts, group = _make_experts()
    _install_fake_contract(monkeypatch, group)
    handle = FSDP2ExpertGraphStorage(experts)
    handle.prepare_before_capture()
    handle.finish_capture()
    retained_bytes = handle.retained_bytes

    group._training_state = _TrainingState.FORWARD
    with pytest.raises(FSDP2ExpertGraphStorageError, match="complete forward/backward step"):
        handle.reset()
    assert handle.retained_bytes == retained_bytes

    group._training_state = _TrainingState.IDLE
    handle.reset()
    assert handle.retained_bytes == 0


def test_rejects_non_te_ops_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _, group = _make_experts()
    _install_fake_contract(monkeypatch, group)

    with pytest.raises(FSDP2ExpertGraphStorageError, match="restricted to experts='te_ops'"):
        FSDP2ExpertGraphStorage(nn.Linear(2, 2))


@pytest.mark.parametrize("version", ("2.9.1", "2.11.0+cu130", "2.14.0a0+gitdeadbeef"))
def test_private_contract_rejects_unaudited_torch_minor(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(torch, "__version__", version)

    with pytest.raises(FSDP2ExpertGraphStorageError, match="audited PyTorch 2.10/2.12/2.13"):
        graph_storage._load_fsdp2_contract()


@pytest.mark.parametrize(
    "version",
    (
        "2.10.0+cu130",
        "2.12.0a0+0291f960b6.nv26.04.48445190",
        "2.13.0a0+8145d630e8.nv26.06",
    ),
)
def test_private_contract_loads_audited_torch_minor(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(torch, "__version__", version)

    contract = graph_storage._load_fsdp2_contract()

    assert callable(contract.state_for_module)
    assert callable(contract.compiled_autograd_enabled)
    assert callable(contract.fsdp_param_type.free_unsharded_param)
    assert callable(contract.fsdp_param_group_type._to_sharded)


@pytest.mark.parametrize(
    "version",
    ("2.12.0a0+0291f960b6.nv26.04.48445190", "2.13.0a0+8145d630e8.nv26.06"),
)
def test_torch_212_plus_contract_reads_compiled_autograd_state(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    import torch._dynamo.compiled_autograd as compiled_autograd

    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(compiled_autograd, "compiled_autograd_enabled", True)

    contract = graph_storage._load_fsdp2_contract()

    assert contract.compiled_autograd_enabled()


def test_torch_213_contract_rejects_changed_compiled_autograd_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._dynamo.compiled_autograd as compiled_autograd

    monkeypatch.setattr(torch, "__version__", "2.13.0a0+8145d630e8.nv26.06")
    monkeypatch.delattr(compiled_autograd, "in_compiled_autograd_region")

    with pytest.raises(FSDP2ExpertGraphStorageError, match="compiled-autograd contract changed"):
        graph_storage._load_fsdp2_contract()


def test_torch_213_contract_retains_structural_method_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    monkeypatch.setattr(torch, "__version__", "2.13.0a0+8145d630e8.nv26.06")
    monkeypatch.setattr(FSDPParamGroup, "_to_sharded", None)

    with pytest.raises(FSDP2ExpertGraphStorageError, match="required methods are missing or non-callable"):
        graph_storage._load_fsdp2_contract()
