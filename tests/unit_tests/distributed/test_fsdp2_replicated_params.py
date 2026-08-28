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

from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.distributed.fsdp2_extensions.replicated import (
    make_fully_shard_with_replicated_parameter_grad_sync,
    select_small_fp32_parameters,
)


class _SensitiveModel(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.bulk = nn.Linear(4, 4, bias=False, dtype=dtype)
        self._fp32_params = nn.Module()
        self._fp32_params.A_log = nn.Parameter(torch.ones(4, dtype=dtype))
        self._fp32_params.dt_bias = nn.Parameter(torch.ones(4, dtype=dtype))


def test_select_small_fp32_parameters_uses_per_managed_module_byte_limit():
    model = _SensitiveModel()

    selection = select_small_fp32_parameters(
        model,
        name_fragments=("_fp32_params",),
        max_bytes_per_module=32,
    )
    assert selection.parameters == (model._fp32_params.A_log, model._fp32_params.dt_bias)
    assert selection.replicated_bytes == 32
    assert selection.oversized_modules == ()

    selection = select_small_fp32_parameters(
        model,
        name_fragments=("_fp32_params",),
        max_bytes_per_module=31,
    )
    assert selection.parameters == ()
    assert selection.replicated_bytes == 0
    assert [(module.name, module.logical_bytes) for module in selection.oversized_modules] == [("_fp32_params", 32)]


def test_select_small_fp32_parameters_applies_limit_independently():
    class MultipleManagedModules(nn.Module):
        def __init__(self):
            super().__init__()
            self.small_fp32_params = nn.Linear(4, 2, bias=False, dtype=torch.float32)  # 32 bytes
            self.large_fp32_params = nn.Linear(5, 2, bias=False, dtype=torch.float32)  # 40 bytes

    model = MultipleManagedModules()
    selection = select_small_fp32_parameters(
        model,
        name_fragments=("_fp32_params",),
        max_bytes_per_module=32,
    )

    assert selection.parameters == (model.small_fp32_params.weight,)
    assert selection.replicated_bytes == 32
    assert [(module.name, module.logical_bytes) for module in selection.oversized_modules] == [
        ("large_fp32_params", 40)
    ]


def test_select_small_fp32_parameters_keeps_lower_precision_residency_sharded():
    model = _SensitiveModel(dtype=torch.bfloat16)
    selection = select_small_fp32_parameters(model, name_fragments=("_fp32_params",))

    assert selection.parameters == ()
    assert selection.oversized_modules == ()
    assert [(module.name, module.sharded_reason) for module in selection.modules] == [
        ("_fp32_params", "non_fp32_residency")
    ]


def test_replicated_grad_sync_reduces_across_both_hsdp_mesh_dimensions(monkeypatch):
    model = _SensitiveModel()
    parameters = (model._fp32_params.A_log, model._fp32_params.dt_bias)
    parameters[0].grad = torch.arange(4, dtype=torch.float32)
    parameters[1].grad = torch.arange(4, 8, dtype=torch.float32)
    original_grads = tuple(parameter.grad.clone() for parameter in parameters)
    replicate_group, shard_group = object(), object()
    mesh = SimpleNamespace(size=lambda: 4, get_all_groups=lambda: [replicate_group, shard_group])
    fsdp_param_group = SimpleNamespace(reduce_grads=True)
    fsdp_state = SimpleNamespace(
        _state_ctx=SimpleNamespace(all_states=[SimpleNamespace(_fsdp_param_group=fsdp_param_group)]),
        _root_post_backward_final_callback=lambda: None,
    )
    model._get_fsdp_state = lambda: fsdp_state
    calls = []

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_reduce(tensor, op, group):
        calls.append((tensor.dtype, tensor.numel(), op, group))
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    fully_shard_with_grad_sync = make_fully_shard_with_replicated_parameter_grad_sync(
        model,
        parameters,
        mesh,
        fully_shard_fn=lambda module, **kwargs: module,
    )
    fully_shard_with_grad_sync(model)
    fsdp_state._root_post_backward_final_callback()
    assert calls == [
        (torch.float32, 11, torch.distributed.ReduceOp.SUM, replicate_group),
        (torch.float32, 11, torch.distributed.ReduceOp.SUM, shard_group),
    ]
    for parameter, expected in zip(parameters, original_grads):
        torch.testing.assert_close(parameter.grad, expected)


def test_replicated_grad_sync_keeps_globally_unused_parameter_without_grad(monkeypatch):
    model = _SensitiveModel()
    parameters = (model._fp32_params.A_log, model._fp32_params.dt_bias)
    parameters[0].grad = torch.ones_like(parameters[0])
    mesh = SimpleNamespace(size=lambda: 2, get_all_groups=lambda: [object()])
    fsdp_state = SimpleNamespace(
        _state_ctx=SimpleNamespace(all_states=[SimpleNamespace(_fsdp_param_group=SimpleNamespace(reduce_grads=True))]),
        _root_post_backward_final_callback=lambda: None,
    )
    model._get_fsdp_state = lambda: fsdp_state
    fully_shard_with_grad_sync = make_fully_shard_with_replicated_parameter_grad_sync(
        model,
        parameters,
        mesh,
        fully_shard_fn=lambda module, **kwargs: module,
    )
    fully_shard_with_grad_sync(model)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op, group: tensor.mul_(2))
    fsdp_state._root_post_backward_final_callback()

    torch.testing.assert_close(parameters[0].grad, torch.ones_like(parameters[0]))
    assert parameters[1].grad is None


def test_replicated_grad_sync_zero_fills_rank_local_unused_parameter(monkeypatch):
    model = _SensitiveModel()
    parameters = (model._fp32_params.A_log, model._fp32_params.dt_bias)
    parameters[0].grad = torch.ones_like(parameters[0])
    group = object()
    mesh = SimpleNamespace(size=lambda: 2, get_all_groups=lambda: [group])
    fsdp_state = SimpleNamespace(
        _state_ctx=SimpleNamespace(all_states=[SimpleNamespace(_fsdp_param_group=SimpleNamespace(reduce_grads=True))]),
        _root_post_backward_final_callback=lambda: None,
    )
    model._get_fsdp_state = lambda: fsdp_state
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def add_peer_payload(tensor, op, group):
        """Add a synthetic peer's flat FP32 gradient payload in place."""
        del op
        assert group is expected_group
        # Header is [error, used(A_log), used(dt_bias)], followed by both grads.
        peer = torch.zeros_like(tensor)
        peer[2] = 1
        peer[7:11] = 2
        tensor.add_(peer)

    expected_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", add_peer_payload)
    fully_shard_with_grad_sync = make_fully_shard_with_replicated_parameter_grad_sync(
        model,
        parameters,
        mesh,
        fully_shard_fn=lambda module, **kwargs: module,
    )
    fully_shard_with_grad_sync(model)
    fsdp_state._root_post_backward_final_callback()

    torch.testing.assert_close(parameters[0].grad, torch.full_like(parameters[0], 0.5))
    torch.testing.assert_close(parameters[1].grad, torch.ones_like(parameters[1]))


def test_fully_shard_wrapper_syncs_from_fsdp_post_backward_lifecycle(monkeypatch):
    model = _SensitiveModel()
    parameters = (model._fp32_params.A_log, model._fp32_params.dt_bias)
    parameters[0].grad = torch.arange(4, dtype=torch.float32)
    parameters[1].grad = torch.arange(4, 8, dtype=torch.float32)
    original_grads = tuple(parameter.grad.clone() for parameter in parameters)
    group = object()
    mesh = SimpleNamespace(size=lambda: 2, get_all_groups=lambda: [group])
    fsdp_param_group = SimpleNamespace(reduce_grads=True)
    callback_calls = []
    fsdp_state = SimpleNamespace(
        _state_ctx=SimpleNamespace(all_states=[SimpleNamespace(_fsdp_param_group=fsdp_param_group)]),
        _root_post_backward_final_callback=lambda: callback_calls.append("fsdp"),
    )
    model._get_fsdp_state = lambda: fsdp_state
    collective_calls = []

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_reduce(tensor, op, group):
        collective_calls.append((tensor.dtype, tensor.numel(), op, group))
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    fully_shard_calls = []

    def fake_fully_shard(module, **kwargs):
        fully_shard_calls.append((module, kwargs))
        return module

    fully_shard_with_grad_sync = make_fully_shard_with_replicated_parameter_grad_sync(
        model,
        parameters,
        mesh,
        fully_shard_fn=fake_fully_shard,
    )
    assert fully_shard_with_grad_sync(model, marker=True) is model

    fsdp_state._root_post_backward_final_callback()

    assert fully_shard_calls == [(model, {"marker": True})]
    assert callback_calls == ["fsdp"]
    assert collective_calls == [(torch.float32, 11, torch.distributed.ReduceOp.SUM, group)]
    for parameter, expected in zip(parameters, original_grads):
        torch.testing.assert_close(parameter.grad, expected)


def test_fully_shard_wrapper_defers_with_fsdp_gradient_sync_state(monkeypatch):
    model = _SensitiveModel()
    parameters = (model._fp32_params.A_log, model._fp32_params.dt_bias)
    parameters[0].grad = torch.ones_like(parameters[0])
    parameters[1].grad = torch.ones_like(parameters[1])
    group = object()
    mesh = SimpleNamespace(size=lambda: 2, get_all_groups=lambda: [group])
    fsdp_param_group = SimpleNamespace(reduce_grads=False)
    fsdp_state = SimpleNamespace(
        _state_ctx=SimpleNamespace(all_states=[SimpleNamespace(_fsdp_param_group=fsdp_param_group)]),
        _root_post_backward_final_callback=lambda: None,
    )
    model._get_fsdp_state = lambda: fsdp_state
    collective_calls = []

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op, group: collective_calls.append(group),
    )
    fully_shard_with_grad_sync = make_fully_shard_with_replicated_parameter_grad_sync(
        model,
        parameters,
        mesh,
        fully_shard_fn=lambda module, **kwargs: module,
    )
    fully_shard_with_grad_sync(model)

    fsdp_state._root_post_backward_final_callback()
    assert collective_calls == []

    fsdp_param_group.reduce_grads = True
    fsdp_state._root_post_backward_final_callback()
    assert collective_calls == [group]
