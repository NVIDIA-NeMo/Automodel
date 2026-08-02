#!/usr/bin/env python3
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from nemo_automodel.components.distributed import megatron_fsdp as mfsdp
from nemo_automodel.components.distributed.config import MegatronFSDPConfig


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        return self.linear(hidden_states)


class _Model(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(4, 4)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply every block and the output head.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.head(hidden_states)


class _ModelWithProtectedFp32(_Model):
    _keep_in_fp32_modules = ["_fp32_params"]

    def __init__(self) -> None:
        super().__init__()
        self.layers[0]._fp32_params = nn.Linear(4, 4, dtype=torch.float32)


def _mesh(*, dp: int = 2, tp: int = 1, cp: int = 1):
    mesh = MagicMock()
    mesh.device_type = "cuda"
    axes = {
        "dp": MagicMock(size=MagicMock(return_value=dp), ndim=1),
        "tp": MagicMock(size=MagicMock(return_value=tp), ndim=1),
        "cp": MagicMock(size=MagicMock(return_value=cp), ndim=1),
    }
    mesh.__getitem__.side_effect = axes.__getitem__
    return mesh, axes


def test_parallelize_shards_blocks_bottom_up_then_root_with_flat_dp_placements(monkeypatch):
    monkeypatch.setattr(mfsdp.dist, "get_world_size", lambda: 2)
    shard_calls = []
    monkeypatch.setattr(mfsdp, "fully_shard", lambda module, **kwargs: shard_calls.append((module, kwargs)))
    optimizer_adapter = MagicMock()
    monkeypatch.setattr(mfsdp, "fully_shard_optimizer", optimizer_adapter)

    mesh, axes = _mesh()
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = mfsdp.MegatronFSDPManager(MegatronFSDPConfig(), mesh)

    result_model, result_optimizer = manager.parallelize(model, optimizer)

    assert result_model is model
    assert result_optimizer is optimizer
    assert [module for module, _ in shard_calls] == [model.layers[1], model.layers[0], model]
    for _, kwargs in shard_calls:
        assert kwargs["mesh"] is axes["dp"]
        placements = kwargs["placements"]
        assert placements.dp_axes == [0]
        assert isinstance(placements.parameter[0], mfsdp.Flat)
        assert isinstance(placements.gradient[0], mfsdp.Flat)
        assert isinstance(placements.optimizer[0], mfsdp.Flat)
        assert kwargs["mixed_precision_policy"] is manager.mp_policy
    optimizer_adapter.assert_called_once_with(optimizer)


def test_parallelize_preserves_protected_fp32_modules_in_fp32_main_storage(monkeypatch):
    monkeypatch.setattr(mfsdp.dist, "get_world_size", lambda: 2)
    shard_calls = []
    monkeypatch.setattr(mfsdp, "fully_shard", lambda module, **kwargs: shard_calls.append((module, kwargs)))

    mesh, _ = _mesh()
    model = _ModelWithProtectedFp32()
    manager = mfsdp.MegatronFSDPManager(MegatronFSDPConfig(), mesh)

    manager.parallelize(model)

    protected_module, protected_kwargs = shard_calls[0]
    assert protected_module is model.layers[0]._fp32_params
    assert protected_kwargs["mixed_precision_policy"].main_params_dtype is torch.float32
    assert [module for module, _ in shard_calls[1:]] == [model.layers[1], model.layers[0], model]
    assert all(kwargs["mixed_precision_policy"] is manager.mp_policy for _, kwargs in shard_calls[1:])


@pytest.mark.parametrize(
    ("preserve_fp32_weights", "grad_reduce_in_fp32", "main_params_dtype", "main_grads_dtype"),
    [
        (False, False, torch.bfloat16, None),
        (True, False, torch.float32, None),
        (False, True, torch.bfloat16, torch.float32),
        (True, True, torch.float32, torch.float32),
    ],
)
def test_precision_fields_map_to_v2_policy(
    preserve_fp32_weights,
    grad_reduce_in_fp32,
    main_params_dtype,
    main_grads_dtype,
):
    mesh, _ = _mesh()
    manager = mfsdp.MegatronFSDPManager(
        MegatronFSDPConfig(
            preserve_fp32_weights=preserve_fp32_weights,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
        ),
        mesh,
    )

    assert manager.mp_policy.main_params_dtype is main_params_dtype
    assert manager.mp_policy.main_grads_dtype is main_grads_dtype
    assert manager.mp_policy.grad_comm_dtype is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("init_fsdp_with_meta_device", True),
        ("overlap_grad_reduce", False),
        ("overlap_param_gather", False),
        ("check_for_nan_in_grad", False),
        ("report_nan_in_param_grad", True),
        ("average_in_collective", True),
        ("disable_bucketing", True),
        ("calculate_per_token_loss", True),
        ("keep_fp8_transpose_cache", True),
        ("nccl_ub", True),
        ("fsdp_double_buffer", True),
    ],
)
def test_unsupported_nondefault_legacy_options_are_rejected(field, value):
    mesh, _ = _mesh()
    with pytest.raises(ValueError, match=field):
        mfsdp.MegatronFSDPManager(MegatronFSDPConfig(**{field: value}), mesh)


@pytest.mark.parametrize(("axis", "feature"), [("tp", "tensor parallelism"), ("cp", "context parallelism")])
def test_parallelize_rejects_non_dp_topology(monkeypatch, axis, feature):
    monkeypatch.setattr(mfsdp.dist, "get_world_size", lambda: 2)
    mesh, _ = _mesh(**{axis: 2})
    manager = mfsdp.MegatronFSDPManager(MegatronFSDPConfig(), mesh)

    with pytest.raises(ValueError, match=feature):
        manager.parallelize(_Model())


def test_zero3_is_required():
    with pytest.raises(ValueError, match="zero_dp_strategy=3"):
        MegatronFSDPConfig(zero_dp_strategy=2)


def test_external_output_embedding_call_unshards_and_reshards_root():
    model = _Model()
    model.get_output_embeddings = lambda: model.head
    model._unshard_event = None
    model._lazy_init_context = MagicMock()
    model._unshard_parameter_groups = MagicMock(side_effect=lambda: setattr(model, "_unshard_event", object()))
    model._reshard_parameter_groups = MagicMock(side_effect=lambda: setattr(model, "_unshard_event", None))
    stream = MagicMock()
    model.context = SimpleNamespace(current_stream=MagicMock(return_value=stream))

    mfsdp.MegatronFSDPManager._register_external_output_embedding_hooks(model)
    model.head(torch.randn(2, 4))

    model._lazy_init_context.assert_called_once_with()
    model._unshard_parameter_groups.assert_called_once_with()
    stream.wait_event.assert_called_once()
    model._reshard_parameter_groups.assert_called_once_with()


def test_external_output_embedding_stays_unsharded_through_backward():
    model = _Model()
    model.get_output_embeddings = lambda: model.head
    model._unshard_event = None
    model._lazy_init_context = MagicMock()
    model._unshard_parameter_groups = MagicMock(side_effect=lambda: setattr(model, "_unshard_event", object()))
    model._reshard_parameter_groups = MagicMock(side_effect=lambda: setattr(model, "_unshard_event", None))
    model.context = SimpleNamespace(current_stream=MagicMock(return_value=MagicMock()))

    mfsdp.MegatronFSDPManager._register_external_output_embedding_hooks(model)
    hidden_states = torch.randn(2, 4, requires_grad=True)
    output = model.head(hidden_states)

    model._reshard_parameter_groups.assert_not_called()
    output.sum().backward()

    assert hidden_states.grad is not None
    model._reshard_parameter_groups.assert_called_once_with()


def test_structured_output_enters_root_backward_lifecycle():
    class StructuredModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, hidden_states):
            return {"logits": self.linear(hidden_states)}

    model = StructuredModel()
    model._unshard_event = None
    model._unshard_parameter_groups = MagicMock(side_effect=lambda: setattr(model, "_unshard_event", object()))
    model._reshard_parameter_groups = MagicMock(side_effect=lambda: setattr(model, "_unshard_event", None))
    model.pre_backward = MagicMock()
    model.post_backward = MagicMock()
    model.context = SimpleNamespace(current_stream=MagicMock(return_value=MagicMock()))

    mfsdp.MegatronFSDPManager._register_structured_output_hooks(model)
    output = model(torch.randn(2, 4, requires_grad=True))

    model._unshard_parameter_groups.assert_called_once_with()
    output["logits"].sum().backward()

    model.pre_backward.assert_called_once_with()
    model.post_backward.assert_called_once_with()
    model._reshard_parameter_groups.assert_called_once_with()


@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_mixed_unit_waits_for_module_and_parameter_backward_boundaries(input_requires_grad):
    class MixedModule(mfsdp.FsdpModule, nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.linear = nn.Linear(4, 4)

        def forward(self, hidden_states):
            return self.linear(hidden_states)

    module = MixedModule()
    module._parameter_groups = (
        SimpleNamespace(requires_grad=False),
        SimpleNamespace(requires_grad=True),
    )
    module._unshard_event = object()
    reshard = MagicMock(side_effect=lambda: setattr(module, "_unshard_event", None))
    module._reshard_parameter_groups = reshard
    module.post_backward = lambda: module._reshard_parameter_groups()

    mfsdp.MegatronFSDPManager._defer_reshard_until_module_backward(module)
    output = module(torch.randn(2, 4, requires_grad=input_requires_grad))
    module.post_backward()

    reshard.assert_not_called()
    output.sum().backward()

    reshard.assert_called_once_with()


def test_backward_prefetch_is_disabled_for_large_units():
    class FsdpLikeModule(mfsdp.FsdpModule, nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)

    module = FsdpLikeModule()
    next_module = object()
    order = SimpleNamespace(next_item=MagicMock(return_value=next_module))
    module._context = SimpleNamespace(backward_order=order)
    observed = []
    module.pre_backward = lambda: observed.append(order.next_item(module))
    release_retained = MagicMock()
    backward_state = {
        "retained_module": object(),
        "reshard": release_retained,
    }

    mfsdp.MegatronFSDPManager._disable_backward_prefetch(module, backward_state)
    module.pre_backward()

    release_retained.assert_called_once_with()
    assert backward_state == {"retained_module": None, "reshard": None}
    assert observed == [None]
    assert order.next_item(module) is next_module
