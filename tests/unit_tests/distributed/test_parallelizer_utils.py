# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import types
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.distributed.parallelizer_utils import (
    iter_maximal_uniform_dtype_subtrees,
    _group_params_by_dtype,
    _get_module_from_path,
    _fully_shard,
    fully_shard_by_dtype,
)


class Block(nn.Module):
    def __init__(
        self,
        dtype_l1: torch.dtype = torch.float16,
        dtype_l2: torch.dtype = torch.float16,
        add_misleading_buffer: bool = False,
        buffer_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.l1 = nn.Linear(4, 4, bias=False).to(dtype_l1)
        self.l2 = nn.Linear(4, 4, bias=False).to(dtype_l2)
        if add_misleading_buffer:
            # Add a floating-point buffer that can break subtree uniformity when included
            self.register_buffer("buf", torch.zeros(1, dtype=buffer_dtype))


class ToyModel(nn.Module):
    def __init__(
        self,
        a_dtype: torch.dtype = torch.float32,
        b_dtype_l1: torch.dtype = torch.float16,
        b_dtype_l2: torch.dtype = torch.float16,
        c_dtype: torch.dtype | None = None,
        block_has_misleading_buffer: bool = False,
        block_buffer_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False).to(a_dtype)
        self.b = Block(
            dtype_l1=b_dtype_l1,
            dtype_l2=b_dtype_l2,
            add_misleading_buffer=block_has_misleading_buffer,
            buffer_dtype=block_buffer_dtype,
        )
        if c_dtype is not None:
            # Optional third distinct subtree for >2 dtype scenarios
            self.c = nn.Linear(4, 4, bias=False).to(c_dtype)


def _collect_return_paths_items(
    items: List[Tuple[str, nn.Module, torch.dtype]]
) -> dict[str, torch.dtype]:
    return {path: dtype for path, _mod, dtype in items}


def _collect_return_modules_items(
    items: List[Tuple[nn.Module, torch.dtype]]
) -> dict[int, torch.dtype]:
    return {id(mod): dtype for mod, dtype in items}


def test_iter_maximal_uniform_dtype_subtrees_basic_paths():
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
    )
    # return_paths=True
    items_with_paths = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=True, tensor_pred=torch.is_floating_point, return_paths=True
        )
    )
    paths_to_dtype = _collect_return_paths_items(items_with_paths)
    assert paths_to_dtype == {
        "a": torch.float32,
        "b": torch.float16,
    }

    # return_paths=False
    items_no_paths = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=True, tensor_pred=torch.is_floating_point, return_paths=False
        )
    )
    mods_to_dtype = _collect_return_modules_items(items_no_paths)
    expected = {id(model.a): torch.float32, id(model.b): torch.float16}
    assert mods_to_dtype == expected


def test_iter_maximal_uniform_dtype_subtrees_include_buffers_effect():
    # Block has a float32 buffer but float16 parameters; including buffers should break uniformity of 'b'
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
        block_has_misleading_buffer=True,
        block_buffer_dtype=torch.float32,
    )
    # include_buffers=True: expect 'a', 'b.l1', 'b.l2'
    items_with_buffers = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=True, tensor_pred=torch.is_floating_point, return_paths=True
        )
    )
    paths_to_dtype_with_buffers = _collect_return_paths_items(items_with_buffers)
    assert paths_to_dtype_with_buffers == {
        "a": torch.float32,
        "b.l1": torch.float16,
        "b.l2": torch.float16,
    }

    # include_buffers=False: buffer ignored, expect maximal 'b' again
    items_no_buffers = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=False, tensor_pred=torch.is_floating_point, return_paths=True
        )
    )
    paths_to_dtype_no_buffers = _collect_return_paths_items(items_no_buffers)
    assert paths_to_dtype_no_buffers == {
        "a": torch.float32,
        "b": torch.float16,
    }


def test_group_params_by_dtype_counts():
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
    )
    grouped = _group_params_by_dtype(model)
    # Expect 1 param tensor in float32 ('a.weight'), 2 param tensors in float16 ('b.l1.weight', 'b.l2.weight')
    assert set(grouped.keys()) == {torch.float32, torch.float16}
    assert len(grouped[torch.float32]) == 1
    assert len(grouped[torch.float16]) == 2


def test_get_module_from_path():
    model = ToyModel()
    mod = _get_module_from_path(model, "b.l1")
    assert mod is model.b.l1
    mod2 = _get_module_from_path(model, "b.l2")
    assert mod2 is model.b.l2


def test__fully_shard_calls_for_single_module(monkeypatch):
    calls: list[tuple[nn.Module, object, object, object]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy):
        calls.append((mod, mesh, mp_policy, offload_policy))

    # Monkeypatch the symbol inside the utils module
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    mod = nn.Linear(2, 2, bias=False)
    mesh, mp_policy, offload_policy = object(), object(), object()
    _fully_shard(mod, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)

    assert len(calls) == 1
    called_mod, called_mesh, called_mp, called_offload = calls[0]
    assert called_mod is mod
    assert called_mesh is mesh and called_mp is mp_policy and called_offload is offload_policy


def test__fully_shard_calls_for_modulelist(monkeypatch):
    calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy):
        calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )

    ml = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)])
    mesh, mp_policy, offload_policy = object(), object(), object()
    _fully_shard(ml, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)

    # Should call for each child, not the ModuleList itself
    assert len(calls) == 2
    assert calls[0] is ml[0]
    assert calls[1] is ml[1]


def test_fully_shard_by_dtype_no_params(monkeypatch):
    fully_calls: list[nn.Module] = []
    sub_calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy):
        fully_calls.append(mod)

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy):
        sub_calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    model = nn.Identity()
    fully_shard_by_dtype(model, mesh=object(), mp_policy=object(), offload_policy=object())
    assert fully_calls == []
    assert sub_calls == []


def test_combined_projection_adapter_dtensor_split_fallback_allgather_for_1d_shard0(monkeypatch):
    """
    Cover the 1D Shard(0) fallback path in _dtensor_aware_split:
    - tensor.split(...) raises
    - we redistribute to Replicate, split locally, then DTensor.from_local + redistribute back to Shard(0)
    """
    import nemo_automodel.components.models.common.combined_projection.state_dict_adapter as mod

    # Minimal placement types for isinstance checks.
    class Shard:
        def __init__(self, dim: int):
            self.dim = dim

    class Replicate:
        pass

    monkeypatch.setattr(mod, "Shard", Shard, raising=True)
    monkeypatch.setattr(mod, "Replicate", Replicate, raising=True)

    calls = {"redistribute": [], "from_local": []}

    class DummyDeviceMesh:
        pass

    mesh = DummyDeviceMesh()

    class DummyDTensor:
        def __init__(self, local, placements):
            self._local = local
            self.placements = placements
            self.device_mesh = mesh
            self.ndim = local.ndim

        @property
        def shape(self):
            return self._local.shape

        def to_local(self):
            return self._local

        def split(self, *_args, **_kwargs):
            raise RuntimeError("force fallback")

        def redistribute(self, *, device_mesh, placements):
            calls["redistribute"].append((device_mesh, placements))
            assert device_mesh is mesh
            # For the test we don't need real sharding; just propagate placements.
            return DummyDTensor(self._local, (placements[0],))

        @classmethod
        def from_local(cls, local_tensor, *, device_mesh, placements, run_check=False):
            calls["from_local"].append((tuple(local_tensor.shape), device_mesh, placements, run_check))
            return cls(local_tensor, placements)

    # Make isinstance(x, DTensor) true inside the module.
    monkeypatch.setattr(mod, "DTensor", DummyDTensor, raising=True)

    full = mod.torch.arange(6, dtype=mod.torch.float32)
    dt = DummyDTensor(full, (Shard(0),))

    out = mod._dtensor_aware_split(dt, [2, 2, 2], dim=0)
    assert len(out) == 3
    assert [tuple(x.shape) for x in out] == [(2,), (2,), (2,)]
    assert any(isinstance(pl[0], Replicate) for _mesh, pl in calls["redistribute"])
    assert len(calls["from_local"]) == 3


def test_combined_projection_adapter_dtensor_cat_fallback_replicate_local_cat(monkeypatch):
    """
    Cover the Replicate fallback in _dtensor_aware_cat:
    - torch.cat(DTensor, ...) raises
    - placements == (Replicate(),) => cat local tensors and DTensor.from_local
    """
    import nemo_automodel.components.models.common.combined_projection.state_dict_adapter as mod

    class Replicate:
        pass

    monkeypatch.setattr(mod, "Replicate", Replicate, raising=True)

    real_cat = mod.torch.cat
    calls = {"from_local": []}

    class DummyDeviceMesh:
        pass

    mesh = DummyDeviceMesh()

    class DummyDTensor:
        def __init__(self, local):
            self._local = local
            self.ndim = local.ndim
            self.device_mesh = mesh
            self.placements = (Replicate(),)

        def to_local(self):
            return self._local

        @classmethod
        def from_local(cls, local_tensor, *, device_mesh, placements, run_check=False):
            calls["from_local"].append((tuple(local_tensor.shape), device_mesh, placements, run_check))
            inst = cls(local_tensor)
            inst.device_mesh = device_mesh
            inst.placements = placements
            return inst

    monkeypatch.setattr(mod, "DTensor", DummyDTensor, raising=True)

    def cat_maybe_raise(tensors, dim=0):
        if tensors and isinstance(tensors[0], DummyDTensor):
            raise RuntimeError("force dtensor cat failure")
        return real_cat(tensors, dim=dim)

    monkeypatch.setattr(mod.torch, "cat", cat_maybe_raise, raising=True)

    a = DummyDTensor(mod.torch.ones((2, 3)))
    b = DummyDTensor(mod.torch.zeros((1, 3)))
    out = mod._dtensor_aware_cat([a, b], dim=0)

    assert isinstance(out, DummyDTensor)
    assert out.to_local().shape == (3, 3)
    assert len(calls["from_local"]) == 1


def test_combined_projection_adapter_gate_up_bias_paths_tensor_only():
    """
    Cover the gate_up_proj.bias concat path (from_hf) and split path (to_hf) using regular tensors.
    This exercises the renamed helper calls in those code paths.
    """
    import types
    import nemo_automodel.components.models.common.combined_projection.state_dict_adapter as mod

    cfg = types.SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=4,
        head_dim=2,
        tie_word_embeddings=False,
    )
    adapter = mod.CombinedProjectionStateDictAdapter(cfg)

    hf_sd = {
        "model.layers.0.mlp.gate_proj.weight": mod.torch.ones((3, 4)),
        "model.layers.0.mlp.up_proj.weight": 2 * mod.torch.ones((3, 4)),
        "model.layers.0.mlp.gate_proj.bias": mod.torch.arange(3, dtype=mod.torch.float32),
        "model.layers.0.mlp.up_proj.bias": mod.torch.arange(3, 6, dtype=mod.torch.float32),
    }
    custom_sd = adapter.from_hf(hf_sd)
    assert "model.layers.0.mlp.gate_up_proj.bias" in custom_sd
    assert tuple(custom_sd["model.layers.0.mlp.gate_up_proj.bias"].shape) == (6,)

    hf_sd2 = adapter.to_hf(custom_sd)
    assert mod.torch.equal(hf_sd2["model.layers.0.mlp.gate_proj.bias"], hf_sd["model.layers.0.mlp.gate_proj.bias"])
    assert mod.torch.equal(hf_sd2["model.layers.0.mlp.up_proj.bias"], hf_sd["model.layers.0.mlp.up_proj.bias"])


def test_fully_shard_by_dtype_single_dtype(monkeypatch):
    fully_calls: list[nn.Module] = []
    sub_calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy):
        fully_calls.append(mod)

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy):
        sub_calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # All parameters are float32
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=object(), offload_policy=object())

    assert fully_calls == [model]  # whole module sharded once
    assert sub_calls == []  # no subtree calls


def test_fully_shard_by_dtype_two_dtypes(monkeypatch):
    fully_calls: list[nn.Module] = []
    sub_calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy):
        fully_calls.append(mod)

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy):
        sub_calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Make float32 the least common (1 param) vs float16 (2 params)
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float16, b_dtype_l2=torch.float16)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=object(), offload_policy=object())

    # Expect subtree sharding for the least common dtype subtree(s) and full sharding once
    assert fully_calls == [model]
    # The least common dtype is float32 ('a'), so only 'a' subtree should be sharded individually
    assert sub_calls == [model.a]


def test_fully_shard_by_dtype_three_dtypes(monkeypatch):
    fully_calls: list[nn.Module] = []
    sub_calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy):
        fully_calls.append(mod)

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy):
        sub_calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Distinct dtypes across three subtrees: a=float32, b=float16, c=bfloat16
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
        c_dtype=torch.bfloat16,
    )
    fully_shard_by_dtype(model, mesh=object(), mp_policy=object(), offload_policy=object())

    # For >2 dtypes: only subtree sharding, no whole-module sharding
    assert fully_calls == []
    # Expect all three subtrees to be individually sharded
    # Note: the 'b' subtree should be sharded as a whole since it is uniform float16
    assert set(sub_calls) == {model.a, model.b, model.c}


