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
    _build_fsdp_shard_placement_fn_from_tp_plan,
    _group_params_by_dtype,
    _get_module_from_path,
    _fully_shard,
    fully_shard_by_dtype,
)
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.placement_types import Shard


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


class _TP_SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=True)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _TP_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(4, 8, bias=False)


class _TP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TP_SelfAttn()
        self.mlp = _TP_MLP()


class _TinyHFLikeModel(nn.Module):
    """Constructs parameter names like model.layers.0.self_attn.q_proj.weight."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, 4)
        self.model.layers = nn.ModuleList([_TP_Block()])
        self.lm_head = nn.Linear(4, 10, bias=False)
        # Scalar param to cover ndim==0 path
        self.scalar = nn.Parameter(torch.tensor(1.0))


def _get_param_by_name(model: nn.Module, name: str) -> nn.Parameter:
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise AssertionError(f"Parameter not found: {name}")


def test_build_fsdp_shard_placement_fn_from_tp_plan_colwise_and_special_cases():
    model = _TinyHFLikeModel()

    tp_plan = {
        # wildcard pattern (suffix should become ".self_attn.q_proj.weight")
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        # rowwise shouldn't be forced to dim=1
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        # concrete pattern (no '*')
        "model.layers.0.mlp.gate_up_proj": ColwiseParallel(),
        # lm_head special-case endswith("lm_head") => "lm_head.weight"
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    fn = _build_fsdp_shard_placement_fn_from_tp_plan(model, tp_plan)

    # Wildcard colwise -> Shard(1) for 2D weight
    q_w = _get_param_by_name(model, "model.layers.0.self_attn.q_proj.weight")
    assert fn(q_w).dim == 1

    # 1D params default to Shard(0)
    q_b = _get_param_by_name(model, "model.layers.0.self_attn.q_proj.bias")
    assert fn(q_b).dim == 0

    # Rowwise should not be forced
    o_w = _get_param_by_name(model, "model.layers.0.self_attn.o_proj.weight")
    assert fn(o_w).dim == 0

    # Concrete colwise -> Shard(1)
    gate_up_w = _get_param_by_name(model, "model.layers.0.mlp.gate_up_proj.weight")
    assert fn(gate_up_w).dim == 1

    # embed tokens always included
    embed_w = _get_param_by_name(model, "model.embed_tokens.weight")
    assert fn(embed_w).dim == 1

    # lm_head weight forced
    lm_head_w = _get_param_by_name(model, "lm_head.weight")
    assert fn(lm_head_w).dim == 1


def test_build_fsdp_shard_placement_fn_from_tp_plan_returns_none_for_0d():
    model = _TinyHFLikeModel()
    tp_plan = {"model.layers.*.self_attn.q_proj": ColwiseParallel()}
    fn = _build_fsdp_shard_placement_fn_from_tp_plan(model, tp_plan)

    scalar = _get_param_by_name(model, "scalar")
    assert scalar.ndim == 0
    assert fn(scalar) is None


