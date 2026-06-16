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

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.parallelizer_utils import (
    _fully_shard,
    _get_module_from_path,
    _group_params_by_dtype,
    _make_compute_dtype_fn,
    _mp_policy_with_param_dtype,
    fully_shard_by_dtype,
    iter_maximal_uniform_dtype_subtrees,
)


def _tag_hf_compute_dtype(model: nn.Module) -> None:
    """Simulate an HF checkpoint load by recording each float tensor's dtype.

    ``_restore_loaded_model_dtype`` does this in production; tagging here lets the
    compute-dtype grouping mirror storage dtype (as it would for a loaded model).
    """
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.dtype.is_floating_point:
            tensor._hf_compute_dtype = tensor.dtype


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


def _collect_return_paths_items(items: List[Tuple[str, nn.Module, torch.dtype]]) -> dict[str, torch.dtype]:
    return {path: dtype for path, _mod, dtype in items}


def _collect_return_modules_items(items: List[Tuple[nn.Module, torch.dtype]]) -> dict[int, torch.dtype]:
    return {id(mod): dtype for mod, dtype in items}


def _make_mp_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )


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

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
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

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
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


def test_mp_policy_with_param_dtype_copies_policy():
    mp_policy = _make_mp_policy()

    copied_policy = _mp_policy_with_param_dtype(mp_policy, torch.float32)

    assert copied_policy is not mp_policy
    assert copied_policy.param_dtype == torch.float32
    assert copied_policy.reduce_dtype == mp_policy.reduce_dtype
    assert copied_policy.output_dtype == mp_policy.output_dtype
    assert mp_policy.param_dtype == torch.bfloat16


def test_fully_shard_by_dtype_no_params(monkeypatch):
    fully_calls: list[nn.Module] = []
    sub_calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append(mod)

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
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
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy, reshard_after_forward))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # All parameters are float32 storage, but the policy requests bf16 compute
    # (fp32 master weights). Compute dtype is decoupled from storage: the bulk
    # computes in mp_policy.param_dtype (bf16), NOT the fp32 storage dtype.
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    mp_policy = _make_mp_policy()
    fully_shard_by_dtype(
        model,
        mesh=object(),
        mp_policy=mp_policy,
        offload_policy=object(),
        reshard_after_forward=False,
    )

    assert [mod for mod, _policy, _reshard in fully_calls] == [model]  # whole module sharded once
    assert fully_calls[0][1] is not mp_policy
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert fully_calls[0][1].reduce_dtype == mp_policy.reduce_dtype
    assert fully_calls[0][1].output_dtype == mp_policy.output_dtype
    assert fully_calls[0][2] is False
    assert mp_policy.param_dtype == torch.bfloat16
    assert sub_calls == []  # no fp32-compute carve-outs declared


def test_fully_shard_by_dtype_storage_equals_compute_keeps_storage_dtype(monkeypatch):
    """Uniform storage that matches the requested compute dtype shards as before."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )

    # bf16 storage and bf16 compute -> param_dtype stays bf16 (no decoupling needed).
    model = ToyModel(a_dtype=torch.bfloat16, b_dtype_l1=torch.bfloat16, b_dtype_l2=torch.bfloat16)
    mp_policy = _make_mp_policy()
    fully_shard_by_dtype(model, mesh=object(), mp_policy=mp_policy, offload_policy=object())

    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.bfloat16


def test_fully_shard_by_dtype_genuine_fp32_compute_unchanged(monkeypatch):
    """Uniform fp32 storage with an fp32-compute policy keeps fp32 compute."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )

    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=mp_policy, offload_policy=object())

    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.float32


def test_make_compute_dtype_fn_precedence():
    """Resolver precedence: pinned fp32 > HF-recorded > mp_policy.param_dtype."""

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(4, 4, bias=False).to(torch.float32)  # master-weight fp32 storage
            self.recorded_fp32 = nn.Linear(4, 4, bias=False).to(torch.float32)
            self.recorded_bf16 = nn.Linear(4, 4, bias=False).to(torch.float32)
            self._fp32_params = Holder()

    mixer = Mixer()
    # Simulate HF load: in_proj had bf16 in the checkpoint, the others fp32/bf16.
    mixer.in_proj.weight._hf_compute_dtype = torch.bfloat16
    mixer.recorded_fp32.weight._hf_compute_dtype = torch.float32
    mixer.recorded_bf16.weight._hf_compute_dtype = torch.bfloat16

    fn = _make_compute_dtype_fn(mixer, _make_mp_policy(), ("_fp32_params",))

    # Pinned wins even though storage is fp32 and nothing was recorded.
    assert fn(mixer._fp32_params.weight) == torch.float32
    # HF-recorded bf16 beats the fp32 master storage.
    assert fn(mixer.in_proj.weight) == torch.bfloat16
    assert fn(mixer.recorded_bf16.weight) == torch.bfloat16
    # HF-recorded fp32 is honored.
    assert fn(mixer.recorded_fp32.weight) == torch.float32


def test_make_compute_dtype_fn_fallback_to_policy_then_storage():
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)

    # No record, no pin, bf16 policy -> fall back to policy (bf16) despite fp32 storage.
    fn = _make_compute_dtype_fn(model, _make_mp_policy(), ())
    assert fn(model.a.weight) == torch.bfloat16

    # No policy -> fall back to storage dtype.
    fn_no_policy = _make_compute_dtype_fn(model, None, ())
    assert fn_no_policy(model.a.weight) == torch.float32


def test_fully_shard_by_dtype_fp32_master_pins_compute(monkeypatch):
    """fp32 master weights (uniform fp32 storage): pinned param keeps fp32, bulk gets bf16."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy, reshard_after_forward))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy, reshard_after_forward))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    class Fp32Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            # Two bulk linears so the bf16-compute group is the strict majority.
            self.in_proj = nn.Linear(4, 4, bias=False).to(torch.float32)
            self.out_proj = nn.Linear(4, 4, bias=False).to(torch.float32)
            self._fp32_params = Fp32Holder()

    mixer = Mixer()
    fully_shard_by_dtype(
        mixer,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        fp32_compute_module_names=("_fp32_params",),
        reshard_after_forward=True,
    )

    # Minority fp32 holder sharded on its own; the bf16 bulk is the parent unit.
    assert [mod for mod, _policy, _reshard in sub_calls] == [mixer._fp32_params]
    assert sub_calls[0][1].param_dtype == torch.float32
    assert sub_calls[0][2] is False
    assert [mod for mod, _policy, _reshard in fully_calls] == [mixer]
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert fully_calls[0][2] is True


def test_fully_shard_by_dtype_fp32_master_hf_recorded_compute(monkeypatch):
    """fp32 master weights with HF-recorded dtypes: recorded-fp32 param stays fp32, no pin needed."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy, reshard_after_forward))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy, reshard_after_forward))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Uniform fp32 storage (master weights), but the checkpoint recorded 'a' as fp32
    # and the rest as bf16 -> 'a' computes fp32 automatically, bulk computes bf16.
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    model.a.weight._hf_compute_dtype = torch.float32
    model.b.l1.weight._hf_compute_dtype = torch.bfloat16
    model.b.l2.weight._hf_compute_dtype = torch.bfloat16

    fully_shard_by_dtype(
        model,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        reshard_after_forward=True,
    )

    assert [mod for mod, _policy, _reshard in sub_calls] == [model.a]
    assert sub_calls[0][1].param_dtype == torch.float32
    assert [mod for mod, _policy, _reshard in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert sub_calls[0][2] is True
    assert fully_calls[0][2] is True


def test_fully_shard_by_dtype_two_dtypes(monkeypatch):
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Make float32 the least common (1 param) vs float16 (2 params)
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float16, b_dtype_l2=torch.float16)
    _tag_hf_compute_dtype(model)  # compute dtype mirrors storage (as for a loaded checkpoint)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=object())

    # Expect subtree sharding for the least common dtype subtree(s) and full sharding once
    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.float16
    # The least common dtype is float32 ('a'), so only 'a' subtree should be sharded individually
    assert [mod for mod, _ in sub_calls] == [model.a]
    assert sub_calls[0][1].param_dtype == torch.float32


def test_fully_shard_by_dtype_three_dtypes(monkeypatch):
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

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
    _tag_hf_compute_dtype(model)  # compute dtype mirrors storage (as for a loaded checkpoint)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=object())

    # For >2 dtypes: only subtree sharding, no whole-module sharding
    assert fully_calls == []
    # Expect all three subtrees to be individually sharded
    # Note: the 'b' subtree should be sharded as a whole since it is uniform float16
    assert {mod for mod, _ in sub_calls} == {model.a, model.b, model.c}
    assert {mod: policy.param_dtype for mod, policy in sub_calls} == {
        model.a: torch.float32,
        model.b: torch.float16,
        model.c: torch.bfloat16,
    }
