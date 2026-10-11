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

"""Real CPU FSDP ownership and budget controls for decoder weight retention."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.parallelizer import Step5Parallelizer, _retain_bf16_fsdp_units
from tests.unit_tests.models.step5.test_model import make_model


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("bf16_retention") / "rendezvous"
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    dist.destroy_process_group()


def _wrap(model, mesh):
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers.values():
        ignored = set(layer.moe.experts.parameters()) if layer.moe is not None else set()
        fully_shard_by_dtype(
            layer, mesh, policy, None, tuple(model._keep_in_fp32_modules_strict), True, ignored or None
        )
    for child in (model.model.embed_tokens, model.lm_head):
        fully_shard(child, mesh=mesh, reshard_after_forward=True, mp_policy=policy)


def _kept(module):
    return fully_shard.state(module)._fsdp_param_group.post_forward_mesh_info is None


def test_budget_is_typed_nonnegative_and_default_off():
    assert Step5BackendConfig().bf16_fsdp_retention_mib_per_part == 0
    enabled = Step5BackendConfig(bf16_fsdp_retention_mib_per_part=2048)
    assert replace(enabled, gate_precision=torch.float32).bf16_fsdp_retention_mib_per_part == 2048
    for invalid in (True, False, 1.0, "2048", None):
        with pytest.raises(TypeError, match="must be an int"):
            Step5BackendConfig(bf16_fsdp_retention_mib_per_part=invalid)
    with pytest.raises(ValueError, match="nonnegative"):
        Step5BackendConfig(bf16_fsdp_retention_mib_per_part=-1)


def test_owned_bf16_only_preserves_native_parameters_and_exclusions(mesh):
    model = make_model(dtype="bfloat16")
    _wrap(model, mesh)
    originals = dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    _retain_bf16_fsdp_units(model.model, budget_mib=1)
    units = [(name, module) for name, module in model.named_modules() if isinstance(module, FSDPModule)]
    changed = 0
    for name, module in units:
        if name in ("model.embed_tokens", "lm_head"):
            assert not _kept(module)
        elif fully_shard.state(module)._mp_policy.param_dtype == torch.bfloat16:
            assert _kept(module), name
            changed += 1
        else:
            assert not _kept(module), name
    assert changed == 10  # Three dense units plus seven CSA projection/shared units.
    for name, parameter in model.named_parameters():
        assert parameter is originals[name]
    assert optimizer_ids == [id(p) for group in optimizer.param_groups for p in group["params"]]
    expert = model.model.layers["3"].moe.experts
    assert not isinstance(expert, FSDPModule)
    assert all(not isinstance(p, torch.distributed.tensor.DTensor) for p in expert.parameters())


@pytest.mark.parametrize("budget,expected", [(0, False), (1, False), (8, True)])
def test_large_group_respects_total_part_budget(mesh, budget, expected):
    model = make_model(
        dtype="bfloat16", hidden_size=512, intermediate_size=1024, num_hidden_layers=1, moe_layer_list=[]
    )
    _wrap(model, mesh)
    _retain_bf16_fsdp_units(model.model, budget_mib=budget)
    assert _kept(model.model.layers["0"]) == expected
    assert not _kept(model.model.embed_tokens)
    assert not _kept(model.lm_head)


def test_ep_sharded_expert_unit_is_excluded_even_when_bf16(mesh):
    model = make_model(dtype="bfloat16")
    expert = model.model.layers["3"].moe.experts
    fully_shard(expert, mesh=mesh, reshard_after_forward=True)
    _wrap(model, mesh)
    _retain_bf16_fsdp_units(model.model, budget_mib=8)
    assert not _kept(expert)
    assert _kept(model.model.layers["3"].self_attn.qkv_proj)


def test_fp32_storage_computing_bf16_remains_sharded(mesh):
    model = make_model(dtype="float32", num_hidden_layers=1, moe_layer_list=[])
    _wrap(model, mesh)
    _retain_bf16_fsdp_units(model.model, budget_mib=8)
    assert not _kept(model.model.layers["0"])


@pytest.mark.parametrize("reverse_order", [False, True])
def test_part_budget_uses_stable_global_names_and_is_not_per_unit(mesh, reverse_order):
    model = make_model(dtype="bfloat16", hidden_size=256, intermediate_size=512, num_hidden_layers=3, moe_layer_list=[])
    if reverse_order:
        model.model.layers = torch.nn.ModuleDict(reversed(list(model.model.layers.items())))
    _wrap(model, mesh)
    _retain_bf16_fsdp_units(model.model, budget_mib=1)
    assert {name for name, layer in model.model.layers.items() if _kept(layer)} == {"0"}


@pytest.mark.parametrize("budget", [None, 0, 1])
def test_sidecar_applies_only_typed_opt_in(monkeypatch, mesh, budget):
    from nemo_automodel.components.distributed.parallelizer import ModelParallelizer

    model = make_model(dtype="bfloat16")
    if budget is not None:
        model.backend = Step5BackendConfig(bf16_fsdp_retention_mib_per_part=budget)
    _wrap(model, mesh)
    monkeypatch.setattr(ModelParallelizer, "parallelize", lambda self, value, context: value)
    assert Step5Parallelizer().parallelize(model, None) is model
    assert _kept(model.model.layers["0"]) == (budget == 1)


def test_existing_no_reshard_policy_cannot_silently_exceed_budget(mesh):
    model = make_model(dtype="bfloat16")
    _wrap(model, mesh)
    model.model.layers["0"].set_reshard_after_forward(False, recurse=False)
    with pytest.raises(ValueError, match="requires reshard_after_forward=True"):
        _retain_bf16_fsdp_units(model.model, budget_mib=1)
    assert not _kept(model.model.layers["1"])  # Validation precedes all changes.
    _retain_bf16_fsdp_units(model.model, budget_mib=0)
    assert _kept(model.model.layers["0"])
