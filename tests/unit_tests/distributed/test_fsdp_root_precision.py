# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Root FSDP units must preserve compute contracts after child wrapping."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.moe import parallelizer as moe_parallelizer


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.register_buffer("float_buffer", torch.ones(4, dtype=torch.float32))


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([_Block()])
        self.norm = nn.LayerNorm(4, dtype=torch.float32)


class _Outer(nn.Module):
    def __init__(self, contract):
        super().__init__()
        self.model = _Inner()
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self._keep_in_fp32_modules_strict = ["model.norm"] if contract == "strict" else []
        if contract == "hf":
            for parameter in self.model.norm.parameters():
                parameter._hf_compute_dtype = torch.float32


def _record_sharding(monkeypatch):
    calls = []

    def shard(module, *, mesh, mp_policy, ignored_params=None, **kwargs):
        assert not isinstance(module, FSDPModule), "An existing FSDP unit was wrapped twice"
        owned = set(module.parameters()) - set(ignored_params or ())
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                owned.difference_update(child.parameters())
        calls.append((module, owned, mp_policy, kwargs))
        module.__class__ = type(f"TestFSDP{type(module).__name__}", (FSDPModule, type(module)), {})
        return module

    monkeypatch.setattr(parallelizer_utils, "fully_shard", shard)
    monkeypatch.setattr(moe_parallelizer, "fully_shard", shard)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_uniform_reduce_dtype", lambda: None)
    return calls


@pytest.mark.parametrize("contract", ["strict", "hf", "master"])
@pytest.mark.parametrize("output_dtype", [None, torch.bfloat16])
def test_moe_root_preserves_final_parameter_compute_contract(monkeypatch, contract, output_dtype):
    model = _Outer(contract)
    final_parameters = set(model.model.norm.parameters())
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=output_dtype,
        cast_forward_inputs=False,
    )
    calls = _record_sharding(monkeypatch)
    moe_parallelizer.apply_fsdp(
        model,
        fsdp_mesh=object(),
        ep_enabled=False,
        ep_shard_enabled=False,
        mp_policy=policy,
        reshard_after_forward=True,
    )
    owners = [(module, mp) for module, params, mp, _ in calls if params & final_parameters]
    assert len(owners) == 1
    owner, final_policy = owners[0]
    expected = torch.bfloat16 if contract == "master" else torch.float32
    assert final_policy.param_dtype == expected
    assert final_policy.output_dtype == output_dtype
    assert final_policy.cast_forward_inputs is False
    assert len([module for module, _, _, _ in calls if module is model.model.layers[0]]) == 1
    assert len([module for module, _, _, _ in calls if module is model]) == 1


@pytest.mark.parametrize("use_external_parameter", [False, True])
def test_root_mixed_compute_excludes_owned_children_and_their_buffers(monkeypatch, use_external_parameter):
    root = nn.Module()
    root.old_block = _Block()
    root.norm = nn.Linear(4, 4, bias=False, dtype=torch.float32)
    root.head = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    excluded = set()
    if use_external_parameter:
        root.external = nn.Parameter(torch.ones(4, dtype=torch.float32))
        excluded.add(root.external)
    norm_weight = root.norm.weight
    head_weight = root.head.weight
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, output_dtype=torch.float32)
    calls = _record_sharding(monkeypatch)
    parallelizer_utils.fully_shard(root.old_block, mesh=object(), mp_policy=policy)
    parallelizer_utils.fully_shard_root_by_dtype(
        root,
        root_model=root,
        mesh=object(),
        mp_policy=policy,
        offload_policy=None,
        fp32_compute_module_names=("norm",),
        ignored_params=excluded or None,
    )
    assert sum(module is root.old_block for module, _, _, _ in calls) == 1
    norm_call = next((module, mp) for module, params, mp, _ in calls if norm_weight in params)
    head_call = next((module, mp) for module, params, mp, _ in calls if head_weight in params)
    assert norm_call[1].param_dtype is torch.float32
    assert norm_call[1].cast_forward_inputs is False
    assert head_call[1].param_dtype is torch.bfloat16
    # Internal precision boundaries preserve natural module output dtypes.
    assert norm_call[1].output_dtype is (torch.float32 if norm_call[0] is root else None)
    root_call = next(mp for module, _, mp, _ in calls if module is root)
    assert root_call.output_dtype is torch.float32
    assert all(not (params & excluded) for _, params, _, _ in calls)


def test_parameterless_root_keeps_fsdp_hooks(monkeypatch):
    model = nn.Module()
    calls = _record_sharding(monkeypatch)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    result = parallelizer_utils.fully_shard_root_by_dtype(
        model,
        root_model=model,
        mesh=object(),
        mp_policy=policy,
        offload_policy=None,
    )
    assert result is model
    assert len(calls) == 1
    assert calls[0][2] is policy
