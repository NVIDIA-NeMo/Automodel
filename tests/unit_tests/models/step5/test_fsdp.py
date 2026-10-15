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

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.models.step5 import fsdp as step5_fsdp
from nemo_automodel.components.models.step5.model import Step5ForCausalLM
from nemo_automodel.components.moe import parallelizer as moe_parallelizer
from tests.unit_tests.models.step5.test_model import make_model


def _record_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[nn.Module, set[nn.Parameter], MixedPrecisionPolicy]]:
    """Replace fully_shard with a recorder of each unit's newly owned parameters."""
    calls = []

    def shard(module, *, mesh, mp_policy, ignored_params=None, **kwargs):
        assert not isinstance(module, FSDPModule), "An existing FSDP unit was wrapped twice"
        owned = set(module.parameters()) - set(ignored_params or ())
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                owned.difference_update(child.parameters())
        calls.append((module, owned, mp_policy))
        module.__class__ = type(f"TestFSDP{type(module).__name__}", (FSDPModule, type(module)), {})
        return module

    for target in (parallelizer_utils, moe_parallelizer, step5_fsdp):
        monkeypatch.setattr(target, "fully_shard", shard)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_uniform_reduce_dtype", lambda: None)
    # The recorder has no real mesh; keep router correction-bias buffers local.
    monkeypatch.setattr(moe_parallelizer, "distribute_tensor", lambda tensor, **kwargs: tensor)
    return calls


def test_step5_uses_model_owned_parallelizer():
    assert Step5ForCausalLM.parallelizer is step5_fsdp.PARALLELIZER
    assert get_model_parallelizer(make_model()) is step5_fsdp.PARALLELIZER


@pytest.mark.parametrize("model_owned", [False, True])
def test_final_norm_computes_in_fp32_only_with_step5_parallelizer(monkeypatch, model_owned):
    model = make_model()
    norm_weight = model.model.norm.weight
    assert norm_weight.dtype == torch.float32
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    calls = _record_sharding(monkeypatch)
    moe_parallelizer.apply_fsdp(
        model,
        fsdp_mesh=SimpleNamespace(ndim=1),
        ep_enabled=False,
        ep_shard_enabled=False,
        mp_policy=policy,
        reshard_after_forward=True,
        model_parallelizer=step5_fsdp.PARALLELIZER if model_owned else None,
    )
    owners = [(module, mp) for module, params, mp in calls if norm_weight in params]
    assert len(owners) == 1
    owner, owner_policy = owners[0]
    if model_owned:
        assert owner is model.model.norm
        assert owner_policy.param_dtype is torch.float32 and owner_policy.cast_forward_inputs is False
    else:
        # Upstream's root unit computes the strict-FP32 final norm in BF16.
        assert owner is model.model and owner_policy.param_dtype is torch.bfloat16
    assert sum(module is model.model for module, _, _ in calls) == 1


def test_pipeline_stage_without_final_norm_wraps_root_once(monkeypatch):
    model = make_model()
    model.model.norm = None
    calls = _record_sharding(monkeypatch)
    step5_fsdp.PARALLELIZER._fully_shard_module(
        model.model, mesh=object(), mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    )
    assert [module for module, _, _ in calls] == [model.model]


def test_outer_root_wraps_contained_final_norm_once(monkeypatch):
    model = make_model()
    calls = _record_sharding(monkeypatch)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    step5_fsdp.PARALLELIZER._fully_shard_module(model, mesh=object(), mp_policy=policy)
    assert [module for module, _, _ in calls] == [model.model.norm, model]
    assert calls[0][2].param_dtype is torch.float32
    assert model.model.norm.weight not in calls[1][1]
