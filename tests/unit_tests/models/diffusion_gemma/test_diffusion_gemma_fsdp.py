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

"""Unit tests for ``diffusion_gemma`` pure-FSDP2 (ep_size=1) sharding.

These cover the lazy factory for ``DiffusionGemmaParallelizationStrategy`` and
the per-module wrapping order of ``fully_shard_diffusion_gemma`` (grouped
experts wrapped as their own FSDP unit before the rest of the layer). They use a
monkeypatched ``fully_shard`` so no process group / GPU is required.
"""

import torch.nn as nn

from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    ParallelizationStrategy,
    get_parallelization_strategy,
)
from nemo_automodel.components.models.diffusion_gemma import fsdp as dg4_fsdp
from nemo_automodel.components.models.diffusion_gemma import parallelizer as dg4_parallelizer


def test_strategy_factory_is_lazy_and_model_owned():
    strategy = dg4_parallelizer.get_parallelization_strategy()
    assert strategy is dg4_parallelizer.get_parallelization_strategy()
    assert isinstance(strategy, ParallelizationStrategy)
    assert isinstance(strategy, DefaultParallelizationStrategy)

    DiffusionGemmaForBlockDiffusion = type("DiffusionGemmaForBlockDiffusion", (nn.Module,), {})
    DiffusionGemmaForBlockDiffusion._nemo_parallelization_strategy_factory = staticmethod(
        dg4_parallelizer.get_parallelization_strategy
    )
    assert get_parallelization_strategy(DiffusionGemmaForBlockDiffusion()) is strategy


def test_fully_shard_wraps_experts_before_the_layer(monkeypatch):
    """A decoder layer's ``moe.experts`` are wrapped first, then the whole layer."""
    calls: list[nn.Module] = []

    def fake_fully_shard(module, **kwargs):
        calls.append(module)
        return module

    # The helper resolves fully_shard from its own module namespace.
    monkeypatch.setattr(dg4_fsdp, "fully_shard", fake_fully_shard)
    # No prior FSDP state on plain modules.
    monkeypatch.setattr(dg4_fsdp, "_has_fsdp_state", lambda m: False)

    experts = nn.Linear(4, 4)
    layer = nn.Module()
    layer.moe = nn.Module()
    layer.moe.experts = experts

    dg4_fsdp.fully_shard_diffusion_gemma(layer, mesh=None, mp_policy=None)

    assert calls == [experts, layer], "experts must be sharded as their own unit before the parent layer"


def test_fully_shard_module_without_moe_wraps_once(monkeypatch):
    """The root model (no ``moe``) is wrapped exactly once."""
    calls: list[nn.Module] = []
    monkeypatch.setattr(dg4_fsdp, "fully_shard", lambda module, **kwargs: calls.append(module) or module)
    monkeypatch.setattr(dg4_fsdp, "_has_fsdp_state", lambda m: False)

    root = nn.Linear(4, 4)
    dg4_fsdp.fully_shard_diffusion_gemma(root, mesh=None, mp_policy=None)

    assert calls == [root]


def test_fully_shard_once_skips_already_wrapped(monkeypatch):
    """Modules already carrying FSDP state are not re-wrapped."""
    monkeypatch.setattr(dg4_fsdp, "_has_fsdp_state", lambda m: True)
    sentinel_calls: list[nn.Module] = []
    monkeypatch.setattr(dg4_fsdp, "fully_shard", lambda module, **kwargs: sentinel_calls.append(module) or module)

    module = nn.Linear(4, 4)
    out = dg4_fsdp._fully_shard_once(module, mesh=None, mp_policy=None, offload_policy=None)

    assert out is module
    assert sentinel_calls == []
