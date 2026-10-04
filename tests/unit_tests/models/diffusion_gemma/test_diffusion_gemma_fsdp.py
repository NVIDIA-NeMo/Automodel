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

"""Unit tests for ``diffusion_gemma`` pure-FSDP2 (ep_size=1) sharding.

``DiffusionGemmaModelParallelizer`` wraps a decoder layer's grouped experts as
their own FSDP unit before the layer itself; every other module is one unit.
``fully_shard`` is monkeypatched in the base parallelizer module, so no process
group or GPU is required. The installed ``transformers`` cannot build the real
decoder layer (its config lacks ``per_layer_config``), so the tests use a real
``DiffusionGemmaMoEDecoderLayer`` instance populated with small stand-in
submodules instead of running its config-driven ``__init__``.
"""

import types

import pytest
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

import nemo_automodel.components.distributed.parallelizer as parallelizer_mod
from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.diffusion_gemma import fsdp as dg4_fsdp
from nemo_automodel.components.models.diffusion_gemma.layers import DiffusionGemmaMoEDecoderLayer


def _decoder_layer() -> DiffusionGemmaMoEDecoderLayer:
    """A ``DiffusionGemmaMoEDecoderLayer`` with stand-in attention and grouped experts."""
    layer = DiffusionGemmaMoEDecoderLayer.__new__(DiffusionGemmaMoEDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = nn.Linear(4, 4, bias=False)
    layer.moe = nn.Module()
    layer.moe.router = nn.Linear(4, 2, bias=False)
    layer.moe.experts = nn.Linear(4, 4, bias=False)
    return layer


def _tiny_model() -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(8, 4)
    model.model.layers = nn.ModuleDict({"0": _decoder_layer(), "1": _decoder_layer()})
    model.model.norm = nn.Linear(4, 4, bias=False)
    return model


@pytest.fixture
def sharded_modules(monkeypatch: pytest.MonkeyPatch) -> list[nn.Module]:
    """Record every module handed to ``fully_shard`` by the base primitive."""
    calls: list[nn.Module] = []
    monkeypatch.setattr(parallelizer_mod, "fully_shard", lambda module, **kwargs: calls.append(module) or module)
    return calls


def test_sidecar_owns_diffusion_gemma_parallelizer():
    assert isinstance(dg4_fsdp.PARALLELIZER, ModelParallelizer)
    assert isinstance(dg4_fsdp.PARALLELIZER, dg4_fsdp.DiffusionGemmaModelParallelizer)


def test_decoder_layer_experts_are_sharded_before_the_layer(sharded_modules):
    """A decoder layer's ``moe.experts`` become their own unit, then the whole layer."""
    layer = _decoder_layer()

    dg4_fsdp.PARALLELIZER._fully_shard_module(layer, mesh=None, mp_policy=None)

    assert sharded_modules == [layer.moe.experts, layer]


def test_checkpoint_wrapped_decoder_layer_keeps_the_experts_unit(sharded_modules):
    """Whole-layer activation checkpointing does not hide the experts from the hook."""
    layer = _decoder_layer()
    wrapped = checkpoint_wrapper(layer)

    dg4_fsdp.PARALLELIZER._fully_shard_module(wrapped, mesh=None, mp_policy=None)

    assert sharded_modules == [layer.moe.experts, wrapped]


def test_modules_without_experts_are_one_unit(sharded_modules):
    """Embeddings, norms and the root are wrapped exactly once."""
    model = _tiny_model()

    dg4_fsdp.PARALLELIZER._fully_shard_module(model.model.embed_tokens, mesh=None, mp_policy=None)
    dg4_fsdp.PARALLELIZER._fully_shard_module(model, mesh=None, mp_policy=None)

    assert sharded_modules == [model.model.embed_tokens, model]


def test_recursive_sharding_orders_units_per_layer(sharded_modules):
    """The generic layer walk yields experts-then-layer for every decoder layer."""
    model = _tiny_model()
    mesh = types.SimpleNamespace(mesh_dim_names=())

    with dg4_fsdp.PARALLELIZER._bind_model(model):
        dg4_fsdp.PARALLELIZER._apply_fsdp_sharding(model, mesh, mp_policy=None, enable_fsdp2_prefetch=False)

    layers = list(model.model.layers.values())
    assert sharded_modules == [unit for layer in layers for unit in (layer.moe.experts, layer)]
