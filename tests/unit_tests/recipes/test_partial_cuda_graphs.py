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

import pytest
import torch
from torch import nn

import nemo_automodel.recipes.llm.partial_cuda_graphs as partial_graphs


class _RouterCore(nn.Module):
    def forward(self, scores, gate):
        return scores[:, :2], torch.zeros_like(scores[:, :2], dtype=torch.long), scores


class _Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.routing_core = _RouterCore()
        self.router_replay = None
        self.e_score_correction_bias = None


class _Preprocess(nn.Module):
    permute_fusion = True

    def forward(self, indices, probabilities):
        return indices >= 0, probabilities


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.attn_module = nn.Module()
        self.self_attn.attn_module.fused_attention = nn.Identity()
        self.mlp = nn.Module()
        self.mlp.gate = _Router()
        self.mlp.experts = nn.Module()
        self.mlp.experts.token_dispatcher = SimpleNamespace(hybridep_metadata_processor=_Preprocess())


class _Model(nn.Module):
    def __init__(self, *, attention=True, router=True, preprocess=True, layer_count=1):
        super().__init__()
        self.config = SimpleNamespace(model_type="test_moe")
        cuda_graph_modules = []
        if attention:
            cuda_graph_modules.append("attn")
        if router:
            cuda_graph_modules.append("moe_router")
        if preprocess:
            cuda_graph_modules.append("moe_preprocess")
        self.backend = SimpleNamespace(
            cuda_graph_modules=cuda_graph_modules,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(index): _Layer() for index in range(layer_count)})


def _install_fake_graph(monkeypatch):
    captured_kwargs = []

    class _GraphedAdapter(nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.captured_call = adapter.captured_call
            self.target_forward = adapter.target.forward
            self.reset_count = 0

        def forward(self, *tensor_inputs):
            args, kwargs = self.captured_call.rebuild(tensor_inputs)
            return self.target_forward(*args, **kwargs)

        def reset(self):
            self.reset_count += 1

    def make_graphed_callables(modules, sample_args, **kwargs):
        del sample_args
        captured_kwargs.append(kwargs)
        return tuple(_GraphedAdapter(module) for module in modules)

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: make_graphed_callables)
    return captured_kwargs


def test_discovers_only_fixed_shape_scopes():
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([_Model()])

    assert manager is not None
    assert [entry.name for entry in manager.entries] == [
        "test_moe.layers.0.fused_attention",
        "test_moe.layers.0.moe_router",
        "test_moe.layers.0.moe_preprocess",
    ]
    manager.close()


def test_module_list_applies_to_every_compatible_layer():
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts(
        [_Model(router=False, preprocess=False, layer_count=2)]
    )

    assert manager is not None
    assert [entry.name for entry in manager.entries] == [
        "test_moe.layers.0.fused_attention",
        "test_moe.layers.1.fused_attention",
    ]
    manager.close()


def test_capture_replays_matching_input_and_falls_back_on_metadata_change(monkeypatch):
    captured_kwargs = _install_fake_graph(monkeypatch)
    target = nn.Identity()
    entry = partial_graphs._PartialGraphEntry(name="attention", target=target)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    target(torch.ones(2, 3))

    manager.capture()
    torch.testing.assert_close(target(torch.full((2, 3), 2.0)), torch.full((2, 3), 2.0))
    torch.testing.assert_close(target(torch.ones(3, 3)), torch.ones(3, 3))

    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 1}
    assert captured_kwargs == [
        {
            "num_warmup_iters": 3,
            "enabled": (False,),
        }
    ]
    manager.close()


def test_capture_preserves_alias_and_non_tensor_control_contract(monkeypatch):
    _install_fake_graph(monkeypatch)

    class _AliasModule(nn.Module):
        def forward(self, left, right, *, mode):
            return left + right if mode == "add" else left - right

    target = _AliasModule()
    entry = partial_graphs._PartialGraphEntry(name="alias", target=target)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    sample = torch.ones(2)
    target(sample, sample, mode="add")
    manager.capture()

    replay = torch.full((2,), 2.0)
    torch.testing.assert_close(target(replay, replay, mode="add"), torch.full((2,), 4.0))
    target(replay, replay.clone(), mode="add")
    target(replay, replay, mode="subtract")
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 2}
    manager.close()


def test_attention_records_two_checkpoint_input_contracts(monkeypatch):
    _install_fake_graph(monkeypatch)
    target = nn.Identity()
    entry = partial_graphs._PartialGraphEntry(
        name="attention",
        target=target,
        capture_input_variants=2,
    )
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    target(torch.ones(2, requires_grad=False))
    target(torch.ones(2, requires_grad=True))

    manager.capture()

    assert entry.capture_count == 2
    manager.close()


def test_missing_eager_sample_fails_closed():
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([_Model(router=False, preprocess=False)])
    assert manager is not None

    with pytest.raises(RuntimeError, match="must have an iteration-0 sample"):
        manager.capture()
    manager.close()


def test_rejects_router_and_preprocess_with_activation_checkpointing():
    with pytest.raises(RuntimeError, match="cannot recompute across the partial MoE router/preprocess"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([_Model()], activation_checkpointing=True)


@pytest.mark.parametrize(
    ("model_parts", "pipeline_parallel"),
    [
        ([_Model()], True),
        ([_Model(), _Model()], False),
    ],
)
def test_rejects_pipeline_stages_and_virtual_chunks(model_parts, pipeline_parallel):
    with pytest.raises(RuntimeError, match="pipeline stages and virtual pipeline chunks"):
        partial_graphs.PartialCudaGraphManager.from_model_parts(
            model_parts,
            pipeline_parallel=pipeline_parallel,
        )


def test_close_restores_eager_forward_and_is_idempotent(monkeypatch):
    _install_fake_graph(monkeypatch)
    target = nn.Identity()
    original_forward = target.forward
    entry = partial_graphs._PartialGraphEntry(name="attention", target=target)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    target(torch.ones(2))
    manager.capture()

    manager.close()
    manager.close()

    assert target.forward == original_forward
    assert manager.stats() == {"captured": 1, "replayed": 0, "fallback": 0}
