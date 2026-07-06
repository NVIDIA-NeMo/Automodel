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
from nemo_automodel.components.moe.experts import GroupedExpertsTE


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


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.attn_module = nn.Module()
        self.attn_module.fused_attention = nn.Identity()

    def forward(self, hidden_states):
        return self.o_proj(self.attn_module.fused_attention(self.q_proj(hidden_states)))


class _MoeExecution(GroupedExpertsTE):
    def __init__(self):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.eye(4))
        self.token_dispatcher = SimpleNamespace(hybridep_metadata_processor=_Preprocess())

    def forward(self, hidden_states, token_mask, weights, indices):
        del token_mask, weights, indices
        return hidden_states @ self.weight


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = nn.Module()
        self.mlp.gate = _Router()
        self.mlp.experts = _MoeExecution()


class _Model(nn.Module):
    def __init__(self, *, te_dpa=True, attention=False, moe=False, router=True, preprocess=True, layer_count=1):
        super().__init__()
        self.config = SimpleNamespace(model_type="test_moe")
        cuda_graph_modules = []
        if te_dpa:
            cuda_graph_modules.append("te_dpa")
        if attention:
            cuda_graph_modules.append("attn")
        if moe:
            cuda_graph_modules.append("moe")
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


def test_whole_attention_scope_uses_explicit_parameters_on_single_device(monkeypatch):
    captured_sample_args = []

    def make_graphed_callables(modules, sample_args, **kwargs):
        captured_sample_args.append(sample_args)
        assert kwargs["retain_graph_in_backward"] is True
        assert kwargs["num_warmup_iters"] == 3
        for module in modules:
            module.reset = lambda: None
        return modules

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: make_graphed_callables)
    model = _Model(te_dpa=False, attention=True, router=False, preprocess=False)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None
    entry = manager.entries[0]
    model.model.layers["0"].self_attn(torch.ones(2, 4))

    manager.capture()

    assert entry.name == "test_moe.layers.0.attention"
    assert entry.explicit_parameters is True
    assert len(captured_sample_args) == 1
    assert len(captured_sample_args[0][0]) == 3
    manager.close()


def test_full_moe_scope_uses_explicit_parameters(monkeypatch):
    captured_sample_args = []

    def make_graphed_callables(modules, sample_args, **kwargs):
        captured_sample_args.append(sample_args)
        assert kwargs["retain_graph_in_backward"] is True
        for module in modules:
            module.reset = lambda: None
        return modules

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: make_graphed_callables)
    model = _Model(te_dpa=False, moe=True, router=False, preprocess=False)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None
    entry = manager.entries[0]
    model.model.layers["0"].mlp.experts(
        torch.ones(2, 4),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, 2),
        torch.zeros(2, 2, dtype=torch.long),
    )

    manager.capture()

    assert entry.name == "test_moe.layers.0.moe"
    assert entry.explicit_parameters is True
    assert len(captured_sample_args[0][0]) == 5
    manager.close()


def test_explicit_parameter_adapter_matches_eager_forward_and_backward():
    torch.manual_seed(123)
    eager_target = _Attention()
    graph_target = _Attention()
    graph_target.load_state_dict(eager_target.state_dict())
    eager_input = torch.randn(2, 4, requires_grad=True)
    graph_input = eager_input.detach().clone().requires_grad_()
    captured_call = partial_graphs._CapturedCall.from_call((graph_input,), {})
    adapter = partial_graphs._ExplicitParameterCallAdapter(graph_target, captured_call)

    eager_output = eager_target(eager_input)
    graph_output = adapter(*adapter.capture_inputs)
    torch.testing.assert_close(graph_output, eager_output)

    eager_output.sum().backward()
    graph_output.sum().backward()
    torch.testing.assert_close(adapter.capture_inputs[0].grad, eager_input.grad)
    for (name, eager_parameter), graph_parameter in zip(eager_target.named_parameters(), adapter.capture_parameters):
        assert graph_parameter.grad is not None, name
        torch.testing.assert_close(graph_parameter.grad, eager_parameter.grad)
    assert all(parameter.grad is None for parameter in graph_target.parameters())


def test_explicit_parameter_adapter_rejects_replaced_buffer_storage():
    class _BufferedAttention(_Attention):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(()))

        def forward(self, x):
            return super().forward(x) * self.scale

    target = _BufferedAttention()
    captured_call = partial_graphs._CapturedCall.from_call((torch.ones(2, 4),), {})
    adapter = partial_graphs._ExplicitParameterCallAdapter(target, captured_call)
    target.scale = torch.full((), 2.0)

    with pytest.raises(RuntimeError, match="buffer storage changed"):
        adapter.replay_inputs((torch.ones(2, 4),))


def test_explicit_parameter_adapter_rejects_parameter_without_local_storage():
    class _NoStorageParameter(nn.Parameter):
        def data_ptr(self):
            raise RuntimeError("no local storage")

    target = _Attention()
    target.q_proj.weight = _NoStorageParameter(target.q_proj.weight.detach())
    captured_call = partial_graphs._CapturedCall.from_call((torch.ones(2, 4),), {})

    with pytest.raises(RuntimeError, match="stable local storage"):
        partial_graphs._ExplicitParameterCallAdapter(target, captured_call)


def test_explicit_parameter_adapter_rejects_unmaterialized_parameter_on_replay():
    class _NoStorageParameter(nn.Parameter):
        def data_ptr(self):
            raise RuntimeError("no local storage")

    target = _Attention()
    captured_call = partial_graphs._CapturedCall.from_call((torch.ones(2, 4),), {})
    adapter = partial_graphs._ExplicitParameterCallAdapter(target, captured_call)
    target.q_proj.weight = _NoStorageParameter(target.q_proj.weight.detach())

    with pytest.raises(RuntimeError, match="stable local storage"):
        adapter.replay_inputs((torch.ones(2, 4),))


def test_explicit_parameter_adapter_rejects_changed_parameter_metadata():
    target = _Attention()
    captured_call = partial_graphs._CapturedCall.from_call((torch.ones(2, 4),), {})
    adapter = partial_graphs._ExplicitParameterCallAdapter(target, captured_call)
    target.q_proj.weight = nn.Parameter(torch.ones(5, 4))

    with pytest.raises(RuntimeError, match="parameter metadata changed"):
        adapter.replay_inputs((torch.ones(2, 4),))


def test_parameter_owner_is_resharded_when_capture_fails(monkeypatch):
    class _CaptureOwner(nn.Module):
        def __init__(self):
            super().__init__()
            self.events = []

        def unshard(self, *, async_op):
            self.events.append(("unshard", async_op))

        def reshard(self):
            self.events.append(("reshard",))

    owner = _CaptureOwner()
    target = _Attention()
    entry = partial_graphs._PartialGraphEntry(
        name="whole_attention",
        target=target,
        explicit_parameters=True,
        capture_owner=owner,
    )
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    target(torch.ones(2, 4))
    monkeypatch.setattr(
        partial_graphs,
        "_get_make_graphed_callables",
        lambda: lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )

    with pytest.raises(RuntimeError, match="Partial CUDA graph capture failed"):
        manager.capture()

    assert owner.events == [("unshard", False), ("reshard",)]
    manager.close()


def test_capture_failure_resets_staged_graphs_and_keeps_all_targets_eager(monkeypatch):
    reset_adapters = []
    capture_count = 0

    def make_graphed_callables(modules, sample_args, **kwargs):
        nonlocal capture_count
        del sample_args, kwargs
        capture_count += 1
        if capture_count == 2:
            raise RuntimeError("second target failed")
        adapter = modules[0]
        adapter.reset = lambda: reset_adapters.append(adapter)
        return (adapter,)

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: make_graphed_callables)
    targets = (nn.Identity(), nn.Identity())
    original_forwards = tuple(target.forward for target in targets)
    manager = partial_graphs.PartialCudaGraphManager(
        [
            partial_graphs._PartialGraphEntry(name=f"target.{index}", target=target)
            for index, target in enumerate(targets)
        ]
    )
    manager.start_recording()
    for target in targets:
        target(torch.ones(2, 3))

    with pytest.raises(RuntimeError, match="Partial CUDA graph capture failed"):
        manager.capture()

    assert len(reset_adapters) == 1
    assert tuple(target.forward for target in targets) == original_forwards
    assert manager.stats() == {"captured": 0, "replayed": 0, "fallback": 0}
    manager.close()


def test_second_input_variant_failure_resets_first_graph(monkeypatch):
    reset_adapters = []
    capture_count = 0

    def make_graphed_callables(modules, sample_args, **kwargs):
        nonlocal capture_count
        del sample_args, kwargs
        capture_count += 1
        if capture_count == 2:
            raise RuntimeError("second variant failed")
        adapter = modules[0]
        adapter.reset = lambda: reset_adapters.append(adapter)
        return (adapter,)

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: make_graphed_callables)
    target = nn.Identity()
    original_forward = target.forward
    entry = partial_graphs._PartialGraphEntry(name="attention", target=target, capture_input_variants=2)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    target(torch.ones(2, requires_grad=False))
    target(torch.ones(2, requires_grad=True))

    with pytest.raises(RuntimeError, match="Partial CUDA graph capture failed"):
        manager.capture()

    assert len(reset_adapters) == 1
    assert target.forward == original_forward
    manager.close()


def test_rejects_nonempty_parameter_and_buffer_without_storage():
    parameter = nn.Parameter(torch.empty(2, device="meta"))
    with pytest.raises(RuntimeError, match="stable local storage"):
        partial_graphs._require_local_parameter_storage("weight", parameter)

    target = nn.Module()
    target.register_buffer("cache", torch.empty(2, device="meta"))
    with pytest.raises(RuntimeError, match="stable local storage"):
        partial_graphs._named_buffer_storage(target)


def test_graph_helper_fails_with_actionable_error_when_te_is_unavailable(monkeypatch):
    monkeypatch.setattr(partial_graphs, "safe_import_te", lambda: (False, object()))

    with pytest.raises(RuntimeError, match="require a working Transformer Engine"):
        partial_graphs._get_make_graphed_callables()


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


def test_rejects_whole_attention_with_activation_checkpointing():
    model = _Model(te_dpa=False, attention=True, router=False, preprocess=False)
    with pytest.raises(RuntimeError, match="Whole-attention CUDA graphs do not support activation checkpointing"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([model], activation_checkpointing=True)


def test_rejects_full_moe_with_activation_checkpointing():
    model = _Model(te_dpa=False, moe=True, router=False, preprocess=False)
    with pytest.raises(RuntimeError, match="Full MoE CUDA graphs do not support activation checkpointing"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([model], activation_checkpointing=True)


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
