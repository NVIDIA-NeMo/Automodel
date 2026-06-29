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

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

import nemo_automodel.recipes.llm.partial_cuda_graphs as partial_graphs


class _DynamicSplitModule(nn.Module):
    def __init__(self, device: torch.device | str = "cpu"):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75, device=device))

    def forward(self, x, splits, probs, splits_alias, probs_alias, *, mode="weighted"):
        assert splits is splits_alias
        assert probs is probs_alias
        if mode == "weighted":
            coefficients = torch.arange(1, splits.numel() + 1, dtype=x.dtype, device=x.device)
        else:
            coefficients = torch.ones(splits.numel(), dtype=x.dtype, device=x.device)
        split_factor = torch.sum(splits.to(x.dtype) * coefficients)
        alias_split_factor = torch.sum(splits_alias.to(x.dtype) * coefficients)
        return x * self.scale + probs * split_factor + probs_alias * alias_split_factor


class _FakeRouterCore(nn.Module):
    def forward(self, scores, gate):
        assert isinstance(gate, _FakeRouter)
        weights = scores[:, :2]
        indices = torch.zeros_like(weights, dtype=torch.long)
        return weights, indices, scores


class _FakeRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.routing_core = _FakeRouterCore()
        self.router_replay = None
        self.e_score_correction_bias = None

    def forward(self, x, token_mask, cp_mesh=None):
        del cp_mesh
        scores = x * self.scale * token_mask.unsqueeze(-1)
        weights, indices, _ = self.routing_core(scores, self)
        return weights, indices, None


class _FakeMoEPreprocess(nn.Module):
    permute_fusion = True

    def forward(self, indices, probs):
        return indices >= 0, probs


def _make_entry(module, *, canonicalizer=None, fp8_enabled=False, expert_bucket_tokens=None):
    return partial_graphs._PartialGraphEntry(
        name="test.dynamic_experts",
        target=module,
        fp8_enabled=fp8_enabled,
        canonicalizer=canonicalizer,
        expert_bucket_tokens=expert_bucket_tokens,
    )


class _FakeLayer(nn.Module):
    def __init__(self, *, expert_device="cpu"):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.attn_module = nn.Module()
        self.self_attn.attn_module.fused_attention = nn.Identity()
        self.mlp = nn.Module()
        self.mlp.gate = _FakeRouter()
        self.mlp.experts = nn.Module()
        self.mlp.experts.use_te_ops = True
        self.mlp.experts.__dict__["_te_grouped_mlp"] = _DynamicSplitModule(expert_device)
        self.mlp.experts.token_dispatcher = SimpleNamespace(hybridep_metadata_processor=_FakeMoEPreprocess())


class _FSDPStyleGptOssWrapper(nn.Module):
    """Match the attribute structure retained by FSDP2's dynamic wrapper class."""

    def __init__(
        self,
        *,
        layer_count=1,
        layer_limit=1,
        expert_device="cpu",
        attention=True,
        moe_router=False,
        moe_preprocess=False,
        experts=True,
        expert_bucket_tokens=None,
    ):
        super().__init__()
        self.config = SimpleNamespace(model_type="gpt_oss")
        self.backend = SimpleNamespace(
            partial_cuda_graph_attention=attention,
            partial_cuda_graph_moe_router=moe_router,
            partial_cuda_graph_moe_preprocess=moe_preprocess,
            partial_cuda_graph_experts=experts,
            partial_cuda_graph_expert_bucket_tokens=expert_bucket_tokens,
            partial_cuda_graph_layer_limit=layer_limit,
            te_fp8=None,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict(
            {str(index): _FakeLayer(expert_device=expert_device) for index in range(layer_count)}
        )


def _install_fake_graph(monkeypatch):
    class _FakeGraphedAdapter(nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self._captured_call = adapter.captured_call
            self._target_forward = adapter.target.forward

        def forward(self, *tensor_inputs):
            args, kwargs = self._captured_call.rebuild(tensor_inputs)
            return self._target_forward(*args, **kwargs)

    def fake_make_graphed_callables(modules, _sample_args, **_kwargs):
        assert all(len(args) == len({id(tensor) for tensor in args}) for args in _sample_args)
        return tuple(_FakeGraphedAdapter(module) for module in modules)

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: fake_make_graphed_callables)


def test_discovers_targets_through_fsdp_style_gpt_oss_wrapper():
    model = _FSDPStyleGptOssWrapper()
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])

    assert manager is not None
    assert [entry.name for entry in manager.entries] == [
        "gpt_oss.layers.0.fused_attention",
        "gpt_oss.layers.0.te_ops_experts",
    ]
    for entry in manager.entries:
        entry.stop_recording()


def test_discovers_megatron_style_dropless_moe_scopes():
    model = _FSDPStyleGptOssWrapper(moe_router=True, moe_preprocess=True, experts=False)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])

    assert manager is not None
    assert [entry.name for entry in manager.entries] == [
        "gpt_oss.layers.0.fused_attention",
        "gpt_oss.layers.0.moe_router",
        "gpt_oss.layers.0.moe_preprocess",
    ]
    for entry in manager.entries:
        entry.stop_recording()


def test_discovers_targets_below_pytorch_checkpoint_wrapper():
    model = _FSDPStyleGptOssWrapper()
    model.model.layers["0"] = checkpoint_wrapper(model.model.layers["0"])

    manager = partial_graphs.PartialCudaGraphManager.from_model_parts(
        [model],
        activation_checkpointing=True,
    )

    assert manager is not None
    assert [entry.name for entry in manager.entries] == [
        "gpt_oss.layers.0.fused_attention",
        "gpt_oss.layers.0.te_ops_experts",
    ]
    for entry in manager.entries:
        entry.stop_recording()


def test_scoped_router_and_preprocess_replay_dynamic_values(monkeypatch):
    _install_fake_graph(monkeypatch)
    model = _FSDPStyleGptOssWrapper(moe_router=True, moe_preprocess=True, experts=False)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None

    layer = model.model.layers["0"]
    attention = layer.self_attn.attn_module.fused_attention
    router = layer.mlp.gate
    preprocess = layer.mlp.experts.token_dispatcher.hybridep_metadata_processor

    attention(torch.ones(2, 3))
    router(torch.ones(2, 3), torch.ones(2, dtype=torch.bool), None)
    preprocess(torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2))
    manager.capture()

    weights, indices, aux_loss = router(
        torch.full((2, 3), 2.0),
        torch.tensor([True, False]),
        None,
    )
    routing_map, probs = preprocess(
        torch.tensor([[0, -1], [1, 2]]),
        torch.tensor([[0.75, 0.0], [0.6, 0.4]]),
    )

    torch.testing.assert_close(weights, torch.tensor([[2.0, 2.0], [0.0, 0.0]]))
    assert indices.shape == (2, 2)
    assert aux_loss is None
    torch.testing.assert_close(routing_map, torch.tensor([[True, False], [True, True]]))
    torch.testing.assert_close(probs, torch.tensor([[0.75, 0.0], [0.6, 0.4]]))
    assert manager.stats() == {"captured": 3, "replayed": 2, "fallback": 0}


def test_empty_expert_sample_is_skipped_while_attention_is_captured(monkeypatch, caplog):
    _install_fake_graph(monkeypatch)
    model = _FSDPStyleGptOssWrapper()
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None

    attention = model.model.layers["0"].self_attn.attn_module.fused_attention
    experts = model.model.layers["0"].mlp.experts._te_grouped_mlp
    attention(torch.ones(2, 3))

    with caplog.at_level("WARNING"):
        manager.capture()

    attention(torch.ones(2, 3))
    splits = torch.tensor([1, 1], dtype=torch.int64)
    probs = torch.ones(2, 1)
    experts(torch.ones(2, 3), splits, probs, splits, probs)

    assert "received no iteration-0 expert tokens" in caplog.text
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 0}
    assert manager.entries[0].capture_count == 1
    assert manager.entries[1].capture_count == 0


def test_missing_attention_sample_still_fails_capture():
    model = _FSDPStyleGptOssWrapper()
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None

    with pytest.raises(RuntimeError, match="non-expert partial CUDA graph target"):
        manager.capture()


def test_allows_activation_checkpointing_for_guarded_graph_scopes():
    model = _FSDPStyleGptOssWrapper()
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model], activation_checkpointing=True)
    assert manager is not None
    for entry in manager.entries:
        entry.stop_recording()


def test_rejects_activation_checkpointing_across_router_preprocess_scope():
    model = _FSDPStyleGptOssWrapper(moe_router=True, moe_preprocess=True, experts=False)
    with pytest.raises(RuntimeError, match="cannot recompute across the partial MoE router/preprocess"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([model], activation_checkpointing=True)


def test_rejects_pipeline_parallel_with_one_local_model_part():
    model = _FSDPStyleGptOssWrapper()
    with pytest.raises(RuntimeError, match="pipeline parallel size 1"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([model], pipeline_parallel=True)


def test_rejects_layer_limit_larger_than_discovered_model():
    model = _FSDPStyleGptOssWrapper(layer_count=1, layer_limit=2)
    with pytest.raises(RuntimeError, match="exceeds the 1 discovered"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([model])


def test_rejects_unallocated_expert_parameters():
    model = _FSDPStyleGptOssWrapper(expert_device="meta")
    with pytest.raises(RuntimeError, match="DTensor or unallocated parameters"):
        partial_graphs.PartialCudaGraphManager.from_model_parts([model])


def test_expert_replay_keeps_changed_split_values_and_aliases(monkeypatch):
    _install_fake_graph(monkeypatch)
    module = _DynamicSplitModule()
    reference = copy.deepcopy(module)
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    first_x = torch.randn(4, 3, requires_grad=True)
    first_probs = torch.randn(4, 1, requires_grad=True)
    first_splits = torch.tensor([1, 3], dtype=torch.int64)
    module(first_x, first_splits, first_probs, first_splits, first_probs)

    assert entry.captured_call is not None
    assert len(entry.captured_call.sample_tensors) == 3
    assert entry.captured_call.tensor_input_indices == (0, 1, 2, 1, 2)
    manager.capture()

    graph_x = torch.randn(4, 3, requires_grad=True)
    graph_probs = torch.randn(4, 1, requires_grad=True)
    changed_splits = torch.tensor([2, 2], dtype=torch.int64)
    graph_output = module(graph_x, changed_splits, graph_probs, changed_splits, graph_probs)
    graph_output.sum().backward()

    ref_x = graph_x.detach().clone().requires_grad_(True)
    ref_probs = graph_probs.detach().clone().requires_grad_(True)
    ref_output = reference(ref_x, changed_splits, ref_probs, changed_splits, ref_probs)
    ref_output.sum().backward()

    torch.testing.assert_close(graph_output, ref_output)
    torch.testing.assert_close(graph_x.grad, ref_x.grad)
    torch.testing.assert_close(graph_probs.grad, ref_probs.grad, rtol=0, atol=0)
    torch.testing.assert_close(graph_probs.grad, torch.full_like(graph_probs, 36), rtol=0, atol=0)
    torch.testing.assert_close(module.scale.grad, reference.scale.grad, rtol=0, atol=0)
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 0}


def test_expert_bucket_replays_changed_token_count_and_preserves_gradients(monkeypatch):
    _install_fake_graph(monkeypatch)
    model = _FSDPStyleGptOssWrapper(attention=False, expert_bucket_tokens=4)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None
    module = model.model.layers["0"].mlp.experts._te_grouped_mlp
    entry = manager.entries[0]

    first_x = torch.randn(2, 3, requires_grad=True)
    first_probs = torch.randn(2, 1, requires_grad=True)
    first_splits = torch.tensor([1, 1], dtype=torch.int64)
    module(first_x, first_splits, first_probs, first_splits, first_probs)

    assert entry.captured_call is not None
    assert entry.captured_call.sample_tensors[0].shape == (4, 3)
    torch.testing.assert_close(entry.captured_call.sample_tensors[1], first_splits)
    assert entry.captured_call.sample_tensors[2].shape == (4, 1)
    manager.capture()

    graph_x = torch.randn(3, 3, requires_grad=True)
    graph_probs = torch.randn(3, 1, requires_grad=True)
    changed_splits = torch.tensor([2, 1], dtype=torch.int64)
    graph_output = module(graph_x, changed_splits, graph_probs, changed_splits, graph_probs)
    graph_output.sum().backward()
    graph_scale_grad = module.scale.grad.detach().clone()
    module.zero_grad(set_to_none=True)

    ref_x = graph_x.detach().clone().requires_grad_(True)
    ref_probs = graph_probs.detach().clone().requires_grad_(True)
    ref_output = entry.original_forward(ref_x, changed_splits, ref_probs, changed_splits, ref_probs)
    ref_output.sum().backward()

    torch.testing.assert_close(graph_output, ref_output)
    torch.testing.assert_close(graph_x.grad, ref_x.grad)
    torch.testing.assert_close(graph_probs.grad, ref_probs.grad)
    torch.testing.assert_close(graph_scale_grad, module.scale.grad)
    assert graph_output.shape == (3, 3)
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 0}
    assert manager.expert_bucket_stats() == {
        "entries": 1,
        "capacity_tokens": 4,
        "bucketed_replay": 1,
        "padding_tokens": 1,
        "overflow_fallback": 0,
        "empty_fallback": 0,
        "capture_overflow_skip": 0,
        "capture_empty_skip": 0,
    }


def test_expert_bucket_overflow_and_empty_calls_fall_back_without_token_loss(monkeypatch):
    _install_fake_graph(monkeypatch)
    model = _FSDPStyleGptOssWrapper(attention=False, expert_bucket_tokens=4)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None
    module = model.model.layers["0"].mlp.experts._te_grouped_mlp
    entry = manager.entries[0]

    first_splits = torch.tensor([1, 1], dtype=torch.int64)
    first_probs = torch.randn(2, 1)
    module(torch.randn(2, 3), first_splits, first_probs, first_splits, first_probs)
    manager.capture()

    overflow_x = torch.randn(5, 3)
    overflow_probs = torch.randn(5, 1)
    overflow_splits = torch.tensor([2, 3], dtype=torch.int64)
    overflow_output = module(
        overflow_x,
        overflow_splits,
        overflow_probs,
        overflow_splits,
        overflow_probs,
    )
    expected_overflow = entry.original_forward(
        overflow_x,
        overflow_splits,
        overflow_probs,
        overflow_splits,
        overflow_probs,
    )

    empty_x = torch.empty(0, 3)
    empty_probs = torch.empty(0, 1)
    empty_splits = torch.tensor([0, 0], dtype=torch.int64)
    empty_output = module(empty_x, empty_splits, empty_probs, empty_splits, empty_probs)

    torch.testing.assert_close(overflow_output, expected_overflow)
    assert overflow_output.shape[0] == 5
    assert empty_output.shape == (0, 3)
    assert manager.stats() == {"captured": 1, "replayed": 0, "fallback": 2}
    bucket_stats = manager.expert_bucket_stats()
    assert bucket_stats["overflow_fallback"] == 1
    assert bucket_stats["empty_fallback"] == 1


def test_iteration_zero_overflow_skips_expert_bucket_even_after_fitting_call(monkeypatch, caplog):
    _install_fake_graph(monkeypatch)
    model = _FSDPStyleGptOssWrapper(attention=False, expert_bucket_tokens=4)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None
    module = model.model.layers["0"].mlp.experts._te_grouped_mlp

    fitting_splits = torch.tensor([1, 1], dtype=torch.int64)
    fitting_probs = torch.randn(2, 1)
    module(torch.randn(2, 3), fitting_splits, fitting_probs, fitting_splits, fitting_probs)
    overflow_splits = torch.tensor([2, 3], dtype=torch.int64)
    overflow_probs = torch.randn(5, 1)
    module(torch.randn(5, 3), overflow_splits, overflow_probs, overflow_splits, overflow_probs)

    with caplog.at_level("WARNING"):
        manager.capture()

    assert "iteration 0 routed token count 5 exceeds expert bucket capacity 4" in caplog.text
    assert manager.stats() == {"captured": 0, "replayed": 0, "fallback": 0}
    assert manager.expert_bucket_stats()["capture_overflow_skip"] == 1
    assert manager.expert_bucket_stats()["capture_empty_skip"] == 0


def test_unobserved_iteration_zero_expert_bucket_is_counted_as_empty_skip(caplog):
    model = _FSDPStyleGptOssWrapper(attention=False, expert_bucket_tokens=4)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None

    with caplog.at_level("WARNING"):
        manager.capture()

    assert "received no iteration-0 expert tokens" in caplog.text
    assert manager.stats() == {"captured": 0, "replayed": 0, "fallback": 0}
    assert manager.expert_bucket_stats()["capture_empty_skip"] == 1


def test_expert_bucket_skips_impossible_zero_local_expert_split_tensor(caplog):
    model = _FSDPStyleGptOssWrapper(attention=False, expert_bucket_tokens=4)
    manager = partial_graphs.PartialCudaGraphManager.from_model_parts([model])
    assert manager is not None
    module = model.model.layers["0"].mlp.experts._te_grouped_mlp

    splits = torch.empty(0, dtype=torch.int64)
    probs = torch.randn(2, 1)
    module(torch.randn(2, 3), splits, probs, splits, probs)
    with caplog.at_level("WARNING"):
        manager.capture()

    assert "expert split tensor has no local experts" in caplog.text
    assert manager.stats() == {"captured": 0, "replayed": 0, "fallback": 0}


def test_tensor_metadata_change_falls_back_eagerly(monkeypatch):
    _install_fake_graph(monkeypatch)
    module = _DynamicSplitModule()
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    splits = torch.tensor([2, 2], dtype=torch.int64)
    probs = torch.randn(4, 1)
    module(torch.randn(4, 3), splits, probs, splits, probs)
    manager.capture()

    larger_x = torch.randn(5, 3)
    larger_probs = torch.randn(5, 1)
    output = module(larger_x, splits, larger_probs, splits, larger_probs)

    torch.testing.assert_close(output, entry.original_forward(larger_x, splits, larger_probs, splits, larger_probs))
    assert manager.stats() == {"captured": 1, "replayed": 0, "fallback": 1}


def test_parameter_pointer_change_falls_back_eagerly(monkeypatch):
    _install_fake_graph(monkeypatch)
    module = _DynamicSplitModule()
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    splits = torch.tensor([2, 2], dtype=torch.int64)
    probs = torch.randn(4, 1)
    x = torch.randn(4, 3)
    module(x, splits, probs, splits, probs)
    manager.capture()
    module.scale = nn.Parameter(module.scale.detach().clone())

    output = module(x, splits, probs, splits, probs)
    torch.testing.assert_close(output, entry.original_forward(x, splits, probs, splits, probs))
    assert manager.stats()["fallback"] == 1


def test_non_tensor_control_change_falls_back_eagerly(monkeypatch):
    _install_fake_graph(monkeypatch)
    module = _DynamicSplitModule()
    entry = _make_entry(module)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    splits = torch.tensor([2, 2], dtype=torch.int64)
    probs = torch.randn(4, 1)
    x = torch.randn(4, 3)
    module(x, splits, probs, splits, probs, mode="weighted")
    manager.capture()

    output = module(x, splits, probs, splits, probs, mode="uniform")
    torch.testing.assert_close(
        output,
        entry.original_forward(x, splits, probs, splits, probs, mode="uniform"),
    )
    assert manager.stats()["fallback"] == 1


def test_explicit_capture_error_is_not_silenced(monkeypatch):
    module = _DynamicSplitModule()
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()
    splits = torch.tensor([2, 2], dtype=torch.int64)
    probs = torch.randn(4, 1)
    module(torch.randn(4, 3), splits, probs, splits, probs)

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: fail_capture)
    with pytest.raises(RuntimeError, match="Explicit partial CUDA graph capture failed"):
        manager.capture()


def test_bf16_attention_canonicalization_ignores_fp8_metadata():
    fp8_meta = {"scale": torch.ones(1)}
    quantizers = {"q": object()}
    args, kwargs = partial_graphs._canonicalize_bf16_fused_attention(
        (torch.ones(1),),
        {"fp8": False, "fp8_meta": fp8_meta, "quantizers": quantizers, "qkv_layout": "bshd_bshd_bshd"},
    )
    assert len(args) == 1
    assert kwargs["fp8_meta"] is None
    assert kwargs["quantizers"] is None

    with pytest.raises(RuntimeError, match="require BF16"):
        partial_graphs._canonicalize_bf16_fused_attention((), {"fp8": True})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA graph parity")
def test_expert_cuda_graph_forward_backward_parity_with_changed_split_contents():
    pytest.importorskip("transformer_engine")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    module = _DynamicSplitModule(device)
    reference = copy.deepcopy(module)
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    first_x = torch.randn(4, 3, device=device, requires_grad=True)
    first_probs = torch.randn(4, 1, device=device, requires_grad=True)
    first_splits = torch.tensor([1, 3], dtype=torch.int64, device=device)
    module(first_x, first_splits, first_probs, first_splits, first_probs)
    assert entry.captured_call is not None
    assert len(entry.captured_call.sample_tensors) == 3
    manager.capture()
    module.zero_grad(set_to_none=True)

    graph_x = torch.randn(4, 3, device=device, requires_grad=True)
    graph_probs = torch.randn(4, 1, device=device, requires_grad=True)
    changed_splits = torch.tensor([2, 2], dtype=torch.int64, device=device)
    graph_output = module(graph_x, changed_splits, graph_probs, changed_splits, graph_probs)
    graph_output.sum().backward()

    ref_x = graph_x.detach().clone().requires_grad_(True)
    ref_probs = graph_probs.detach().clone().requires_grad_(True)
    ref_output = reference(ref_x, changed_splits, ref_probs, changed_splits, ref_probs)
    ref_output.sum().backward()

    torch.testing.assert_close(graph_output, ref_output)
    torch.testing.assert_close(graph_x.grad, ref_x.grad)
    torch.testing.assert_close(graph_probs.grad, ref_probs.grad, rtol=0, atol=0)
    torch.testing.assert_close(graph_probs.grad, torch.full_like(graph_probs, 36), rtol=0, atol=0)
    torch.testing.assert_close(module.scale.grad, reference.scale.grad, rtol=0, atol=0)
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA graph GA parity")
def test_expert_cuda_graph_ga2_accumulates_exact_gradients():
    pytest.importorskip("transformer_engine")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    module = _DynamicSplitModule(device)
    reference = copy.deepcopy(module)
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    first_x = torch.randn(4, 3, device=device, requires_grad=True)
    first_probs = torch.randn(4, 1, device=device, requires_grad=True)
    first_splits = torch.tensor([1, 3], dtype=torch.int64, device=device)
    module(first_x, first_splits, first_probs, first_splits, first_probs)
    assert entry.captured_call is not None
    assert len(entry.captured_call.sample_tensors) == 3
    manager.capture()

    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=False)
    reference_optimizer.zero_grad(set_to_none=False)

    for split_values, expected_prob_grad in (([2, 2], 36), ([3, 1], 30)):
        graph_x = torch.randn(4, 3, device=device, requires_grad=True)
        graph_probs = torch.randn(4, 1, device=device, requires_grad=True)
        splits = torch.tensor(split_values, dtype=torch.int64, device=device)
        graph_output = module(graph_x, splits, graph_probs, splits, graph_probs)
        graph_output.sum().backward()

        ref_x = graph_x.detach().clone().requires_grad_(True)
        ref_probs = graph_probs.detach().clone().requires_grad_(True)
        ref_output = reference(ref_x, splits, ref_probs, splits, ref_probs)
        ref_output.sum().backward()

        torch.testing.assert_close(graph_output, ref_output, rtol=0, atol=0)
        torch.testing.assert_close(graph_x.grad, ref_x.grad, rtol=0, atol=0)
        torch.testing.assert_close(graph_probs.grad, ref_probs.grad, rtol=0, atol=0)
        torch.testing.assert_close(
            graph_probs.grad,
            torch.full_like(graph_probs, expected_prob_grad),
            rtol=0,
            atol=0,
        )

    torch.testing.assert_close(module.scale.grad, reference.scale.grad, rtol=0, atol=0)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(module.scale, reference.scale, rtol=0, atol=0)
    assert manager.stats() == {"captured": 1, "replayed": 2, "fallback": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TE-ops graph parity")
def test_te_ops_expert_graph_forward_backward_parity_with_changed_split_contents():
    pytest.importorskip("transformer_engine")
    from transformer_engine.pytorch import ops as te_ops

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    module = te_ops.Sequential(
        te_ops.GroupedLinear(
            num_groups=2,
            in_features=64,
            out_features=128,
            bias=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        te_ops.ScaledClampedQGeGLU(glu_interleave_size=32, limit=7.0, alpha=1.702),
        te_ops.GroupedLinear(
            num_groups=2,
            in_features=64,
            out_features=64,
            bias=True,
            dtype=torch.bfloat16,
            device=device,
            scale_bias=True,
        ),
    )
    entry = _make_entry(module, canonicalizer=partial_graphs._canonicalize_te_ops_experts)
    manager = partial_graphs.PartialCudaGraphManager([entry])
    manager.start_recording()

    first_x = torch.randn(64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    first_probs = torch.randn(64, device=device, dtype=torch.bfloat16, requires_grad=True)
    first_splits = torch.tensor([32, 32], dtype=torch.int64, device=device)
    module(first_x, first_splits, first_probs, first_splits, first_probs)
    assert entry.captured_call is not None
    assert len(entry.captured_call.sample_tensors) == 3
    manager.capture()
    module.zero_grad(set_to_none=True)

    graph_x = torch.randn(64, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    graph_probs = torch.randn(64, device=device, dtype=torch.bfloat16, requires_grad=True)
    changed_splits = torch.tensor([16, 48], dtype=torch.int64, device=device)
    graph_output = module(graph_x, changed_splits, graph_probs, changed_splits, graph_probs)
    graph_output.float().square().mean().backward()
    saved_graph_output = graph_output.detach().clone()
    saved_graph_x_grad = graph_x.grad.detach().clone()
    saved_graph_probs_grad = graph_probs.grad.detach().clone()
    saved_graph_param_grads = [parameter.grad.detach().clone() for parameter in module.parameters()]
    module.zero_grad(set_to_none=True)

    ref_x = graph_x.detach().clone().requires_grad_(True)
    ref_probs = graph_probs.detach().clone().requires_grad_(True)
    ref_output = entry.original_forward(ref_x, changed_splits, ref_probs, changed_splits, ref_probs)
    ref_output.float().square().mean().backward()

    torch.testing.assert_close(saved_graph_output, ref_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(saved_graph_x_grad, ref_x.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(saved_graph_probs_grad, ref_probs.grad, rtol=0, atol=0)
    for graph_grad, parameter in zip(saved_graph_param_grads, module.parameters()):
        torch.testing.assert_close(graph_grad, parameter.grad, rtol=0, atol=0)
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MXFP8 graph parity")
def test_te_ops_mxfp8_expert_graph_parity_with_changed_split_contents():
    pytest.importorskip("transformer_engine")
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import ops as te_ops

    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 grouped-MLP fusion requires Blackwell")
    if not te_ops.fused.ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported():
        pytest.skip("TE MXFP8 fused grouped MLP is not supported on this system")

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    recipe = MXFP8BlockScaling(fp8_dpa=False)
    module = te_ops.Sequential(
        te_ops.GroupedLinear(
            num_groups=2,
            in_features=256,
            out_features=512,
            bias=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        te_ops.ScaledClampedQGeGLU(glu_interleave_size=32, limit=7.0, alpha=1.702),
        te_ops.GroupedLinear(
            num_groups=2,
            in_features=256,
            out_features=256,
            bias=True,
            dtype=torch.bfloat16,
            device=device,
            scale_bias=True,
        ),
    )
    entry = _make_entry(
        module,
        canonicalizer=partial_graphs._canonicalize_te_ops_experts,
        fp8_enabled=True,
    )
    manager = partial_graphs.PartialCudaGraphManager([entry], fp8_recipe=recipe)
    manager.start_recording()

    first_x = torch.randn(768, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    first_probs = torch.randn(768, device=device, dtype=torch.bfloat16, requires_grad=True)
    first_splits = torch.tensor([256, 512], dtype=torch.int64, device=device)
    with te.autocast(enabled=True, recipe=recipe):
        module(first_x, first_splits, first_probs, first_splits, first_probs)
    assert entry.captured_call is not None
    assert len(entry.captured_call.sample_tensors) == 3
    manager.capture()
    module.zero_grad(set_to_none=True)

    graph_x = torch.randn(768, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    graph_probs = torch.randn(768, device=device, dtype=torch.bfloat16, requires_grad=True)
    changed_splits = torch.tensor([512, 256], dtype=torch.int64, device=device)
    graph_output = module(graph_x, changed_splits, graph_probs, changed_splits, graph_probs)
    graph_output.float().square().mean().backward()
    saved_graph_output = graph_output.detach().clone()
    saved_graph_x_grad = graph_x.grad.detach().clone()
    saved_graph_probs_grad = graph_probs.grad.detach().clone()
    saved_graph_param_grads = [parameter.grad.detach().clone() for parameter in module.parameters()]
    module.zero_grad(set_to_none=True)

    ref_x = graph_x.detach().clone().requires_grad_(True)
    ref_probs = graph_probs.detach().clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=recipe):
        ref_output = entry.original_forward(ref_x, changed_splits, ref_probs, changed_splits, ref_probs)
    ref_output.float().square().mean().backward()

    torch.testing.assert_close(saved_graph_output, ref_output, rtol=2.5e-1, atol=5e-1)
    torch.testing.assert_close(saved_graph_x_grad, ref_x.grad, rtol=2.5e-1, atol=5e-1)
    torch.testing.assert_close(saved_graph_probs_grad, ref_probs.grad, rtol=0, atol=0)
    for graph_grad, parameter in zip(saved_graph_param_grads, module.parameters()):
        torch.testing.assert_close(graph_grad, parameter.grad, rtol=0, atol=0)
    assert manager.stats() == {"captured": 1, "replayed": 1, "fallback": 0}
