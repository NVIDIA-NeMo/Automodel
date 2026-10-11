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

"""Model-owned CP scope for HybridEP input extents, without CUDA kernels."""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import nemo_automodel.components.models.step5.model as model_module
import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a
import nemo_automodel.components.moe.megatron.token_dispatcher as dispatcher_module
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    shard_batch_contiguous,
)
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer
from nemo_automodel.components.distributed.pipelining.runtime import PipelineRuntimeInitializer
from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.parallelizer import Step5Parallelizer
from nemo_automodel.components.moe.megatron.token_dispatcher import HybridEPPipelineRuntimeInitializer, _HybridEPManager
from tests.unit_tests.models.step5.test_model import make_model


class _CPMesh:
    def __init__(self, rank: int = 0) -> None:
        self.rank = rank

    def size(self) -> int:
        return 2

    def get_local_rank(self) -> int:
        return self.rank

    def get_group(self) -> None:
        return None


class _Provider(nn.Module):
    def __init__(self, initializer: PipelineRuntimeInitializer) -> None:
        super().__init__()
        self.initializer = initializer

    def get_pipeline_runtime_initializers(self) -> tuple[PipelineRuntimeInitializer, ...]:
        return (self.initializer,)


@pytest.fixture
def runtime(monkeypatch):
    fused_a2a.reset_hybrid_ep_buffer()
    monkeypatch.setattr(dispatcher_module, "hybrid_ep_dispatch", Mock())
    manager = _HybridEPManager(group=None, num_local_experts=2, num_experts=4, router_topk=2)
    initialize = Mock()
    monkeypatch.setattr(manager, "initialize_runtime", initialize)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    initializer = HybridEPPipelineRuntimeInitializer(manager, 4, torch.float32)
    yield manager, initializer, initialize
    fused_a2a.reset_hybrid_ep_buffer()


def _model(initializers=()):
    model = make_model(num_hidden_layers=1, moe_layer_list=[])
    model.cp_runtime_initializers = tuple(initializers)
    return model


def _batch(*, batch_size: int = 2, seq_len: int = 7, all_padding: bool = False, embeds: bool = False):
    ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    batch = {
        "labels": ids.clone(),
        "position_ids": torch.arange(seq_len).expand(batch_size, -1),
        "_packed_seq_ids": torch.ones_like(ids),
        "padding_mask": torch.full_like(ids, all_padding, dtype=torch.bool),
    }
    if embeds:
        batch["inputs_embeds"] = ids.float().unsqueeze(-1).expand(-1, -1, 4).contiguous()
    else:
        batch["input_ids"] = ids
    return batch


def _shard(model, batch: dict[str, torch.Tensor], *, rank: int = 0):
    """Invoke the model's real hook and contiguous sharder.

    Args:
        model: Tiny model with the original CP hook.
        batch: Token tensors [batch, global_sequence] or embeddings
            [batch, global_sequence, hidden], mutated by the sharder.
        rank: Local CP rank in a two-rank mesh.

    Returns:
        Context factory, local batch [batch, local_sequence, ...] and layout.
    """
    prepared = model.prepare_model_inputs_for_cp(batch)
    batch.update({key: value for key, value in prepared.items() if key != "cp_sharder"})
    return prepared["cp_sharder"].shard_batch(_CPMesh(rank), None, batch)


def test_config_default_and_strict_validation():
    assert Step5BackendConfig().prepare_hybridep_extent is False
    backend = Step5BackendConfig(dispatcher="hybridep", prepare_hybridep_extent=True)
    assert replace(backend, gate_precision=torch.float32).prepare_hybridep_extent
    for value in (1, 0, "true", None):
        with pytest.raises(TypeError, match="must be a bool"):
            Step5BackendConfig(prepare_hybridep_extent=value)
    with pytest.raises(ValueError, match="shared HybridEP"):
        Step5BackendConfig(dispatcher="torch", prepare_hybridep_extent=True)
    with pytest.raises(ValueError, match="shared HybridEP"):
        Step5BackendConfig(dispatcher="hybridep", dispatcher_share_token_dispatcher=False, prepare_hybridep_extent=True)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("embeds,all_padding", [(False, False), (True, False), (False, True)])
def test_enabled_sharder_preserves_every_tensor_and_layout(runtime, rank: int, embeds: bool, all_padding: bool):
    manager, initializer, initialize = runtime
    batch = _batch(embeds=embeds, all_padding=all_padding)
    _, expected, expected_layout = shard_batch_contiguous(
        _CPMesh(rank),
        None,
        {key: value.clone() for key, value in batch.items()},
        extra_seq_keys={"_packed_seq_ids": 1},
    )
    context, actual, layout = _shard(_model([initializer]), batch, rank=rank)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert layout == expected_layout
    assert initialize.call_count == 0
    with context():
        assert manager._pipeline_extent.num_tokens == 8  # B2 times four physical rows, even all-padding.
    assert manager._pipeline_extent is None
    assert initialize.call_count == 1


def test_zero_sized_local_batch_still_negotiates_a_positive_bound(runtime, monkeypatch):
    manager, initializer, _ = runtime
    seen = []
    monkeypatch.setattr(initializer, "_resolve_group_tokens", lambda **kw: seen.append(kw["num_tokens"]) or 9)
    context, batch, _ = _shard(_model([initializer]), _batch(batch_size=0))
    assert batch["input_ids"].shape == (0, 4)
    with context():
        assert manager._pipeline_extent.num_tokens == 12
    assert seen == [1]
    assert manager._pipeline_extent is None


def test_repeated_ga_eval_and_changed_peer_extent_always_negotiate(runtime, monkeypatch):
    manager, initializer, initialize = runtime
    model = _model([initializer])
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    peer_sizes = iter([8, 21, 9, 8])
    observed = []

    def group_max(count: torch.Tensor, *, op, group) -> None:
        """Update scalar int64 count [] with an independently supplied EP maximum."""
        assert count.dtype == torch.int64 and count.ndim == 0
        observed.append(count.item())
        count.fill_(max(int(count.item()), next(peer_sizes)))

    monkeypatch.setattr(torch.distributed, "all_reduce", group_max)
    extents = []
    for training, seq_len in [(True, 7), (True, 7), (False, 9), (True, 3)]:
        model.train(training)
        context, _, _ = _shard(model, _batch(seq_len=seq_len))
        with context():
            extents.append(manager._pipeline_extent.num_tokens)
        assert manager._pipeline_extent is None
    assert observed == [8, 8, 12, 4]
    assert extents == [8, 24, 12, 8]
    assert initialize.call_count == 1


def test_context_composes_original_scope_and_cleans_up_on_exception(runtime, monkeypatch):
    manager, initializer, _ = runtime
    events = []
    original_sharder = model_module.shard_batch_contiguous

    @contextmanager
    def original_context():
        events.append("enter_original")
        try:
            yield
        finally:
            assert manager._pipeline_extent is None
            events.append("exit_original")

    def nontrivial_sharder(cp_mesh, tp_mesh, batch: dict[str, torch.Tensor], **kwargs):
        """Retain the original sharder with a non-null context.

        Args:
            cp_mesh: Test CP mesh.
            tp_mesh: Unchanged optional mesh.
            batch: Tensor mapping [batch, global_sequence, ...], mutated in place.
            **kwargs: Original padding/loss-mask arguments; a loss mask has shape
                [batch, global_sequence].

        Returns:
            Non-null context, local batch [batch, local_sequence, ...] and layout.
        """
        _, batch, layout = original_sharder(cp_mesh, tp_mesh, batch, **kwargs)
        return original_context, batch, layout

    monkeypatch.setattr(model_module, "shard_batch_contiguous", nontrivial_sharder)
    context, _, _ = _shard(_model([initializer]), _batch())
    with pytest.raises(RuntimeError, match="forward failed"):
        with context():
            events.append("body")
            assert manager._pipeline_extent is not None
            raise RuntimeError("forward failed")
    assert events == ["enter_original", "body", "exit_original"]
    assert manager._pipeline_extent is None
    with pytest.raises(RuntimeError, match="prepare must run"):
        with initializer.step_context():
            pass
    with context():
        assert manager._pipeline_extent is not None
    assert events[-2:] == ["enter_original", "exit_original"]


def test_nested_context_rejected_and_failed_prepare_leaves_no_active_extent(runtime, monkeypatch):
    manager, initializer, _ = runtime
    context, _, _ = _shard(_model([initializer]), _batch())
    with context():
        with pytest.raises(RuntimeError, match="Cannot prepare"):
            with context():
                pass
        assert manager._pipeline_extent is not None
    assert manager._pipeline_extent is None
    fused_a2a.reset_hybrid_ep_buffer()
    monkeypatch.setattr(manager, "initialize_runtime", Mock(side_effect=ValueError("initialization failed")))
    with pytest.raises(ValueError, match="initialization failed"):
        with context():
            pass
    assert manager._pipeline_extent is None
    with pytest.raises(RuntimeError, match="prepare must run"):
        with initializer.step_context():
            pass


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_context_covers_backward_recompute_and_preserves_gradients(runtime, use_checkpoint: bool):
    manager, initializer, _ = runtime
    context, _, _ = _shard(_model([initializer]), _batch())
    calls = []

    def layer(hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate within the active runtime extent.

        Args:
            hidden: FP32 tensor [batch, local_sequence, hidden].

        Returns:
            Tensor with the same shape and FP32 dtype.
        """
        assert manager._pipeline_extent is not None
        calls.append(tuple(hidden.shape))
        return hidden.sin() * hidden

    torch.manual_seed(1)
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    expected = hidden.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(hidden)
    (expected.sin() * expected).backward(upstream)
    with context():
        output = checkpoint(layer, hidden, use_reentrant=False) if use_checkpoint else layer(hidden)
        output.backward(upstream)
    assert len(calls) == (2 if use_checkpoint else 1)
    torch.testing.assert_close(hidden.grad, expected.grad, rtol=0, atol=0)
    assert manager._pipeline_extent is None
    # This negative control proves that ending the scope before AC backward is invalid.
    if use_checkpoint:
        hidden = hidden.detach().requires_grad_(True)
        with context():
            output = checkpoint(layer, hidden, use_reentrant=False)
        with pytest.raises(AssertionError):
            output.backward(upstream)


@pytest.mark.parametrize(
    "enabled,pp,cp,ep,expected",
    [
        (False, 1, 2, 2, 0),
        (True, 2, 2, 2, 0),
        (True, 1, 1, 2, 0),
        (True, 1, 2, 1, 0),
        (True, 1, 2, 2, 1),
    ],
)
def test_parallelizer_collects_only_supported_opt_in_layouts(runtime, monkeypatch, enabled, pp, cp, ep, expected):
    _, initializer, _ = runtime
    model = _model()
    model.backend = Step5BackendConfig(
        attn="sdpa", linear="torch", experts="torch_mm", dispatcher="hybridep", prepare_hybridep_extent=enabled
    )
    model.providers = nn.ModuleList([_Provider(initializer), _Provider(initializer)])
    monkeypatch.setattr(ModelParallelizer, "parallelize", lambda self, model, mesh_context: model)
    mesh = Mock(spec=MeshContext, pp_size=pp, cp_size=cp, ep_size=ep)
    returned = Step5Parallelizer().parallelize(model, mesh)
    assert returned is model
    assert len(model.cp_runtime_initializers) == expected
    if expected:
        assert model.cp_runtime_initializers[0] is initializer
    else:
        hook = model.prepare_model_inputs_for_cp(_batch())["cp_sharder"]
        assert hook.shard_batch.func is shard_batch_contiguous


def test_parallelizer_rejects_incompatible_shared_resources(runtime, monkeypatch):
    manager, initializer, _ = runtime
    incompatible = HybridEPPipelineRuntimeInitializer(manager, 8, torch.float32)
    model = _model()
    model.backend = Step5BackendConfig(dispatcher="hybridep", prepare_hybridep_extent=True)
    model.providers = nn.ModuleList([_Provider(initializer), _Provider(incompatible)])
    monkeypatch.setattr(ModelParallelizer, "parallelize", lambda self, model, mesh_context: model)
    mesh = Mock(spec=MeshContext, pp_size=1, cp_size=2, ep_size=2)
    with pytest.raises(RuntimeError, match="incompatible signatures"):
        Step5Parallelizer().parallelize(model, mesh)
    assert model.cp_runtime_initializers == ()


def test_disabled_native_cp1_dispatch_never_invokes_model_hook(monkeypatch):
    model = _model()
    hook = Mock(side_effect=AssertionError("CP1 hook must not run"))
    monkeypatch.setattr(model, "prepare_model_inputs_for_cp", hook)
    batch = _batch()
    sharder = ContextParallelSharder(model=model, device_mesh=None, batch=batch)
    context, returned = sharder.shard(batch)
    with context():
        assert returned is batch
    hook.assert_not_called()
