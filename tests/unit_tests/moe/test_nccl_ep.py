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

"""CPU contract tests; the emulated transport does not validate NCCL kernels."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nemo_automodel.components.moe.nccl_ep as nccl_ep
from nemo_automodel.components.models.common.utils import BackendConfig, NcclEpConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP, _has_dispatched_rows
from nemo_automodel.components.moe.layers import MoE

_ORDERED_LAYOUT_ENV = "NCCL_EP_HT_EM_AG_SCAN_MODE"


class _Dispatch(torch.autograd.Function):
    """Emulated TE dispatch whose routing handle rides on ctx, outside saved tensors."""

    @staticmethod
    def forward(ctx, tokens, weights, buffer):
        """Gather routed rows in the buffer's layout.

        Args:
            tokens: Tensor of shape [tokens, hidden].
            weights: Tensor of shape [tokens, top_k].
            buffer: Routing state whose ``rows`` and ``slots`` have shape [received_tokens].

        Returns:
            Received rows [received_tokens, hidden] and their weights [received_tokens].
        """
        ctx.buffer = buffer
        ctx.shapes = tokens.shape, weights.shape
        return tokens[buffer.rows], weights[buffer.rows, buffer.slots]

    @staticmethod
    def backward(ctx, grad_received, grad_weights):
        """Scatter received-row gradients back through the handle held on ctx.

        Args:
            grad_received: Tensor of shape [received_tokens, hidden].
            grad_weights: Tensor of shape [received_tokens].

        Returns:
            Gradients for tokens [tokens, hidden] and weights [tokens, top_k].
        """
        rows, slots = ctx.buffer.rows, ctx.buffer.slots
        token_shape, weight_shape = ctx.shapes
        grad_tokens = grad_received.new_zeros(token_shape).index_add(0, rows, grad_received)
        grad_topk = grad_weights.new_zeros(weight_shape).index_put((rows, slots), grad_weights, accumulate=True)
        return grad_tokens, grad_topk, None


class _Combine(torch.autograd.Function):
    """Emulated TE combine with the same ctx-held routing handle."""

    @staticmethod
    def forward(ctx, expert_out, buffer, num_local_tokens):
        """Sum received rows back into source tokens.

        Args:
            expert_out: Tensor of shape [received_tokens, hidden].
            buffer: Routing state from the matching dispatch.
            num_local_tokens: Rows of the result.

        Returns:
            Tensor of shape [num_local_tokens, hidden].
        """
        ctx.buffer = buffer
        return expert_out.new_zeros(num_local_tokens, expert_out.shape[1]).index_add(0, buffer.rows, expert_out)

    @staticmethod
    def backward(ctx, grad):
        """Gather source-token gradients into the layout of the handle held on ctx.

        Args:
            grad: Tensor of shape [num_local_tokens, hidden].

        Returns:
            Gradient for the received rows, [received_tokens, hidden].
        """
        return grad[ctx.buffer.rows], None, None


class _Transport:
    def __init__(self):
        self.bootstraps = []
        self.buffers = []
        self.finalized = 0
        self.ordered = False
        self.dispatches = 0

    def is_ep_bootstrapped(self):
        return False

    def ep_bootstrap(self, group, **kwargs):
        # NCCL-EP reads its layout option from the environment when the group is created.
        self.ordered = os.environ.get(_ORDERED_LAYOUT_ENV) == "1"
        self.bootstraps.append((group, kwargs))

    def ep_finalize(self):
        self.finalized += 1

    def EpBuffer(self, **kwargs):
        buffer = SimpleNamespace(**kwargs)
        self.buffers.append(buffer)
        return buffer

    def ep_dispatch(self, buffer, tokens, indices, weights):
        """Route CPU tensors to emulate the public TE contract.

        Without the ordered layout, rows within an expert change order between
        dispatches of identical routing, as with NCCL-EP's atomic slot assignment.

        Args:
            buffer: Per-call routing state.
            tokens: Tensor of shape [tokens, hidden], with at most capacity rows.
            indices: Tensor of shape [tokens, top_k], with -1 for masked routes.
            weights: Tensor of shape [tokens, top_k].

        Returns:
            Expert-packed activations [received_tokens, hidden], probabilities
            [received_tokens], and counts [local_experts].
        """
        assert tokens.shape[0] == indices.shape[0] == weights.shape[0] <= buffer.max_tokens_per_rank
        assert indices.dtype == torch.int64 and weights.dtype == torch.float32
        assert tokens.is_contiguous() and indices.is_contiguous() and weights.is_contiguous()
        rows, slots = (indices >= 0).nonzero(as_tuple=True)
        experts = indices[rows, slots]
        position = torch.arange(rows.numel())
        if not self.ordered and self.dispatches % 2:
            position = position.flip(0)
        self.dispatches += 1
        order = (experts * (rows.numel() + 1) + position).argsort()
        buffer.rows, buffer.slots = rows[order], slots[order]
        received, probs = _Dispatch.apply(tokens, weights, buffer)
        return received, probs, torch.bincount(experts, minlength=buffer.num_local_experts)

    def ep_combine(self, buffer, expert_out, *, num_local_tokens):
        """Sum packed expert outputs back into source rows.

        Args:
            buffer: Per-call routing state.
            expert_out: Tensor of shape [received_tokens, hidden].
            num_local_tokens: Rows of the dispatched input.

        Returns:
            Tensor of shape [num_local_tokens, hidden].
        """
        return _Combine.apply(expert_out, buffer, num_local_tokens)


@pytest.fixture
def transport(monkeypatch):
    api = _Transport()
    monkeypatch.setattr(nccl_ep, "_runtime", nccl_ep._Runtime())
    monkeypatch.setattr(nccl_ep, "safe_import", lambda name: (True, api))
    monkeypatch.setattr(nccl_ep, "safe_import_from", lambda module, name: (True, api.ep_bootstrap))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(nccl_ep.dist, "get_backend", lambda group: "nccl")
    yield api
    nccl_ep.finalize_nccl_ep()


@pytest.fixture
def group():
    return SimpleNamespace(size=lambda: 2)


def _manager(group, *, max_tokens=5, hidden_dim=4):
    return nccl_ep.NcclEpManager(
        NcclEpConfig(max_tokens_per_rank=max_tokens), group=group, hidden_dim=hidden_dim, num_experts=4, top_k=2
    )


def _roundtrip(manager, hidden, weights, indices, scales):
    """Apply scalar experts through the manager.

    Args:
        manager: Initialized communication manager.
        hidden: BF16 tensor of shape [tokens, hidden].
        weights: Floating tensor of shape [tokens, top_k].
        indices: Integer tensor of shape [tokens, top_k].
        scales: Trainable tensor of shape [local_experts].

    Returns:
        Combined tensor of shape [tokens, hidden].
    """
    manager.token_indices, manager.token_probs = indices, weights
    received = manager.dispatch(hidden)
    received, probs = manager.get_permuted_hidden_states_by_experts(received)
    counts = manager.get_number_of_tokens_per_expert()
    chunks = received.split(counts.tolist())
    expert_out = torch.cat([x * scale for x, scale in zip(chunks, scales)])
    expert_out = (expert_out.float() * probs[:, None]).to(hidden.dtype)
    return manager.combine(manager.get_restored_hidden_states_by_experts(expert_out))


def _reference(hidden, weights, indices, scales):
    """Evaluate the weighted experts directly, without routing buffers.

    Args:
        hidden: BF16 tensor of shape [tokens, hidden].
        weights: Tensor of shape [tokens, top_k].
        indices: Tensor of shape [tokens, top_k], with -1 for masked routes.
        scales: Tensor of shape [local_experts].

    Returns:
        Tensor of shape [tokens, hidden].
    """
    output = hidden * 0 + (weights.sum() + scales.sum()).to(hidden.dtype) * 0
    for expert in range(scales.numel()):
        contribution = hidden * scales[expert]
        prob = (weights * (indices == expert)).sum(-1)
        output = output + (contribution.float() * prob[:, None]).to(hidden.dtype)
    return output


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("tokens", [0, 3, 5])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_gradients_and_inflight_buffers(transport, group, checkpointed, tokens, weights_dtype):
    manager = _manager(group)
    torch.manual_seed(7)
    hidden = torch.randn(tokens, 4, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.tensor([[0.25, 0.75]] * tokens, dtype=weights_dtype).reshape(tokens, 2).requires_grad_()
    indices = torch.tensor([[1, 0]] * tokens, dtype=torch.int32).reshape(tokens, 2)
    if tokens:
        indices[0] = -1
    scales = torch.tensor([2.0, 4.0], dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = [x.detach().clone().requires_grad_() for x in (hidden, weights, scales)]
    actual = []
    expected = []
    for _ in range(2):
        actual.append(
            checkpoint(_roundtrip, manager, hidden, weights, indices, scales, use_reentrant=False)
            if checkpointed
            else _roundtrip(manager, hidden, weights, indices, scales)
        )
        expected.append(_reference(reference_inputs[0], reference_inputs[1], indices, reference_inputs[2]))
    upstream = torch.randn_like(hidden)
    sum((x.float() * upstream).sum() for x in actual).backward()
    sum((x.float() * upstream).sum() for x in expected).backward()
    for out, ref in zip(actual, expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    for value, ref in zip((hidden, weights, scales), reference_inputs):
        torch.testing.assert_close(value.grad, ref.grad, rtol=0.02, atol=0.0625)
    assert len({id(x) for x in transport.buffers}) == len(transport.buffers)
    assert len(transport.bootstraps) == 1
    # Checkpoint replay can stop before combine. A subsequent full forward must
    # replace that abandoned handle and release its own handle after combine.
    followup = _roundtrip(manager, hidden, weights, indices, scales)
    torch.testing.assert_close(followup, expected[0], rtol=0, atol=0)
    assert manager._buffer is None


def test_context_reuse_rejection_and_cleanup(transport, group):
    first = _manager(group)
    second = _manager(group)
    assert first is not second
    assert len(transport.bootstraps) == 1
    assert transport.bootstraps[0][1]["max_tokens_per_rank"] == 64
    assert transport.bootstraps[0][1]["recv_capacity_per_rank"] is None
    with pytest.raises(ValueError, match="one EP group"):
        _manager(SimpleNamespace(size=lambda: 2))
    with pytest.raises(ValueError, match="one EP group"):
        _manager(group, hidden_dim=8)
    nccl_ep.finalize_nccl_ep()
    nccl_ep.finalize_nccl_ep()
    assert transport.finalized == 1
    _manager(group, hidden_dim=8)
    assert len(transport.bootstraps) == 2


@pytest.mark.parametrize("previous", [None, "0"])
def test_group_uses_ordered_layout_and_automatic_sms(transport, group, monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv(_ORDERED_LAYOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(_ORDERED_LAYOUT_ENV, previous)
    _manager(group)
    assert transport.ordered
    assert transport.bootstraps[0][1]["max_num_sms"] == 0
    assert os.environ.get(_ORDERED_LAYOUT_ENV) == previous


@pytest.mark.parametrize("rows", [0, 3])
def test_nccl_ep_row_check_reads_the_host_extent(rows):
    # A meta tensor cannot be read on the host: the NCCL-EP check must not touch the counts.
    counts = torch.empty(2, dtype=torch.int64, device="meta")
    received = torch.empty(rows, 4)
    assert _has_dispatched_rows(received, counts, never_empty=False, dispatcher_backend="nccl_ep") == (rows > 0)
    assert _has_dispatched_rows(received, counts, never_empty=True, dispatcher_backend="deepep")
    with pytest.raises(RuntimeError):
        _has_dispatched_rows(received, counts, never_empty=False, dispatcher_backend="deepep")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "max_tokens_per_rank"),
        ({"nccl_ep": {"max_tokens_per_rank": 0}}, "positive"),
        ({"nccl_ep": {"max_tokens_per_rank": 5}, "dispatcher_async_dispatch": True}, "async_dispatch"),
    ],
)
def test_invalid_backend_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BackendConfig(dispatcher="nccl_ep", **kwargs)


def test_unavailable_dependency_is_actionable(monkeypatch, group):
    monkeypatch.setattr(nccl_ep, "safe_import", lambda name: (False, None))
    with pytest.raises(ImportError, match="default public-wheel"):
        _manager(group)


def test_disabled_native_binding_is_actionable(transport, group, monkeypatch):
    monkeypatch.setattr(nccl_ep, "safe_import_from", lambda module, name: (False, None))
    with pytest.raises(ImportError, match="default public-wheel"):
        _manager(group)
    assert not transport.bootstraps


def test_capture_overflow_and_dtype_fail_before_dispatch(transport, group, monkeypatch):
    manager = _manager(group)
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        manager.dispatch(torch.zeros(6, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16"):
        manager.dispatch(torch.zeros(3, 4))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA Graph"):
        manager.dispatch(torch.zeros(3, 4, dtype=torch.bfloat16))
    assert not transport.buffers


def test_production_expert_selection_and_dispatch(transport, group, monkeypatch):
    backend = BackendConfig(
        dispatcher="nccl_ep", experts="torch_mm", nccl_ep={"max_tokens_per_rank": 5}, attn="sdpa", linear="torch"
    )
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=4,
        inter_dim=8,
        moe_inter_dim=8,
        norm_topk_prob=False,
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    moe = MoE(config, backend)
    assert isinstance(moe.experts, GroupedExpertsDeepEP)
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 0, get_group=lambda: group)
    moe.experts.init_token_dispatcher(mesh)
    dispatcher = moe.experts.token_dispatcher
    hidden = torch.ones(3, 4, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.tensor([[0.25, 0.75]] * 3, requires_grad=True)
    indices = torch.tensor([[0, 1]] * 3)
    received, counts, weights = dispatcher.token_permutation2(hidden, 3, probs, indices)
    assert counts.tolist() == [3, 3]
    out = dispatcher.token_unpermutation((received.float() * weights[:, None]).bfloat16())
    torch.testing.assert_close(out, hidden)
    out.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 4))
    assert dispatcher.get_pipeline_runtime_initializers(hidden_dim=4, dtype=torch.bfloat16) == ()


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("expert_bias", [False, True])
def test_empty_experts_keep_dispatch_backward_edge(tokens, lora, packed, expert_bias):
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=64,
        moe_inter_dim=64,
        norm_topk_prob=False,
        expert_bias=expert_bias,
    )
    experts = GroupedExpertsDeepEP(config)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.fill_(0.125)
    if packed and lora:
        from nemo_automodel.components._peft.lora_experts_mxfp4 import GroupedExpertsDeepEPLoRAMXFP4

        experts = GroupedExpertsDeepEPLoRAMXFP4(experts, lora_dim=2)
    elif packed:
        from nemo_automodel.components.moe.quantized_experts import GroupedExpertsDeepEPMXFP4

        experts = GroupedExpertsDeepEPMXFP4(experts)
    elif lora:
        from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA

        experts = GroupedExpertsDeepEPLoRA(experts, lora_dim=2)
    experts.ep_size = 2
    experts.n_routed_experts = 4
    # The production sharder supplies DTensors. These local weight wrappers let
    # the CPU test isolate empty-expert autograd without initializing a process group.
    num_local_experts = config.n_routed_experts // experts.ep_size
    for parameter in experts.parameters():
        parameter.to_local = lambda parameter=parameter: parameter[:num_local_experts]
        if parameter.requires_grad:
            with torch.no_grad():
                parameter.fill_(0.125)
    received = torch.empty(0, config.dim, dtype=torch.float32, requires_grad=True)
    probs = torch.empty(0, dtype=torch.float32, requires_grad=True)

    def permutation(**kwargs):
        """Return empty per-rank activations [0, hidden] and probabilities [0]."""
        return received, torch.zeros(2, dtype=torch.int64), probs

    def combine(output):
        """Produce local output while preserving the zero-row expert graph.

        Args:
            output: Tensor of shape [0, hidden].

        Returns:
            Tensor of shape [tokens, hidden].
        """
        return output.new_zeros(tokens, config.dim) + output.sum() * 0

    experts.token_dispatcher = SimpleNamespace(token_permutation2=permutation, token_unpermutation=combine)
    out = experts(
        torch.ones(tokens, config.dim),
        torch.zeros(tokens, dtype=torch.bool),
        torch.ones(tokens, 2, requires_grad=True),
        torch.zeros(tokens, 2, dtype=torch.int64),
    )
    out.sum().backward()
    assert received.grad is not None
    assert probs.grad is not None
    for parameter in experts.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0
    trainable = [parameter for parameter in experts.parameters() if parameter.requires_grad]
    if trainable:
        expected = [parameter.detach().clone() * 0.99 for parameter in trainable]
        torch.optim.SGD(trainable, lr=0.1, weight_decay=0.1).step()
        for parameter, reference in zip(trainable, expected):
            torch.testing.assert_close(parameter, reference)


@pytest.mark.parametrize("experts", ["torch_mm", "te"])
def test_backend_preserves_explicit_selection(experts):
    config = BackendConfig(dispatcher="nccl_ep", experts=experts, nccl_ep={"max_tokens_per_rank": 17})
    assert config.dispatcher == "nccl_ep"
    assert config.experts == experts
    assert isinstance(config.nccl_ep, NcclEpConfig)
    assert config.nccl_ep.max_tokens_per_rank == 17


def test_lora_preserves_nccl_ep_capacity():
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA

    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=4,
        inter_dim=8,
        moe_inter_dim=8,
        norm_topk_prob=False,
    )
    backend = BackendConfig(dispatcher="nccl_ep", experts="torch_mm", nccl_ep={"max_tokens_per_rank": 17})
    original = GroupedExpertsDeepEP(config, backend=backend, dispatcher_backend=backend.dispatcher)
    wrapped = GroupedExpertsDeepEPLoRA(original, lora_dim=2)
    assert wrapped.nccl_ep is original.nccl_ep
    assert wrapped.dispatcher_backend == "nccl_ep"
