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

"""Optimizer-level parity tests for benchmark partial CUDA graphs."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import nemo_automodel.recipes.llm.partial_cuda_graphs as partial_graphs


class _AttentionTransform(nn.Module):
    """Small parameter-free stand-in for the graphable fused-attention boundary."""

    def forward(self, x, *, fp8=False, fp8_meta=None, quantizers=None):
        assert fp8 is False
        assert fp8_meta is None
        assert quantizers is None
        return torch.tanh(x) + x * 0.125


class _DynamicExpert(nn.Module):
    """Differentiable expert stand-in whose output depends on dynamic split contents."""

    def __init__(self, feature_size: int, *, device: torch.device | str, dtype: torch.dtype):
        super().__init__()
        weight = torch.linspace(0.2, 0.8, feature_size, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.tensor(-0.15, device=device, dtype=dtype))
        self.register_buffer("split_coefficients", torch.tensor([1.0, 2.0], device=device, dtype=dtype))
        self.alias_checks = 0

    def forward(self, x, splits, probs, splits_alias, probs_alias):
        assert splits is splits_alias
        assert probs is probs_alias
        self.alias_checks += 1
        split_signal = torch.sum(splits.to(x.dtype) * self.split_coefficients)
        gate = (probs + probs_alias).reshape(x.shape[0], -1).mean(dim=-1, keepdim=True)
        return x * self.weight + gate * split_signal + self.bias


class _ConvergenceModel(nn.Module):
    """Keep routing eager while exposing attention and expert graph boundaries."""

    def __init__(
        self,
        feature_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        expert: nn.Module | None = None,
    ):
        super().__init__()
        self.attention = _AttentionTransform()
        self.expert = expert or _DynamicExpert(feature_size, device=device, dtype=dtype)
        self.router_gain = nn.Parameter(torch.tensor(0.35, device=device, dtype=dtype))

    def forward(self, attention_x, expert_x, router_logits, splits):
        attention_output = self.attention(
            attention_x,
            fp8=False,
            fp8_meta=None,
            quantizers=None,
        )
        probs = torch.sigmoid(router_logits * self.router_gain).to(expert_x.dtype)
        expert_output = self.expert(expert_x, splits, probs, splits, probs)
        return attention_output, expert_output, probs


class _CheckpointConvergenceModel(_ConvergenceModel):
    """Run the graph boundaries inside a PyTorch activation checkpoint."""

    def __init__(self, *args, use_reentrant: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reentrant = use_reentrant
        self.checkpoint_phase = "outside"
        self.checkpoint_phases = []
        self.probability_gradient_events = []

    @contextmanager
    def _phase(self, phase: str):
        previous_phase = self.checkpoint_phase
        self.checkpoint_phase = phase
        try:
            yield
        finally:
            self.checkpoint_phase = previous_phase

    def _checkpoint_body(self, attention_x, expert_x, router_logits, splits):
        if self.use_reentrant:
            phase = "recompute" if torch.is_grad_enabled() else "forward"
        else:
            phase = self.checkpoint_phase
        self.checkpoint_phases.append(phase)

        # Real Q/K/V, dispatched hidden states, and router logits are produced inside the
        # checkpointed attention/MLP. Reentrant checkpointing therefore sees no-grad tensors
        # in the original forward and grad-enabled tensors during recompute.
        attention_x = attention_x * 1.0
        expert_x = expert_x * 1.0
        router_logits = router_logits * 1.0
        outputs = super().forward(attention_x, expert_x, router_logits, splits)
        probs = outputs[2]
        if probs.requires_grad:
            probs.register_hook(
                lambda grad, phase=phase: self.probability_gradient_events.append((phase, grad.detach().clone()))
            )
        return outputs

    def checkpointed_forward(self, attention_x, expert_x, router_logits, splits):
        checkpoint_kwargs = {"use_reentrant": self.use_reentrant}
        if not self.use_reentrant:
            checkpoint_kwargs["context_fn"] = lambda: (self._phase("forward"), self._phase("recompute"))
        return checkpoint(
            self._checkpoint_body,
            attention_x,
            expert_x,
            router_logits,
            splits,
            **checkpoint_kwargs,
        )


def _install_fake_graph(monkeypatch, capture_callback=None):
    class _FakeGraphedAdapter(nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self._captured_call = adapter.captured_call
            self._target_forward = adapter.target.forward

        def forward(self, *tensor_inputs):
            args, kwargs = self._captured_call.rebuild(tensor_inputs)
            return self._target_forward(*args, **kwargs)

    def fake_make_graphed_callables(modules, sample_args, **_kwargs):
        if capture_callback is not None:
            capture_callback()
        for args in sample_args:
            assert len(args) == len({id(tensor) for tensor in args})
        return tuple(_FakeGraphedAdapter(module) for module in modules)

    monkeypatch.setattr(partial_graphs, "_get_make_graphed_callables", lambda: fake_make_graphed_callables)


def _make_manager(
    model: _ConvergenceModel,
    *,
    expert_fp8_enabled: bool = False,
    fp8_recipe=None,
) -> partial_graphs.PartialCudaGraphManager:
    attention_entry = partial_graphs._PartialGraphEntry(
        name="test.attention",
        target=model.attention,
        fp8_enabled=False,
        canonicalizer=partial_graphs._canonicalize_bf16_fused_attention,
    )
    expert_entry = partial_graphs._PartialGraphEntry(
        name="test.experts",
        target=model.expert,
        fp8_enabled=expert_fp8_enabled,
        canonicalizer=partial_graphs._canonicalize_te_ops_experts,
    )
    manager = partial_graphs.PartialCudaGraphManager([attention_entry, expert_entry], fp8_recipe=fp8_recipe)
    manager.start_recording()
    return manager


def _leaf_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _make_batch(
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    feature_size: int,
    token_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    attention_x = torch.randn(4, feature_size, device=device, dtype=dtype, generator=generator)
    expert_x = torch.randn(token_count, feature_size, device=device, dtype=dtype, generator=generator)
    router_logits = torch.randn(token_count, device=device, dtype=dtype, generator=generator)
    return attention_x, expert_x, router_logits


def _assert_model_parity(
    graph_model: nn.Module,
    reference_model: nn.Module,
    *,
    rtol: float,
    atol: float,
) -> None:
    graph_parameters = dict(graph_model.named_parameters())
    reference_parameters = dict(reference_model.named_parameters())
    assert graph_parameters.keys() == reference_parameters.keys()
    for name, graph_parameter in graph_parameters.items():
        reference_parameter = reference_parameters[name]
        torch.testing.assert_close(graph_parameter, reference_parameter, rtol=rtol, atol=atol)
        if graph_parameter.grad is None or reference_parameter.grad is None:
            assert graph_parameter.grad is reference_parameter.grad
        else:
            torch.testing.assert_close(graph_parameter.grad, reference_parameter.grad, rtol=rtol, atol=atol)


def _assert_optimizer_parity(
    graph_optimizer: torch.optim.Optimizer,
    reference_optimizer: torch.optim.Optimizer,
    graph_model: nn.Module,
    reference_model: nn.Module,
    *,
    rtol: float,
    atol: float,
) -> None:
    graph_parameters = dict(graph_model.named_parameters())
    reference_parameters = dict(reference_model.named_parameters())
    for name, graph_parameter in graph_parameters.items():
        graph_state = graph_optimizer.state.get(graph_parameter, {})
        reference_state = reference_optimizer.state.get(reference_parameters[name], {})
        assert graph_state.keys() == reference_state.keys()
        for key, graph_value in graph_state.items():
            reference_value = reference_state[key]
            if isinstance(graph_value, torch.Tensor):
                torch.testing.assert_close(graph_value, reference_value, rtol=rtol, atol=atol)
            else:
                assert graph_value == reference_value


def _run_optimizer_steps(
    graph_model: _ConvergenceModel,
    reference_model: _ConvergenceModel,
    graph_optimizer: torch.optim.Optimizer,
    reference_optimizer: torch.optim.Optimizer,
    step_splits: Sequence[Sequence[Sequence[int]]],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    feature_size: int,
    rtol: float = 0,
    atol: float = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    graph_loss_sequence = []
    reference_loss_sequence = []
    seed = 100

    for microbatch_splits in step_splits:
        graph_optimizer.zero_grad(set_to_none=False)
        reference_optimizer.zero_grad(set_to_none=False)
        graph_step_loss = torch.zeros((), device=device, dtype=torch.float64)
        reference_step_loss = torch.zeros((), device=device, dtype=torch.float64)

        for split_values in microbatch_splits:
            token_count = sum(split_values)
            base_attention_x, base_expert_x, base_router_logits = _make_batch(
                device=device,
                dtype=dtype,
                feature_size=feature_size,
                token_count=token_count,
                seed=seed,
            )
            seed += 1
            splits = torch.tensor(split_values, device=device, dtype=torch.int64)

            graph_attention_x = _leaf_clone(base_attention_x)
            graph_expert_x = _leaf_clone(base_expert_x)
            graph_router_logits = _leaf_clone(base_router_logits)
            graph_attention, graph_expert, graph_probs = graph_model(
                graph_attention_x,
                graph_expert_x,
                graph_router_logits,
                splits,
            )
            graph_probs.retain_grad()
            graph_loss = graph_attention.float().square().mean() + graph_expert.float().square().mean()
            (graph_loss / len(microbatch_splits)).backward()

            reference_attention_x = _leaf_clone(base_attention_x)
            reference_expert_x = _leaf_clone(base_expert_x)
            reference_router_logits = _leaf_clone(base_router_logits)
            reference_attention, reference_expert, reference_probs = reference_model(
                reference_attention_x,
                reference_expert_x,
                reference_router_logits,
                splits,
            )
            reference_probs.retain_grad()
            reference_loss = reference_attention.float().square().mean() + reference_expert.float().square().mean()
            (reference_loss / len(microbatch_splits)).backward()

            torch.testing.assert_close(graph_attention, reference_attention, rtol=rtol, atol=atol)
            torch.testing.assert_close(graph_expert, reference_expert, rtol=rtol, atol=atol)
            torch.testing.assert_close(graph_loss, reference_loss, rtol=rtol, atol=atol)
            torch.testing.assert_close(graph_attention_x.grad, reference_attention_x.grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(graph_expert_x.grad, reference_expert_x.grad, rtol=rtol, atol=atol)
            torch.testing.assert_close(
                graph_router_logits.grad,
                reference_router_logits.grad,
                rtol=rtol,
                atol=atol,
            )
            torch.testing.assert_close(graph_probs.grad, reference_probs.grad, rtol=rtol, atol=atol)
            _assert_model_parity(graph_model, reference_model, rtol=rtol, atol=atol)

            graph_step_loss += graph_loss.detach().to(torch.float64)
            reference_step_loss += reference_loss.detach().to(torch.float64)

        graph_optimizer.step()
        reference_optimizer.step()
        _assert_model_parity(graph_model, reference_model, rtol=rtol, atol=atol)
        _assert_optimizer_parity(
            graph_optimizer,
            reference_optimizer,
            graph_model,
            reference_model,
            rtol=rtol,
            atol=atol,
        )
        graph_loss_sequence.append(graph_step_loss.cpu())
        reference_loss_sequence.append(reference_step_loss.cpu())

    return graph_loss_sequence, reference_loss_sequence


def _run_checkpointed_microbatch(
    graph_model: _CheckpointConvergenceModel,
    reference_model: _ConvergenceModel,
    split_values: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    feature_size: int,
    seed: int,
    accumulation_steps: int,
    rtol: float,
    atol: float,
    context_factory,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = sum(split_values)
    base_attention_x, base_expert_x, base_router_logits = _make_batch(
        device=device,
        dtype=dtype,
        feature_size=feature_size,
        token_count=token_count,
        seed=seed,
    )
    splits = torch.tensor(split_values, device=device, dtype=torch.int64)

    with context_factory():
        graph_attention_x = _leaf_clone(base_attention_x)
        graph_expert_x = _leaf_clone(base_expert_x)
        graph_router_logits = _leaf_clone(base_router_logits)
        probability_event_count = len(graph_model.probability_gradient_events)
        graph_attention, graph_expert, graph_probs = graph_model.checkpointed_forward(
            graph_attention_x,
            graph_expert_x,
            graph_router_logits,
            splits,
        )
        graph_loss = (
            graph_attention.float().square().mean()
            + graph_expert.float().square().mean()
            + 0.05 * graph_probs.float().square().mean()
        )
        (graph_loss / accumulation_steps).backward()

    with context_factory():
        reference_attention_x = _leaf_clone(base_attention_x)
        reference_expert_x = _leaf_clone(base_expert_x)
        reference_router_logits = _leaf_clone(base_router_logits)
        reference_attention, reference_expert, reference_probs = reference_model(
            reference_attention_x,
            reference_expert_x,
            reference_router_logits,
            splits,
        )
        reference_probs.retain_grad()
        reference_loss = (
            reference_attention.float().square().mean()
            + reference_expert.float().square().mean()
            + 0.05 * reference_probs.float().square().mean()
        )
        (reference_loss / accumulation_steps).backward()

    torch.testing.assert_close(graph_attention, reference_attention, rtol=rtol, atol=atol)
    torch.testing.assert_close(graph_expert, reference_expert, rtol=rtol, atol=atol)
    torch.testing.assert_close(graph_probs, reference_probs, rtol=rtol, atol=atol)
    torch.testing.assert_close(graph_loss, reference_loss, rtol=rtol, atol=atol)
    torch.testing.assert_close(graph_attention_x.grad, reference_attention_x.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(graph_expert_x.grad, reference_expert_x.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(graph_router_logits.grad, reference_router_logits.grad, rtol=rtol, atol=atol)

    assert len(graph_model.probability_gradient_events) == probability_event_count + 1
    probability_phase, graph_probability_grad = graph_model.probability_gradient_events[-1]
    expected_probability_phase = "recompute" if graph_model.use_reentrant else "forward"
    assert probability_phase == expected_probability_phase
    torch.testing.assert_close(graph_probability_grad, reference_probs.grad, rtol=rtol, atol=atol)
    _assert_model_parity(graph_model, reference_model, rtol=rtol, atol=atol)
    return graph_loss.detach(), reference_loss.detach()


def _run_checkpointed_optimizer_steps(
    graph_model: _CheckpointConvergenceModel,
    reference_model: _ConvergenceModel,
    graph_optimizer: torch.optim.Optimizer,
    reference_optimizer: torch.optim.Optimizer,
    manager: partial_graphs.PartialCudaGraphManager,
    step_splits: Sequence[Sequence[Sequence[int]]],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    feature_size: int,
    rtol: float = 0,
    atol: float = 0,
    context_factory=nullcontext,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    graph_loss_sequence = []
    reference_loss_sequence = []
    seed = 500

    for step, microbatch_splits in enumerate(step_splits):
        graph_optimizer.zero_grad(set_to_none=False)
        reference_optimizer.zero_grad(set_to_none=False)
        graph_step_loss = torch.zeros((), device=device, dtype=torch.float64)
        reference_step_loss = torch.zeros((), device=device, dtype=torch.float64)

        for split_values in microbatch_splits:
            graph_loss, reference_loss = _run_checkpointed_microbatch(
                graph_model,
                reference_model,
                split_values,
                device=device,
                dtype=dtype,
                feature_size=feature_size,
                seed=seed,
                accumulation_steps=len(microbatch_splits),
                rtol=rtol,
                atol=atol,
                context_factory=context_factory,
            )
            seed += 1
            graph_step_loss += graph_loss.to(torch.float64)
            reference_step_loss += reference_loss.to(torch.float64)

        graph_optimizer.step()
        reference_optimizer.step()
        _assert_model_parity(graph_model, reference_model, rtol=rtol, atol=atol)
        _assert_optimizer_parity(
            graph_optimizer,
            reference_optimizer,
            graph_model,
            reference_model,
            rtol=rtol,
            atol=atol,
        )
        graph_loss_sequence.append(graph_step_loss.cpu())
        reference_loss_sequence.append(reference_step_loss.cpu())

        if step == 0:
            manager.capture()

    return graph_loss_sequence, reference_loss_sequence


def _capture_initial_sample(
    model: _ConvergenceModel,
    manager: partial_graphs.PartialCudaGraphManager,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    feature_size: int,
    token_count: int,
    split_values: Sequence[int],
) -> None:
    attention_x, expert_x, router_logits = _make_batch(
        device=device,
        dtype=dtype,
        feature_size=feature_size,
        token_count=token_count,
        seed=7,
    )
    attention_x.requires_grad_(True)
    expert_x.requires_grad_(True)
    router_logits.requires_grad_(True)
    splits = torch.tensor(split_values, device=device, dtype=torch.int64)
    model(attention_x, expert_x, router_logits, splits)
    expert_call = manager.entries[1].captured_call
    assert expert_call is not None
    assert expert_call.tensor_input_indices == (0, 1, 2, 1, 2)
    assert len(expert_call.sample_tensors) == 3
    manager.capture()


def test_multistep_ga_convergence_with_dynamic_splits_and_shape_fallback(monkeypatch):
    _install_fake_graph(monkeypatch)
    graph_model = _ConvergenceModel(3, device="cpu", dtype=torch.float64)
    reference_model = copy.deepcopy(graph_model)
    manager = _make_manager(graph_model)
    _capture_initial_sample(
        graph_model,
        manager,
        device="cpu",
        dtype=torch.float64,
        feature_size=3,
        token_count=4,
        split_values=[1, 3],
    )

    graph_optimizer = torch.optim.AdamW(graph_model.parameters(), lr=0.02, betas=(0.8, 0.9), weight_decay=0.01)
    reference_optimizer = torch.optim.AdamW(
        reference_model.parameters(),
        lr=0.02,
        betas=(0.8, 0.9),
        weight_decay=0.01,
    )
    graph_losses, reference_losses = _run_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        [([2, 2], [3, 1]), ([1, 3], [2, 3]), ([3, 1], [2, 2])],
        device="cpu",
        dtype=torch.float64,
        feature_size=3,
    )

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=0, atol=0)
    assert manager.entries[0].replay_count == 6
    assert manager.entries[0].fallback_count == 0
    assert manager.entries[1].replay_count == 5
    assert manager.entries[1].fallback_count == 1
    assert manager.stats() == {"captured": 2, "replayed": 11, "fallback": 1}
    assert graph_model.expert.alias_checks >= 7


@pytest.mark.parametrize("use_reentrant", [False, True], ids=["non_reentrant", "reentrant"])
def test_checkpoint_recompute_multistep_ga_parity_and_safe_capture(monkeypatch, use_reentrant):
    graph_model = _CheckpointConvergenceModel(
        3,
        device="cpu",
        dtype=torch.float64,
        use_reentrant=use_reentrant,
    )
    reference_model = _ConvergenceModel(3, device="cpu", dtype=torch.float64)
    reference_model.load_state_dict(graph_model.state_dict())
    manager = _make_manager(graph_model)
    capture_events = []

    def record_capture():
        capture_events.append(
            (
                graph_model.checkpoint_phase,
                graph_model.checkpoint_phases.count("forward"),
                graph_model.checkpoint_phases.count("recompute"),
            )
        )

    _install_fake_graph(monkeypatch, capture_callback=record_capture)
    graph_optimizer = torch.optim.AdamW(
        graph_model.parameters(),
        lr=0.02,
        betas=(0.8, 0.9),
        weight_decay=0.01,
    )
    reference_optimizer = torch.optim.AdamW(
        reference_model.parameters(),
        lr=0.02,
        betas=(0.8, 0.9),
        weight_decay=0.01,
    )
    graph_losses, reference_losses = _run_checkpointed_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        manager,
        [
            ([1, 3], [2, 2]),
            ([2, 2], [3, 1]),
            ([1, 3], [2, 3]),
            ([3, 1], [2, 2]),
        ],
        device="cpu",
        dtype=torch.float64,
        feature_size=3,
    )

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=0, atol=0)
    # Attention and expert entries use independent graph pools so either can
    # safely take its guarded eager fallback without violating replay order.
    assert capture_events == [("outside", 2, 2), ("outside", 2, 2)]
    assert graph_model.checkpoint_phases.count("forward") == 8
    assert graph_model.checkpoint_phases.count("recompute") == 8
    if use_reentrant:
        # The no-grad reentrant forward records internal attention/expert tensors with
        # requires_grad=False. Recompute correctly falls back when they require gradients.
        assert manager.entries[0].replay_count == 6
        assert manager.entries[0].fallback_count == 6
        assert manager.entries[1].replay_count == 5
        assert manager.entries[1].fallback_count == 7
        assert manager.stats() == {"captured": 2, "replayed": 11, "fallback": 13}
    else:
        assert manager.entries[0].replay_count == 12
        assert manager.entries[0].fallback_count == 0
        assert manager.entries[1].replay_count == 10
        assert manager.entries[1].fallback_count == 2
        assert manager.stats() == {"captured": 2, "replayed": 22, "fallback": 2}
    assert graph_model.expert.alias_checks >= 16


def test_empty_capture_iteration_skips_expert_but_attention_keeps_replaying(monkeypatch, caplog):
    _install_fake_graph(monkeypatch)
    graph_model = _ConvergenceModel(3, device="cpu", dtype=torch.float64)
    reference_model = copy.deepcopy(graph_model)
    manager = _make_manager(graph_model)

    empty_attention = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    reference_empty_attention = empty_attention.detach().clone().requires_grad_(True)
    graph_empty_output = graph_model.attention(
        empty_attention,
        fp8=False,
        fp8_meta=None,
        quantizers=None,
    )
    reference_empty_output = reference_model.attention(
        reference_empty_attention,
        fp8=False,
        fp8_meta=None,
        quantizers=None,
    )
    graph_empty_output.square().mean().backward()
    reference_empty_output.square().mean().backward()
    torch.testing.assert_close(graph_empty_output, reference_empty_output, rtol=0, atol=0)
    torch.testing.assert_close(empty_attention.grad, reference_empty_attention.grad, rtol=0, atol=0)

    with caplog.at_level("WARNING"):
        manager.capture()

    assert "received no iteration-0 expert tokens" in caplog.text
    assert manager.entries[0].capture_count == 1
    assert manager.entries[1].capture_count == 0
    assert manager.entries[1].adapter is None

    graph_optimizer = torch.optim.AdamW(graph_model.parameters(), lr=0.02)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.02)
    graph_losses, reference_losses = _run_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        [([2, 2], [1, 3])],
        device="cpu",
        dtype=torch.float64,
        feature_size=3,
    )

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=0, atol=0)
    assert manager.entries[0].replay_count == 2
    assert manager.entries[1].replay_count == 0
    assert manager.entries[1].fallback_count == 0
    assert manager.stats() == {"captured": 1, "replayed": 2, "fallback": 0}
    assert graph_model.expert.alias_checks == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for optimizer-level graph parity")
def test_cuda_multistep_ga_convergence_with_dynamic_splits_and_shape_fallback():
    pytest.importorskip("transformer_engine")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    graph_model = _ConvergenceModel(3, device=device, dtype=torch.float32)
    reference_model = copy.deepcopy(graph_model)
    manager = _make_manager(graph_model)
    _capture_initial_sample(
        graph_model,
        manager,
        device=device,
        dtype=torch.float32,
        feature_size=3,
        token_count=4,
        split_values=[1, 3],
    )

    graph_optimizer = torch.optim.AdamW(graph_model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.01)
    graph_losses, reference_losses = _run_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        [([2, 2], [3, 1]), ([1, 3], [2, 3]), ([3, 1], [2, 2])],
        device=device,
        dtype=torch.float32,
        feature_size=3,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=1e-6, atol=1e-6)
    assert manager.entries[0].replay_count == 6
    assert manager.entries[1].replay_count == 5
    assert manager.entries[1].fallback_count == 1


def _make_te_expert(device: torch.device, feature_size: int = 64) -> nn.Module:
    from transformer_engine.pytorch import ops as te_ops

    return te_ops.Sequential(
        te_ops.GroupedLinear(
            num_groups=2,
            in_features=feature_size,
            out_features=feature_size * 2,
            bias=True,
            dtype=torch.bfloat16,
            device=device,
        ),
        te_ops.ScaledClampedQGeGLU(glu_interleave_size=32, limit=7.0, alpha=1.702),
        te_ops.GroupedLinear(
            num_groups=2,
            in_features=feature_size,
            out_features=feature_size,
            bias=True,
            dtype=torch.bfloat16,
            device=device,
            scale_bias=True,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and Transformer Engine are required")
def test_te_ops_multistep_ga_optimizer_state_parity_with_changed_split_contents():
    pytest.importorskip("transformer_engine")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    graph_expert = _make_te_expert(device)
    reference_expert = _make_te_expert(device)
    reference_expert.load_state_dict(graph_expert.state_dict())
    graph_model = _ConvergenceModel(
        64,
        device=device,
        dtype=torch.bfloat16,
        expert=graph_expert,
    )
    reference_model = _ConvergenceModel(
        64,
        device=device,
        dtype=torch.bfloat16,
        expert=reference_expert,
    )
    reference_model.router_gain.data.copy_(graph_model.router_gain.data)
    manager = _make_manager(graph_model)
    _capture_initial_sample(
        graph_model,
        manager,
        device=device,
        dtype=torch.bfloat16,
        feature_size=64,
        token_count=64,
        split_values=[32, 32],
    )

    graph_optimizer = torch.optim.AdamW(graph_model.parameters(), lr=0.002)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.002)
    graph_losses, reference_losses = _run_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        [([16, 48], [32, 32]), ([48, 16], [24, 40])],
        device=device,
        dtype=torch.bfloat16,
        feature_size=64,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=2e-2, atol=2e-2)
    assert manager.stats() == {"captured": 2, "replayed": 8, "fallback": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and Transformer Engine are required")
def test_te_ops_non_reentrant_checkpoint_recompute_and_changed_total_fallback():
    pytest.importorskip("transformer_engine")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    graph_expert = _make_te_expert(device)
    reference_expert = _make_te_expert(device)
    reference_expert.load_state_dict(graph_expert.state_dict())
    graph_model = _CheckpointConvergenceModel(
        64,
        device=device,
        dtype=torch.bfloat16,
        expert=graph_expert,
        use_reentrant=False,
    )
    reference_model = _ConvergenceModel(
        64,
        device=device,
        dtype=torch.bfloat16,
        expert=reference_expert,
    )
    reference_model.router_gain.data.copy_(graph_model.router_gain.data)
    manager = _make_manager(graph_model)
    graph_optimizer = torch.optim.AdamW(graph_model.parameters(), lr=0.002)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.002)
    graph_losses, reference_losses = _run_checkpointed_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        manager,
        [
            ([32, 32], [16, 48]),
            ([16, 48], [32, 32]),
            ([48, 16], [24, 56]),
            ([40, 24], [8, 56]),
        ],
        device=device,
        dtype=torch.bfloat16,
        feature_size=64,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=2e-2, atol=2e-2)
    assert graph_model.checkpoint_phases.count("forward") == 8
    assert graph_model.checkpoint_phases.count("recompute") == 8
    assert manager.entries[0].replay_count == 12
    assert manager.entries[1].replay_count == 10
    assert manager.entries[1].fallback_count == 2
    assert manager.stats() == {"captured": 2, "replayed": 22, "fallback": 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Blackwell and Transformer Engine are required")
def test_te_ops_mxfp8_non_reentrant_checkpoint_keeps_attention_bf16():
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
    graph_expert = _make_te_expert(device, feature_size=256)
    reference_expert = _make_te_expert(device, feature_size=256)
    reference_expert.load_state_dict(graph_expert.state_dict())
    graph_model = _CheckpointConvergenceModel(
        256,
        device=device,
        dtype=torch.bfloat16,
        expert=graph_expert,
        use_reentrant=False,
    )
    reference_model = _ConvergenceModel(
        256,
        device=device,
        dtype=torch.bfloat16,
        expert=reference_expert,
    )
    reference_model.router_gain.data.copy_(graph_model.router_gain.data)
    manager = _make_manager(
        graph_model,
        expert_fp8_enabled=True,
        fp8_recipe=recipe,
    )
    assert manager.entries[0].fp8_enabled is False
    assert manager.entries[1].fp8_enabled is True

    graph_optimizer = torch.optim.AdamW(graph_model.parameters(), lr=0.002)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.002)

    def fp8_context_factory():
        return te.autocast(enabled=True, recipe=recipe)

    graph_losses, reference_losses = _run_checkpointed_optimizer_steps(
        graph_model,
        reference_model,
        graph_optimizer,
        reference_optimizer,
        manager,
        [
            ([256, 512], [384, 384]),
            ([512, 256], [128, 640]),
            ([384, 384], [256, 768]),
        ],
        device=device,
        dtype=torch.bfloat16,
        feature_size=256,
        rtol=2.5e-1,
        atol=5e-1,
        context_factory=fp8_context_factory,
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(torch.stack(graph_losses), torch.stack(reference_losses), rtol=2.5e-1, atol=5e-1)
    assert graph_model.checkpoint_phases.count("forward") == 6
    assert graph_model.checkpoint_phases.count("recompute") == 6
    assert manager.entries[0].replay_count == 8
    assert manager.entries[1].replay_count == 6
    assert manager.entries[1].fallback_count == 2
    assert manager.stats() == {"captured": 2, "replayed": 14, "fallback": 2}
