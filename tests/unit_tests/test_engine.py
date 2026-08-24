# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import nemo_automodel.engine._engine as engine_module
from nemo_automodel import Engine as PublicEngine
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.engine import Engine


class _Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, values: torch.Tensor, *, offset: float = 0.0) -> torch.Tensor:
        """Scale a tensor and add a scalar offset.

        Args:
            values: Input values with arbitrary shape.
            offset: Scalar added to every output value.

        Returns:
            Tensor with the same shape as ``values``.
        """
        return values * self.weight + offset


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.1)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, *args, **kwargs):
        self.zero_grad_calls += 1
        return super().zero_grad(*args, **kwargs)


class _CountingScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> None:
        self.calls += 1


def _skip_gradient_finalization(monkeypatch, norm: float = 2.0) -> None:
    monkeypatch.setattr(
        engine_module,
        "scale_grads_and_clip_grad_norm",
        lambda **_kwargs: torch.tensor(norm),
    )


def test_engine_is_a_public_module_and_raw_forward_delegate() -> None:
    module = _Scale()
    engine = Engine(module)

    output = engine(torch.tensor([1.0, 2.0]), offset=3.0)

    assert PublicEngine is Engine
    assert isinstance(engine, nn.Module)
    assert engine.module is module
    torch.testing.assert_close(output, torch.tensor([4.0, 5.0]))


def test_forward_only_engine_rejects_training_operations() -> None:
    engine = Engine(_Scale())

    with pytest.raises(RuntimeError, match="backward requires an optimizer"):
        engine.backward(torch.tensor(1.0, requires_grad=True))
    with pytest.raises(RuntimeError, match="step requires an optimizer"):
        engine.step()


def test_engine_rejects_pipeline_contract(monkeypatch) -> None:
    class _Pipeline(nn.Module):
        pass

    monkeypatch.setattr(engine_module, "AutoPipeline", _Pipeline)

    with pytest.raises(NotImplementedError, match="pipeline schedule"):
        Engine(_Pipeline())

    with pytest.raises(NotImplementedError, match="pipeline stages"):
        Engine(nn.Linear(2, 2), mesh_context=SimpleNamespace(pp_enabled=True))


def test_gradient_accumulation_updates_only_at_boundary(monkeypatch) -> None:
    _skip_gradient_finalization(monkeypatch)
    module = _Scale()
    optimizer = _CountingSGD(module.parameters())
    scheduler = _CountingScheduler()
    engine = Engine(
        module,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        gradient_accumulation_steps=2,
    )

    engine.backward(engine(torch.ones(())) ** 2)
    assert not engine.is_gradient_accumulation_boundary()
    engine.step()

    assert module.weight.item() == 1.0
    assert optimizer.step_calls == 0
    assert optimizer.zero_grad_calls == 0
    assert scheduler.calls == 0

    engine.backward(engine(torch.ones(())) ** 2)
    assert engine.is_gradient_accumulation_boundary()
    engine.step()

    torch.testing.assert_close(module.weight, torch.tensor(0.8))
    assert optimizer.step_calls == 1
    assert optimizer.zero_grad_calls == 1
    assert scheduler.calls == 1
    torch.testing.assert_close(engine.get_global_grad_norm(), torch.tensor(2.0))


def test_short_accumulation_window_updates_after_runtime_reconfiguration(monkeypatch) -> None:
    _skip_gradient_finalization(monkeypatch)
    module = _Scale()
    optimizer = _CountingSGD(module.parameters())
    engine = Engine(module, optimizer=optimizer, gradient_accumulation_steps=4)
    engine.set_gradient_accumulation_steps(2)

    for _ in range(2):
        engine.backward(engine(torch.ones(())) ** 2)
        engine.step()

    assert optimizer.step_calls == 1
    torch.testing.assert_close(module.weight, torch.tensor(0.8))


def test_backward_can_skip_main_loss_gas_scaling() -> None:
    module = _Scale()
    engine = Engine(
        module,
        optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
        gradient_accumulation_steps=4,
    )

    engine.backward(engine(torch.tensor(3.0)), scale_wrt_gas=False)

    torch.testing.assert_close(module.weight.grad, torch.tensor(3.0))
    torch.testing.assert_close(MoEAuxLossAutoScaler.main_loss_backward_scale, torch.tensor(0.25))


@pytest.mark.parametrize(
    ("declared_mode", "expected_gradient", "expected_aux_scale"),
    [
        pytest.param(None, 2.0, 2.0 / 3.0, id="undeclared-averaged"),
        pytest.param(False, 2.0, 2.0 / 3.0, id="explicit-averaged"),
        pytest.param(True, 0.25, 1.0 / 12.0, id="summed"),
    ],
)
def test_backward_compensates_backend_gradient_reduction(
    monkeypatch,
    declared_mode,
    expected_gradient,
    expected_aux_scale,
) -> None:
    module = _Scale()
    if declared_mode is not None:
        module.calculate_per_token_loss = declared_mode
    engine = Engine(
        module,
        optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
        gradient_accumulation_steps=3,
    )
    monkeypatch.setattr(engine, "_gradient_group_size", lambda: 8)
    monkeypatch.setattr(engine, "_context_parallel_size", lambda: 2)

    engine.backward(engine(torch.tensor(2.0)), scale_wrt_gas=False)

    torch.testing.assert_close(module.weight.grad, torch.tensor(expected_gradient))
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(expected_aux_scale)


def test_summed_gradient_declaration_is_found_below_wrapper() -> None:
    module = _Scale()
    module.distributed_backend = nn.Identity()
    module.distributed_backend.calculate_per_token_loss = True

    engine = Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1))

    assert engine._gradient_reduction_compensation(8) == pytest.approx(1.0 / 8.0)


@pytest.mark.parametrize(
    ("root_mode", "nested_mode", "error_type", "match"),
    [
        (True, False, ValueError, "mixes calculate_per_token_loss"),
        ("yes", None, TypeError, "must be boolean"),
    ],
)
def test_gradient_reduction_declarations_must_be_consistent(root_mode, nested_mode, error_type, match) -> None:
    module = _Scale()
    module.calculate_per_token_loss = root_mode
    if nested_mode is not None:
        module.mode_probe = nn.Identity()
        module.mode_probe.calculate_per_token_loss = nested_mode

    with pytest.raises(error_type, match=match):
        Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1))


def test_forward_keeps_sync_context_open_through_backward(monkeypatch) -> None:
    events: list[str] = []
    boundaries: list[tuple[bool, bool]] = []

    @contextmanager
    def sync_context(_module, boundary, defer_fsdp_grad_sync):
        boundaries.append((boundary, defer_fsdp_grad_sync))
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class _ObservedScale(_Scale):
        def forward(self, values: torch.Tensor, *, offset: float = 0.0) -> torch.Tensor:
            """Record forward before delegating to ``_Scale``."""
            events.append("forward")
            return super().forward(values, offset=offset)

    module = _ObservedScale()
    module.weight.register_hook(lambda gradient: events.append("backward") or gradient)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    engine = Engine(module, optimizer=optimizer, gradient_accumulation_steps=2)
    monkeypatch.setattr(engine_module, "get_sync_ctx", sync_context)
    monkeypatch.setattr(
        engine_module,
        "prepare_for_grad_accumulation",
        lambda *_args, **_kwargs: events.append("prepare"),
    )
    monkeypatch.setattr(
        engine_module,
        "prepare_for_final_backward",
        lambda *_args, **_kwargs: events.append("final"),
    )
    monkeypatch.setattr(engine_module, "prepare_after_first_microbatch", lambda: events.append("first_done"))
    _skip_gradient_finalization(monkeypatch)

    engine.backward(engine(torch.ones(())))
    engine.step()
    engine.backward(engine(torch.ones(())))
    engine.step()

    assert boundaries == [(False, True), (True, True)]
    assert events == [
        "prepare",
        "enter",
        "forward",
        "backward",
        "exit",
        "first_done",
        "final",
        "enter",
        "forward",
        "backward",
        "exit",
    ]


def test_no_grad_forward_does_not_open_training_context(monkeypatch) -> None:
    module = _Scale()
    engine = Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1))
    monkeypatch.setattr(
        engine_module,
        "get_sync_ctx",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected training context")),
    )

    with torch.no_grad():
        output = engine(torch.tensor(2.0))

    torch.testing.assert_close(output, torch.tensor(2.0))


def test_eval_forward_does_not_open_training_context(monkeypatch) -> None:
    module = _Scale().eval()
    engine = Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1))
    monkeypatch.setattr(
        engine_module,
        "get_sync_ctx",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected training context")),
    )

    output = engine(torch.tensor(2.0))

    torch.testing.assert_close(output, torch.tensor(2.0))


def test_forward_failure_closes_sync_context(monkeypatch) -> None:
    exits = 0

    @contextmanager
    def sync_context(*_args, **_kwargs):
        nonlocal exits
        try:
            yield
        finally:
            exits += 1

    class _FailOnce(_Scale):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def forward(self, values: torch.Tensor, *, offset: float = 0.0) -> torch.Tensor:
            """Fail once, then return a tensor shaped like ``values``."""
            if self.fail:
                self.fail = False
                raise RuntimeError("forward failed")
            return super().forward(values, offset=offset)

    module = _FailOnce()
    engine = Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1))
    monkeypatch.setattr(engine_module, "get_sync_ctx", sync_context)

    with pytest.raises(RuntimeError, match="forward failed"):
        engine(torch.ones(()))
    engine.backward(engine(torch.ones(())))

    assert exits == 2


def test_accumulation_steps_validate_and_cannot_change_mid_window() -> None:
    module = _Scale()
    engine = Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1), gradient_accumulation_steps=2)

    with pytest.raises(ValueError, match="positive integer"):
        engine.set_gradient_accumulation_steps(0)
    engine.backward(engine(torch.ones(())))
    engine.step()
    with pytest.raises(RuntimeError, match="active window"):
        engine.set_gradient_accumulation_steps(3)


def test_optimizer_boundary_runs_model_post_step_hooks(monkeypatch) -> None:
    _skip_gradient_finalization(monkeypatch)
    precomputed: list[nn.Module] = []

    class _PostStepModel(_Scale):
        precompute_float8_dynamic_scale_for_fsdp = True

        def __init__(self) -> None:
            super().__init__()
            self.gate_updates = 0

        def update_moe_gate_bias(self) -> None:
            self.gate_updates += 1

    monkeypatch.setattr(
        engine_module,
        "safe_import_from",
        lambda *_args, **_kwargs: (True, lambda module: precomputed.append(module)),
    )
    module = _PostStepModel()
    engine = Engine(module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1))

    engine.backward(engine(torch.ones(())))
    engine.step()

    assert module.gate_updates == 1
    assert precomputed == [module]


def test_optimizer_param_scheduler_step_defaults_to_one() -> None:
    default = inspect.signature(OptimizerParamScheduler.step).parameters["increment"].default

    assert default == 1
