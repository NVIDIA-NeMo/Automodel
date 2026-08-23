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

from __future__ import annotations

import pickle
import sys
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import nemo_automodel.engine as engine_module
from nemo_automodel import CollatedLossInputs as PublicCollatedLossInputs
from nemo_automodel import Datum as PublicDatum
from nemo_automodel import Engine as PublicEngine
from nemo_automodel import LossInputLayout as PublicLossInputLayout
from nemo_automodel.components.datasets.datum import (
    CollatedLossInputs,
    Datum,
    LossInputLayout,
    collate_datums,
    collate_vlm_datums,
)
from nemo_automodel.components.datasets.vlm.pp_media import VLM_PP_MEDIA_KEY, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.config import FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    shard_batch_contiguous,
)
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.common.mtp import prepare_mtp_context_parallel_inputs
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.engine import Engine, ForwardBackwardResult, ForwardResult, OptimStepResult, collate_prebatched
from nemo_automodel.engine.outputs import LossFnOutputBatch, PerTokenOutput


class ScaleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        self.forward_calls += 1
        return input_ids.to(torch.float32) * self.weight


class _MainAndAuxScaleModel(nn.Module):
    """Small model whose main and auto-scaled auxiliary gradients are separable."""

    def __init__(self) -> None:
        super().__init__()
        self.main_weight = nn.Parameter(torch.tensor(1.0))
        self.aux_weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        self.forward_calls += 1
        output = input_ids.to(torch.float32) * self.main_weight
        return MoEAuxLossAutoScaler.apply(output, self.aux_weight)


class _UniformTokenDispatcherProbe(nn.Module):
    """Expose the live-dispatcher capability Engine resolves after EP setup."""

    def __init__(self) -> None:
        super().__init__()
        self.token_dispatcher = SimpleNamespace(requires_uniform_token_count=True)


class _SubMesh:
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    def get_group(self):
        return None


class _CPMesh(dict):
    def __init__(self, size, rank):
        super().__init__(cp=_SubMesh(size, rank), tp=_SubMesh(1))
        self.mesh_dim_names = ("cp", "tp")


class _NamedMesh(dict):
    def __init__(self, names, **axes):
        super().__init__(axes)
        self.mesh_dim_names = tuple(names)


class _FakeAutoPipeline(AutoPipeline):
    def __init__(
        self,
        model,
        *,
        parts=None,
        num_microbatches=2,
        scale_grads=False,
        events=None,
        callback_order=None,
        has_last_stage=True,
        pp_microbatch_size=1,
    ):
        self.compute_model = model
        self._parts = parts or [model]
        self._num_microbatches = num_microbatches
        self.pp_microbatch_size = pp_microbatch_size
        self.scale_grads_in_schedule = scale_grads
        self.pp_mesh = _SubMesh(2)
        self._info = SimpleNamespace(has_last_stage=has_last_stage)
        self.events = events
        self.callback_order = callback_order or list(range(num_microbatches))
        self.step_calls = 0
        self.eval_calls = 0
        self.backward_calls = 0
        self.updated_seq_lens = []
        self.updated_microbatch_sizes = []
        self.updated_input_shapes = []
        self.callback_losses = []
        self.prepared_inputs = []

    @property
    def parts(self):
        return self._parts

    @property
    def num_microbatches(self):
        return self._num_microbatches

    @property
    def info(self):
        return self._info

    def update_seq_len(self, seq_len, *, microbatch_size=None, input_tensor=None):
        self.updated_seq_lens.append(seq_len)
        self.updated_microbatch_sizes.append(microbatch_size)
        self.updated_input_shapes.append(tuple(input_tensor.shape) if input_tensor is not None else None)

    def step_microbatches(
        self,
        model_inputs,
        *,
        loss_fn,
        losses,
        return_outputs,
        batch_context_fns=None,
    ):
        assert return_outputs is False
        assert len(model_inputs) == self.num_microbatches
        self.prepared_inputs.append(model_inputs)
        self.step_calls += 1
        if self.events is not None:
            self.events.append("step")

        for index in self.callback_order:
            inputs = dict(model_inputs[index])
            primary_name = "inputs_embeds" if "inputs_embeds" in inputs else "input_ids"
            primary = inputs.pop(primary_name)
            context_fn = nullcontext if batch_context_fns is None else batch_context_fns[index]
            with context_fn():
                output = self.compute_model(primary, **inputs)
            scaled_loss = loss_fn(output, index)
            self.callback_losses.append(scaled_loss.detach())
            with context_fn():
                scaled_loss.backward()
            self.backward_calls += 1

    def eval_microbatches(
        self,
        model_inputs,
        *,
        loss_fn,
        losses,
        return_outputs,
        batch_context_fns=None,
    ):
        assert return_outputs is False
        assert len(model_inputs) == self.num_microbatches
        self.prepared_inputs.append(model_inputs)
        self.eval_calls += 1
        if self.events is not None:
            self.events.append("eval")

        for index in self.callback_order:
            inputs = dict(model_inputs[index])
            primary_name = "inputs_embeds" if "inputs_embeds" in inputs else "input_ids"
            primary = inputs.pop(primary_name)
            context_fn = nullcontext if batch_context_fns is None else batch_context_fns[index]
            with context_fn():
                output = self.compute_model(primary, **inputs)
            loss = loss_fn(output, index)
            self.callback_losses.append(loss.detach())


def _pipeline_mesh_context():
    return SimpleNamespace(pp_size=2, cp_size=1, device_mesh=None, moe_mesh=None, process_group=None)


class _DDPWithCP(nn.parallel.DistributedDataParallel):
    def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
        return self.module.prepare_model_inputs_for_cp(batch, num_chunks=num_chunks)


class _DistributedCPModel(ScaleModel):
    def prepare_model_inputs_for_cp(self, _batch, *, num_chunks):
        assert num_chunks == 1
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=partial(shard_batch_contiguous, pad_multiple=1),
                local_token_global_indices=contiguous_local_indices,
            )
        }


def _datum(values, weights=None) -> Datum:
    values = torch.tensor(values, dtype=torch.long)
    weights = torch.ones_like(values, dtype=torch.float32) if weights is None else torch.tensor(weights)
    return Datum(model_inputs={"input_ids": values}, loss_fn_inputs={"weights": weights})


def _identity_loss(output, _loss_inputs):
    return output


def _configure_fake_gradient_group(engine: Engine, monkeypatch, *, group_size: int) -> None:
    """Expose a logical gradient group without running a real collective."""
    engine._gradient_group_and_size = lambda _group, _size: (None, group_size)
    engine._validate_window_size_across_group = lambda *_args, **_kwargs: None
    monkeypatch.setattr(engine_module.dist, "all_reduce", lambda *_args, **_kwargs: None)


def test_engine_and_datum_are_lazy_top_level_exports():
    assert PublicEngine is Engine
    assert PublicDatum is Datum
    assert PublicLossInputLayout is LossInputLayout
    assert PublicCollatedLossInputs is CollatedLossInputs


def test_fp8_scale_resolver_skips_import_without_capable_model_parts(monkeypatch):
    first = ScaleModel()
    second = ScaleModel()
    second.precompute_float8_dynamic_scale_for_fsdp = False
    monkeypatch.setattr(
        engine_module,
        "safe_import_from",
        lambda *_args, **_kwargs: pytest.fail("non-FP8 Engine construction must not import torchao"),
    )

    parts, precompute = engine_module._resolve_fp8_scale_precompute([first, second])

    assert parts == ()
    assert precompute is None


def test_fp8_scale_resolver_filters_capable_parts_and_caches_callable(monkeypatch):
    first = ScaleModel()
    disabled = ScaleModel()
    third = ScaleModel()
    first.precompute_float8_dynamic_scale_for_fsdp = True
    disabled.precompute_float8_dynamic_scale_for_fsdp = False
    third.precompute_float8_dynamic_scale_for_fsdp = True
    precompute = lambda _part: None
    calls = []

    def resolve(module, symbol, *, msg):
        calls.append((module, symbol, msg))
        return True, precompute

    monkeypatch.setattr(engine_module, "safe_import_from", resolve)

    parts, resolved = engine_module._resolve_fp8_scale_precompute([first, disabled, third])

    assert parts == (first, third)
    assert resolved is precompute
    assert calls == [
        (
            "torchao.float8",
            "precompute_float8_dynamic_scale_for_fsdp",
            engine_module.MISSING_TORCHAO_MSG,
        )
    ]


def test_capable_fp8_model_rejects_missing_torchao_api_during_engine_construction(monkeypatch):
    model = ScaleModel()
    model.precompute_float8_dynamic_scale_for_fsdp = True
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(engine_module, "safe_import_from", lambda *_args, **_kwargs: (False, object()))

    with pytest.raises(ImportError, match="torchao"):
        Engine(model, device="cpu", optimizers=optimizer)

    torch.testing.assert_close(model.weight, torch.tensor(1.0))
    assert model.weight.grad is None


def test_optim_step_clips_updates_and_clears_real_gradients():
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = Engine(model, device="cpu", optimizers=optimizer, max_grad_norm=0.5)

    engine.forward_backward([_datum([4])], _identity_loss)
    torch.testing.assert_close(model.weight.grad, torch.tensor(4.0))

    result = engine.optim_step()

    assert isinstance(result, OptimStepResult)
    torch.testing.assert_close(result.grad_norm, torch.tensor(4.0, dtype=torch.float64))
    torch.testing.assert_close(model.weight, torch.tensor(0.95))
    assert model.weight.grad is None
    assert result.learning_rates == (0.1,)


def test_optim_step_orders_multi_optimizer_topology_and_post_step_work(monkeypatch):
    events = []

    class GateModel(ScaleModel):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def update_moe_gate_bias(self):
            events.append(f"gate-{self.name}")

    class RecordingSGD(torch.optim.SGD):
        def __init__(self, name, parameters, lr):
            self.name = name
            super().__init__(parameters, lr=lr)

        def step(self, closure=None):
            events.append(f"step-{self.name}")
            return super().step(closure)

        def zero_grad(self, set_to_none=True):
            assert set_to_none is True
            events.append(f"zero-{self.name}")
            return super().zero_grad(set_to_none=set_to_none)

    class RecordingScheduler:
        def __init__(self, name, optimizer):
            self.name = name
            self.optimizer = optimizer

        def step(self, increment):
            assert increment == 1
            events.append(f"scheduler-{self.name}")
            for group in self.optimizer.param_groups:
                group["lr"] += 0.01

    first = GateModel("first")
    second = GateModel("second")
    first._nemo_moe_tp_requires_replica_sync = True
    first.weight.grad = torch.tensor(2.0)
    second.weight.grad = torch.tensor(3.0)
    pipeline = _FakeAutoPipeline(first, parts=[first, second])
    first_optimizer = RecordingSGD("first", first.parameters(), lr=0.1)
    second_optimizer = RecordingSGD("second", second.parameters(), lr=0.2)
    first_scheduler = RecordingScheduler("first", first_optimizer)
    second_scheduler = RecordingScheduler("second", second_optimizer)
    device_mesh = _NamedMesh(("cp", "tp"), cp=_SubMesh(2), tp=_SubMesh(4))
    moe_mesh = _NamedMesh(("ep_shard", "ep"), ep_shard=_SubMesh(2), ep=_SubMesh(2))

    def precompute_fp8_scale(part):
        events.append(f"fp8-{part.name}")

    def resolve_fp8_scale_precompute(parts):
        assert parts == [first, second]
        return tuple(parts), precompute_fp8_scale

    monkeypatch.setattr(engine_module, "_resolve_fp8_scale_precompute", resolve_fp8_scale_precompute)
    engine = Engine(
        pipeline,
        device="cpu",
        mesh_context=SimpleNamespace(
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            cp_size=2,
            pp_size=2,
            process_group=None,
        ),
        optimizers=[first_optimizer, second_optimizer],
        lr_schedulers=[first_scheduler, second_scheduler],
        max_grad_norm=None,
    )
    engine._dp_group_and_size = lambda: (None, 2)
    engine._gradient_group_and_size = lambda _group, _size: (None, 8)
    observed = {}

    def finalize(**kwargs):
        events.append("finalize")
        observed.update(kwargs)
        return torch.tensor(7.0)

    monkeypatch.setattr(engine_module, "scale_grads_and_clip_grad_norm", finalize)

    def before_optimizer_step():
        events.append("before-step")

    result = engine.optim_step(before_optimizer_step=before_optimizer_step)

    assert engine.optimizers == (first_optimizer, second_optimizer)
    assert engine.lr_schedulers == (first_scheduler, second_scheduler)
    assert events == [
        "finalize",
        "before-step",
        "step-first",
        "zero-first",
        "step-second",
        "zero-second",
        "gate-first",
        "gate-second",
        "fp8-first",
        "fp8-second",
        "scheduler-first",
        "scheduler-second",
    ]
    assert observed == {
        "max_grad_norm": None,
        "model_parts": [first, second],
        "norm_type": 2.0,
        "pp_enabled": True,
        "device_mesh": device_mesh,
        "moe_mesh": moe_mesh,
        "ep_axis_name": "ep",
        "pp_axis_name": "pp",
        "foreach": True,
        "num_label_tokens": None,
        "dp_group_size": 8,
        "expert_tp_replication_factor": 4,
    }
    assert result.grad_norm.item() == pytest.approx(7.0)
    assert result.learning_rates == pytest.approx((0.11, 0.21))
    assert first.weight.grad is None
    assert second.weight.grad is None


def test_optim_step_fp8_post_step_failure_does_not_advance_scheduler(monkeypatch):
    events = []

    class GateModel(ScaleModel):
        def update_moe_gate_bias(self):
            events.append("gate")

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("step")
            return super().step(closure)

        def zero_grad(self, set_to_none=True):
            events.append("zero")
            return super().zero_grad(set_to_none=set_to_none)

    class RecordingScheduler:
        def step(self, increment):
            events.append("scheduler")

    model = GateModel()
    model.weight.grad = torch.tensor(2.0)
    optimizer = RecordingSGD(model.parameters(), lr=0.1)

    def fail_fp8_post_step(part):
        assert part is model
        events.append("fp8")
        raise RuntimeError("fp8 scale precompute failed")

    monkeypatch.setattr(
        engine_module,
        "_resolve_fp8_scale_precompute",
        lambda parts: (tuple(parts), fail_fp8_post_step),
    )
    engine = Engine(
        model,
        device="cpu",
        optimizers=optimizer,
        lr_schedulers=RecordingScheduler(),
    )
    monkeypatch.setattr(
        engine_module,
        "scale_grads_and_clip_grad_norm",
        lambda **_kwargs: torch.tensor(2.0),
    )

    with pytest.raises(RuntimeError, match="fp8 scale precompute failed"):
        engine.optim_step()

    assert events == ["step", "zero", "gate", "fp8"]
    torch.testing.assert_close(model.weight, torch.tensor(0.8))
    assert model.weight.grad is None

    with pytest.raises(RuntimeError, match="already consumed|cannot be optimized"):
        engine.optim_step()
    assert events == ["step", "zero", "gate", "fp8"]


def test_optim_step_callback_failure_preserves_optimizer_and_post_step_state(monkeypatch):
    events = []

    class GateModel(ScaleModel):
        def update_moe_gate_bias(self):
            events.append("gate")

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("step")
            return super().step(closure)

        def zero_grad(self, set_to_none=True):
            events.append("zero")
            return super().zero_grad(set_to_none=set_to_none)

    class RecordingScheduler:
        def step(self, increment):
            events.append("scheduler")

    model = GateModel()
    model.weight.grad = torch.tensor(2.0)
    optimizer = RecordingSGD(model.parameters(), lr=0.1)
    engine = Engine(
        model,
        device="cpu",
        optimizers=optimizer,
        lr_schedulers=RecordingScheduler(),
    )

    def finalize(**_kwargs):
        events.append("finalize")
        return torch.tensor(2.0)

    monkeypatch.setattr(engine_module, "scale_grads_and_clip_grad_norm", finalize)

    def fail_before_step():
        events.append("before-step")
        raise ValueError("staging failed")

    with pytest.raises(ValueError, match="staging failed"):
        engine.optim_step(before_optimizer_step=fail_before_step)

    assert events == ["finalize", "before-step"]
    torch.testing.assert_close(model.weight, torch.tensor(1.0))
    torch.testing.assert_close(model.weight.grad, torch.tensor(2.0))

    # Retrying the mutation fence must consume the already-finalized gradients;
    # running finalization/scaling twice would silently change the update.
    result = engine.optim_step()
    assert events == ["finalize", "before-step", "step", "zero", "gate", "scheduler"]
    torch.testing.assert_close(model.weight, torch.tensor(0.8))
    assert model.weight.grad is None
    assert result.learning_rates == (0.1,)


def test_optim_step_requires_an_optimizer(monkeypatch):
    monkeypatch.setattr(
        engine_module,
        "scale_grads_and_clip_grad_norm",
        lambda **_kwargs: pytest.fail("gradient finalization must not run without an optimizer"),
    )

    with pytest.raises(RuntimeError, match="requires at least one optimizer"):
        Engine(ScaleModel(), device="cpu").optim_step()


def test_optim_step_rejects_noncallable_mutation_fence_before_finalization(monkeypatch):
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(
        engine_module,
        "scale_grads_and_clip_grad_norm",
        lambda **_kwargs: pytest.fail("invalid callback must fail before gradient finalization"),
    )

    with pytest.raises(TypeError, match="before_optimizer_step must be callable or None"):
        Engine(model, device="cpu", optimizers=optimizer).optim_step(before_optimizer_step=object())


def test_forward_runs_eval_without_grad_lifecycle_and_returns_local_statistics(monkeypatch):
    class EvalModel(ScaleModel):
        def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
            assert not self.training
            assert not torch.is_grad_enabled()
            return super().forward(input_ids, **kwargs)

    model = EvalModel()
    model.weight.grad = torch.tensor(7.0)
    monkeypatch.setattr(
        engine_module,
        "prepare_for_grad_accumulation",
        lambda *_args, **_kwargs: pytest.fail("forward-only execution must not prepare gradient accumulation"),
    )
    monkeypatch.setattr(
        engine_module,
        "prepare_for_final_backward",
        lambda *_args, **_kwargs: pytest.fail("forward-only execution must not prepare backward"),
    )
    monkeypatch.setattr(
        engine_module,
        "get_sync_ctx",
        lambda *_args, **_kwargs: pytest.fail("forward-only execution must not enter a gradient sync context"),
    )
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(9.0))

    def loss_with_outputs(output, _loss_inputs):
        assert not torch.is_grad_enabled()
        return output, [{"value": output.sum()}]

    engine = Engine(model, device="cpu")
    engine._dp_group_and_size = lambda: pytest.fail("forward must not synchronize data-parallel replicas")
    result = engine.forward(
        [_datum([1, 100], [1.0, 0.0]), _datum([3], [0.5])],
        loss_with_outputs,
    )

    assert isinstance(result, ForwardResult)
    assert result.loss_sum.item() == pytest.approx(2.5)
    assert result.weight_sum.item() == pytest.approx(1.5)
    assert [item["value"].item() for item in result.loss_fn_outputs] == [101.0, 3.0]
    assert all(not item["value"].requires_grad for item in result.loss_fn_outputs)
    assert model.weight.grad.item() == 7.0
    assert model.forward_calls == 2
    assert not model.training
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == 9.0


def test_forward_zero_weight_window_still_executes_without_gradients():
    model = ScaleModel()

    result = Engine(model, device="cpu").forward(
        [_datum([1, 2], [0.0, 0.0])],
        lambda output, _inputs: output.sum(),
    )

    assert result.loss_sum.item() == 0
    assert result.weight_sum.item() == 0
    assert result.loss_fn_outputs == []
    assert model.forward_calls == 1
    assert model.weight.grad is None


def test_forward_backward_uses_one_denominator_for_the_window():
    model = ScaleModel()
    initial_weight = model.weight.detach().clone()
    engine = Engine(model, device="cpu")

    result = engine.forward_backward(
        [_datum([1, 2]), _datum([3])],
        _identity_loss,
    )

    assert isinstance(result, ForwardBackwardResult)
    assert result.loss.item() == pytest.approx(2.0)
    assert result.loss_sum.item() == pytest.approx(6.0)
    assert result.weight_sum.item() == pytest.approx(3.0)
    assert result.loss_fn_outputs == []
    assert model.weight.grad.item() == pytest.approx(2.0)
    assert torch.equal(model.weight, initial_weight)
    assert model.forward_calls == 2


def test_multiple_backward_calls_with_an_optimizer_fail_fast():
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = Engine(model, device="cpu", optimizers=optimizer, max_grad_norm=None)

    engine.forward_backward([_datum([2])], _identity_loss)
    with pytest.raises(RuntimeError, match="optim_step"):
        engine.forward_backward([_datum([6])], _identity_loss)

    torch.testing.assert_close(model.weight.grad, torch.tensor(2.0))
    engine.optim_step()
    torch.testing.assert_close(model.weight, torch.tensor(0.8))


def test_failed_backward_poisoned_gradients_cannot_be_reused():
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = Engine(model, device="cpu", optimizers=optimizer, max_grad_norm=None)
    calls = 0

    def fail_on_second_microbatch(output, _loss_inputs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second microbatch failed")
        return output

    with pytest.raises(ValueError, match="second microbatch failed"):
        engine.forward_backward([_datum([2]), _datum([6])], fail_on_second_microbatch)

    torch.testing.assert_close(model.weight.grad, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="failed"):
        engine.forward_backward([_datum([4])], _identity_loss)
    with pytest.raises(RuntimeError, match="failed"):
        engine.optim_step()
    torch.testing.assert_close(model.weight, torch.tensor(1.0))


def test_optim_step_rejects_double_step_and_accepts_a_new_window():
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = Engine(model, device="cpu", optimizers=optimizer, max_grad_norm=None)

    engine.forward_backward([_datum([2])], _identity_loss)
    engine.optim_step()
    torch.testing.assert_close(model.weight, torch.tensor(0.8))

    with pytest.raises(RuntimeError, match="already consumed"):
        engine.optim_step()

    engine.forward_backward([_datum([5])], _identity_loss)
    engine.optim_step()
    torch.testing.assert_close(model.weight, torch.tensor(0.3))


def test_forward_backward_groups_flat_datums_by_microbatch_size():
    group_sizes = []

    def recording_collate(datums):
        group_sizes.append(len(datums))
        return collate_datums(datums)

    model = ScaleModel()
    result = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=recording_collate,
    ).forward_backward([_datum([value]) for value in range(1, 6)], _identity_loss)

    assert group_sizes == [2, 2, 1]
    assert model.forward_calls == 3
    assert result.loss.item() == pytest.approx(3.0)


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
def test_explicit_microbatch_sizes_override_fixed_grouping(execution):
    groups = []

    def recording_collate(datums):
        """Record and collate one explicit group of ``[S]`` token Datums."""
        groups.append(tuple(int(datum.model_inputs["input_ids"].item()) for datum in datums))
        return collate_datums(datums)

    def loss_with_outputs(output, _loss_inputs):
        """Return ``[B, S]`` losses and one scalar record per Datum."""
        return output, [{"value": row.flatten()[0]} for row in output]

    model = ScaleModel()
    result = getattr(Engine(model, device="cpu", microbatch_size=1, collate_fn=recording_collate), execution)(
        [_datum([value]) for value in range(1, 5)],
        loss_with_outputs,
        microbatch_sizes=[1, 3],
    )

    assert groups == [(1,), (2, 3, 4)]
    assert model.forward_calls == 2
    assert [item["value"].item() for item in result.loss_fn_outputs] == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    ("microbatch_sizes", "error_type"),
    [
        pytest.param(1, TypeError, id="not-a-sequence"),
        pytest.param([], ValueError, id="empty"),
        pytest.param([0, 4], ValueError, id="zero"),
        pytest.param([-1, 5], ValueError, id="negative"),
        pytest.param([True, 3], TypeError, id="bool"),
        pytest.param([1.5, 2.5], TypeError, id="non-integer"),
        pytest.param([1, 2], ValueError, id="sum-too-small"),
        pytest.param([1, 4], ValueError, id="sum-too-large"),
    ],
)
def test_explicit_microbatch_sizes_fail_before_collation(microbatch_sizes, error_type):
    def unexpected_collate(_datums):
        pytest.fail("invalid explicit microbatch sizes must fail before collation")

    model = ScaleModel()
    with pytest.raises(error_type):
        Engine(model, device="cpu", collate_fn=unexpected_collate).forward(
            [_datum([value]) for value in range(1, 5)],
            _identity_loss,
            microbatch_sizes=microbatch_sizes,
        )

    assert model.forward_calls == 0


def test_explicit_microbatch_sizes_use_one_weighted_optimizer_window():
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    datums = [
        _datum([2, 100], [3.0, 0.0]),
        _datum([4], [1.0]),
        _datum([6, 8], [2.0, 1.0]),
        _datum([10], [3.0]),
    ]
    engine = Engine(
        model,
        device="cpu",
        microbatch_size=1,
        optimizers=optimizer,
        max_grad_norm=None,
    )

    result = engine.forward_backward(datums, _identity_loss, microbatch_sizes=(1, 3))

    assert model.forward_calls == 2
    assert result.loss_sum.item() == pytest.approx(60.0)
    assert result.weight_sum.item() == pytest.approx(10.0)
    assert result.loss.item() == pytest.approx(6.0)
    assert model.weight.grad.item() == pytest.approx(6.0)

    engine.optim_step()
    assert model.weight.item() == pytest.approx(0.4)


def test_hybridep_padding_extends_only_the_physical_thd_extent():
    """Synthetic EP padding preserves documents and every side-channel sentinel."""
    model_inputs = {
        "input_ids": torch.tensor([[7, 99, 8]]),
        "attention_mask": torch.tensor([[1, 0, 1]]),
        "position_ids": torch.tensor([[[0, 0, 1]], [[10, 0, 11]], [[20, 0, 21]]]),
        "mm_token_type_ids": torch.tensor([[1, 0, 0]]),
        "seq_lens": torch.tensor([[1, 1, -1000]], dtype=torch.int32),
        "seq_lens_padded": torch.tensor([[2, 1, -1000]], dtype=torch.int32),
        "qkv_format": "thd",
    }
    loss_inputs = {
        "weights": torch.tensor([[1.0, 0.0, 1.0]]),
        "labels": torch.tensor([[8, -100, 9]]),
        "routed_experts": torch.tensor([[[[2]], [[-1]], [[3]]]], dtype=torch.int16),
        "sequence_ids": torch.tensor([[0, -1, 1]]),
        "global_scale": torch.tensor(0.25),
    }
    layouts = {
        "weights": LossInputLayout.PER_TOKEN,
        "labels": LossInputLayout.PER_TOKEN,
        "routed_experts": LossInputLayout.PER_TOKEN,
        "sequence_ids": LossInputLayout.PER_TOKEN,
        "global_scale": LossInputLayout.REPLICATED,
    }

    engine_module._pad_hybridep_packed_thd(
        model_inputs,
        loss_inputs,
        layouts,
        {"routed_experts": -1, "sequence_ids": -1},
        loss_seq_dim=1,
        target_tokens=5,
        padding_token_id=99,
    )

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[7, 99, 8, 99, 99]]))
    torch.testing.assert_close(model_inputs["attention_mask"], torch.tensor([[1, 0, 1, 0, 0]]))
    torch.testing.assert_close(model_inputs["mm_token_type_ids"], torch.tensor([[1, 0, 0, 0, 0]]))
    torch.testing.assert_close(
        model_inputs["position_ids"],
        torch.tensor([[[0, 0, 1, 2, 3]], [[10, 0, 11, 12, 13]], [[20, 0, 21, 22, 23]]]),
    )
    torch.testing.assert_close(model_inputs["seq_lens"], torch.tensor([[1, 1, -1000]], dtype=torch.int32))
    torch.testing.assert_close(model_inputs["seq_lens_padded"], torch.tensor([[2, 3, -1000]], dtype=torch.int32))
    torch.testing.assert_close(model_inputs["padding_mask"], torch.tensor([[False, True, False, True, True]]))
    torch.testing.assert_close(loss_inputs["weights"], torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]]))
    torch.testing.assert_close(loss_inputs["labels"], torch.tensor([[8, -100, 9, -100, -100]]))
    torch.testing.assert_close(
        loss_inputs["routed_experts"],
        torch.tensor([[[[2]], [[-1]], [[3]], [[-1]], [[-1]]]], dtype=torch.int16),
    )
    torch.testing.assert_close(loss_inputs["sequence_ids"], torch.tensor([[0, -1, 1, -1, -1]]))
    torch.testing.assert_close(loss_inputs["global_scale"], torch.tensor(0.25))

    real, padded, token_mask = engine_module._output_sequence_lengths(
        [_datum([7]), _datum([8])],
        model_inputs,
        loss_inputs,
        (0, 1),
        is_thd=True,
    )
    assert real == (1, 1)
    assert padded == (2, 3)
    assert token_mask is None


def test_hybridep_padded_equalization_extends_sequence_and_side_channels():
    """Synthetic sequence columns are masked without changing media coordinates."""
    pixel_values = torch.arange(6).view(2, 3)
    model_inputs = {
        "input_ids": torch.tensor([[7, 8, 0], [9, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "position_ids": torch.tensor(
            [
                [[0, 1, 0], [0, 0, 0]],
                [[10, 11, 0], [20, 0, 0]],
            ]
        ),
        "mm_token_type_ids": torch.tensor([[1, 0, 0], [0, 0, 0]]),
        "pixel_values": pixel_values,
    }
    loss_inputs = {
        "weights": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "labels": torch.tensor([[8, 9, -100], [10, -100, -100]]),
        "routed_experts": torch.tensor(
            [[[[2]], [[3]], [[-1]]], [[[4]], [[-1]], [[-1]]]],
            dtype=torch.int16,
        ),
        "scores": torch.tensor([0.25, 0.5]),
        "global_scale": torch.tensor(0.75),
    }
    layouts = {
        "weights": LossInputLayout.PER_TOKEN,
        "labels": LossInputLayout.PER_TOKEN,
        "routed_experts": LossInputLayout.PER_TOKEN,
        "scores": LossInputLayout.PER_DATUM,
        "global_scale": LossInputLayout.REPLICATED,
    }

    engine_module._pad_hybridep_padded_sequence(
        model_inputs,
        loss_inputs,
        layouts,
        {"weights": 7, "labels": -100, "routed_experts": -1},
        target_sequence_length=5,
        padding_token_id=99,
    )

    torch.testing.assert_close(
        model_inputs["input_ids"],
        torch.tensor([[7, 8, 0, 99, 99], [9, 0, 0, 99, 99]]),
    )
    torch.testing.assert_close(
        model_inputs["attention_mask"],
        torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 0, 0]]),
    )
    torch.testing.assert_close(
        model_inputs["padding_mask"],
        torch.tensor(
            [
                [False, False, True, True, True],
                [False, True, True, True, True],
            ]
        ),
    )
    assert model_inputs["position_ids"].shape == (2, 2, 5)
    assert model_inputs["mm_token_type_ids"].shape == (2, 5)
    assert model_inputs["pixel_values"] is pixel_values
    torch.testing.assert_close(
        loss_inputs["weights"],
        torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(loss_inputs["labels"][:, -2:], torch.full((2, 2), -100))
    torch.testing.assert_close(loss_inputs["routed_experts"][:, -2:], torch.full((2, 2, 1, 1), -1, dtype=torch.int16))
    torch.testing.assert_close(loss_inputs["scores"], torch.tensor([0.25, 0.5]))
    torch.testing.assert_close(loss_inputs["global_scale"], torch.tensor(0.75))


@pytest.mark.parametrize(
    ("cp_size", "ep_shard_size", "expected_names"),
    [(1, 3, ("ep",)), (2, 3, ("ep", "ep_shard")), (2, 1, ("ep",))],
)
def test_hybridep_resolver_uses_ordered_moe_mesh_axes(
    monkeypatch,
    cp_size,
    ep_shard_size,
    expected_names,
):
    """CP equalization spans the MoE stage in deterministic EP-then-shard order."""
    calls = []

    class _GroupMesh:
        def __init__(self, name, size):
            self.name = name
            self._size = size

        def size(self):
            return self._size

        def get_group(self):
            calls.append(self.name)
            return self.name

    mesh_context = SimpleNamespace(
        cp_size=cp_size,
        moe_mesh=_NamedMesh(
            ("ep_shard", "ep"),
            ep=_GroupMesh("ep", 2),
            ep_shard=_GroupMesh("ep_shard", ep_shard_size),
        ),
    )
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    groups = engine_module._resolve_hybridep_equalization_groups(True, mesh_context)

    assert groups == expected_names
    assert tuple(calls) == expected_names


def test_model_owned_hybridep_cp_runs_consensus_but_skips_engine_padding(monkeypatch):
    """DSV4-style owners validate rank symmetry before their model-owned equalization."""
    engine = object.__new__(Engine)
    engine.pipeline = None
    engine._pipeline_uses_hybridep = False
    engine._hybridep_equalization_groups = (object(),)
    engine._model_owns_hybridep_packed_cp_equalization = True
    engine._cp_size = lambda: 2
    engine.device = torch.device("cpu")
    calls = []

    def record_all_reduce(tensor, *, op, group):
        """Record one metadata consensus reduction without changing its value.

        Args:
            tensor: Tensor of shape ``[10]`` containing packed-layout metadata
                followed by its negation.
            op: Reduction operation selected by Engine.
            group: HybridEP equalization process group.

        Returns:
            None; ``tensor`` is intentionally left unchanged.
        """
        calls.append((op, group))

    monkeypatch.setattr(dist, "all_reduce", record_all_reduce)

    assert (
        engine._hybridep_equalization_target(
            {
                # A model-owned source layout need not satisfy the generic
                # raw-THD or one-row contract.
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "qkv_format": "thd",
            }
        )
        is None
    )
    assert calls == [(dist.ReduceOp.MAX, engine._hybridep_equalization_groups[0])]


def test_equal_shape_padded_hybridep_is_a_noop(monkeypatch):
    """Equal padded extents keep legacy masks and model fields untouched."""
    engine = object.__new__(Engine)
    engine.pipeline = None
    engine._pipeline_uses_hybridep = False
    engine._hybridep_equalization_groups = (object(),)
    engine.device = torch.device("cpu")
    model_inputs = {
        "input_ids": torch.ones((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 1, 3, 3), dtype=torch.bool),
        "custom_token_field": torch.ones((2, 3)),
    }
    monkeypatch.setattr(dist, "all_reduce", lambda *_args, **_kwargs: None)

    assert engine._hybridep_equalization_target(model_inputs) is None
    assert "padding_mask" not in model_inputs
    assert model_inputs["attention_mask"].shape == (2, 1, 3, 3)


def test_raw_thd_packed_collater_is_prepared_by_context_parallel_sharder():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    seen = {}
    expected_routes = torch.tensor([[[0], [10]], [[1], [11]], [[2], [12]]], dtype=torch.int16)

    @contextmanager
    def batch_context(model_inputs, loss_fn_inputs):
        """Check the Engine's final THD ``[tokens, layers, topk]`` route layout.

        Args:
            model_inputs: Final THD model mapping with ``input_ids [tokens]``.
            loss_fn_inputs: Final THD side channels with
                ``routed_experts [tokens, layers, topk]``.

        Yields:
            ``None`` while the packed forward and backward execute.
        """
        assert model_inputs["input_ids"].shape == (3,)
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], expected_routes)
        yield

    def loss_fn(output, inputs):
        """Return one loss per local THD token.

        Args:
            output: Model values with shape ``[tokens]``.
            inputs: Loss mapping containing ``weights [tokens]`` and
                ``routed_experts [tokens, layers, topk]``.

        Returns:
            Unreduced token loss with shape ``[tokens]``.
        """
        seen.update(inputs)
        assert inputs["weights"].shape == output.shape == (3,)
        return output

    datums = [
        Datum(
            input_ids=torch.tensor([1, 2]),
            loss_fn_inputs={
                "weights": torch.ones(2),
                "routed_experts": expected_routes[:2],
            },
            loss_fn_input_layouts={"routed_experts": LossInputLayout.PER_TOKEN},
            loss_fn_input_pad_values={"routed_experts": -1},
        ),
        Datum(
            input_ids=torch.tensor([3]),
            loss_fn_inputs={
                "weights": torch.ones(1),
                "routed_experts": expected_routes[2:],
            },
            loss_fn_input_layouts={"routed_experts": LossInputLayout.PER_TOKEN},
            loss_fn_input_pad_values={"routed_experts": -1},
        ),
    ]

    result = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=partial(collate_datums, packed=True),
        batch_context_fn=batch_context,
    ).forward_backward(datums, loss_fn)

    assert result.loss.item() == pytest.approx(2.0)
    assert "seq_lens" not in seen
    assert "seq_lens_padded" not in seen
    assert seen["cu_seqlens"].tolist() == [0, 2, 3]
    torch.testing.assert_close(seen["routed_experts"], expected_routes)


def test_raw_thd_requires_a_thd_capable_context_parallel_sharder():
    model = ScaleModel()

    with pytest.raises(ValueError, match="could not prepare raw THD inputs"):
        Engine(
            model,
            device="cpu",
            microbatch_size=2,
            collate_fn=partial(collate_datums, packed=True),
        ).forward_backward([_datum([1, 2]), _datum([3])], _identity_loss)

    assert model.forward_calls == 0


def test_context_parallel_shards_model_and_rl_loss_inputs_together():
    cp_context_active = False
    batch_context_active = False

    @contextmanager
    def cp_context():
        nonlocal cp_context_active
        cp_context_active = True
        try:
            yield
        finally:
            cp_context_active = False

    def shard_batch(*args, **kwargs):
        _, batch, layout = shard_batch_contiguous(*args, pad_multiple=1, **kwargs)
        return cp_context, batch, layout

    class CPModel(ScaleModel):
        def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
            assert num_chunks == 1
            return {
                "cp_sharder": ContextParallelSharder(
                    shard_batch=shard_batch,
                    local_token_global_indices=contiguous_local_indices,
                )
            }

        def forward(self, input_ids, *args, **kwargs):
            """Scale one CP-local token shard while both runtime contexts are active.

            Args:
                input_ids: CP-local token ids with shape ``[B, S_local]``.
                *args: Additional positional model arguments forwarded to the toy model.
                **kwargs: Additional model inputs; loss-only routing data must be absent.

            Returns:
                Per-token values with shape ``[B, S_local]``.
            """
            assert cp_context_active
            assert batch_context_active
            assert "routed_experts" not in kwargs
            assert input_ids.tolist() == [[5, 6, 9, 9]]
            return super().forward(input_ids, *args, **kwargs)

    model = CPModel()
    mesh = _CPMesh(size=2, rank=1)
    mesh_context = SimpleNamespace(pp_size=1, cp_size=2, device_mesh=mesh)
    token_ids = torch.arange(6, dtype=torch.int16)
    routed_experts = torch.stack((token_ids, token_ids + 10), dim=-1).unsqueeze(0).unsqueeze(-1)
    expected_local_routes = torch.tensor(
        [[[[4], [14]], [[5], [15]], [[-1], [-1]], [[-1], [-1]]]],
        dtype=torch.int16,
    )
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])},
        loss_fn_inputs={
            "target_tokens": torch.tensor([[11, 12, 13, 14, 15, 16]]),
            "weights": torch.ones(1, 6),
            "advantages": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
            "routed_experts": routed_experts,
        },
        loss_fn_input_layouts={
            "target_tokens": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
            "advantages": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    @contextmanager
    def batch_context(model_inputs, loss_fn_inputs):
        """Validate CP-local routing data and mark the batch context active.

        Args:
            model_inputs: CP-local model mapping with ``input_ids`` shaped
                ``[B, S_local]``.
            loss_fn_inputs: CP-local loss mapping with ``routed_experts`` shaped
                ``[B, S_local, G, K]``.

        Yields:
            None while model forward, loss, and backward execute.
        """
        nonlocal batch_context_active
        assert cp_context_active
        assert not batch_context_active
        assert "routed_experts" not in model_inputs
        torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[5, 6, 9, 9]]))
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], expected_local_routes)
        batch_context_active = True
        try:
            yield
        finally:
            batch_context_active = False

    engine = Engine(
        model,
        device="cpu",
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
        padding_token_id=9,
        batch_context_fn=batch_context,
    )
    # This is a layout-only CPU test with a fake mesh. Distributed CP loss and
    # gradient scaling are covered separately with a real process group.
    engine._dp_group_and_size = lambda: (None, 1)
    engine._gradient_group_and_size = lambda _group, _size: (None, 1)

    def assert_backward_context(grad):
        """Require both runtime contexts while propagating a scalar parameter gradient.

        Args:
            grad: Scalar gradient for the toy model weight.

        Returns:
            The unchanged scalar gradient.
        """
        if not cp_context_active or not batch_context_active:
            pytest.fail("CP or batch context ended before backward")
        return grad

    model.weight.register_hook(assert_backward_context)

    def loss_fn(output, inputs):
        """Return CP-local per-token losses after checking aligned side inputs.

        Args:
            output: CP-local model values with shape ``[B, S_local]``.
            inputs: CP-local loss mapping whose token tensors share
                ``[B, S_local]`` and whose routes add trailing ``[G, K]`` axes.

        Returns:
            Per-token losses with shape ``[B, S_local]``.
        """
        assert cp_context_active
        assert batch_context_active
        assert inputs["target_tokens"].tolist() == [[15, 16, 0, 0]]
        assert inputs["weights"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
        torch.testing.assert_close(inputs["advantages"], torch.tensor([[0.5, 0.6, 0.0, 0.0]]))
        torch.testing.assert_close(inputs["routed_experts"], expected_local_routes)
        return output

    result = engine.forward_backward([datum], loss_fn)

    assert result.loss.item() == pytest.approx(11 / 6)
    assert model.weight.grad.item() == pytest.approx(11 / 6)
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(2.0)
    assert not cp_context_active
    assert not batch_context_active


def test_context_parallel_rejects_legacy_non_token_weights():
    class CPModel(ScaleModel):
        def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
            assert num_chunks == 1
            return {
                "cp_sharder": ContextParallelSharder(
                    shard_batch=lambda *args, **kwargs: shard_batch_contiguous(
                        *args,
                        pad_multiple=1,
                        **kwargs,
                    ),
                    local_token_global_indices=contiguous_local_indices,
                )
            }

    model = CPModel()
    mesh_context = SimpleNamespace(pp_size=1, cp_size=2, device_mesh=_CPMesh(size=2, rank=0))
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
        loss_fn_inputs={
            "labels": torch.tensor([[2, 3, 4, -100]]),
            "weights": torch.tensor([1.0]),
        },
    )

    with pytest.raises(ValueError, match="context-parallel loss weights must match"):
        Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
        ).forward([datum], _identity_loss)
    assert model.forward_calls == 0


def test_packed_thd_rejects_legacy_non_token_weights_without_pp_splitting():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "position_ids": torch.tensor([[0, 1, 2, 3]]),
            "seq_lens": torch.tensor([[4]], dtype=torch.int32),
            "seq_lens_padded": torch.tensor([[4]], dtype=torch.int32),
            "qkv_format": "thd",
        },
        loss_fn_inputs={
            "labels": torch.tensor([[2, 3, 4, -100]]),
            "weights": torch.tensor([1.0]),
        },
    )

    with pytest.raises(ValueError, match="packed THD.*token-aligned loss weights"):
        Engine(model, device="cpu", collate_fn=collate_prebatched).forward([datum], _identity_loss)
    assert model.forward_calls == 0


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
def test_context_parallel_prepares_mtp_futures_before_sharding(execution):
    class MTPModel(ScaleModel):
        def __init__(self):
            super().__init__()
            self.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=True, supports_mtp_cp_pp=False)
            self.cp_inputs_prepared = False
            self.mtp_forward_inputs = None

        def prepare_model_inputs_for_cp(self, batch, *, num_chunks):
            assert num_chunks == 1
            positions = torch.arange(batch["input_ids"].shape[1])
            batch["position_ids"] = torch.stack((positions, positions + 100)).unsqueeze(1)
            self.cp_inputs_prepared = True
            return {
                "cp_sharder": ContextParallelSharder(
                    shard_batch=partial(shard_batch_contiguous, pad_multiple=1),
                    local_token_global_indices=contiguous_local_indices,
                )
            }

        def prepare_mtp_inputs_for_cp(self, batch, *, ignore_index):
            assert self.cp_inputs_prepared
            assert ignore_index == -7
            return prepare_mtp_context_parallel_inputs(batch, num_depths=2, ignore_index=ignore_index)

        def forward(
            self,
            input_ids,
            *,
            mtp_per_depth_input_ids,
            mtp_per_depth_position_ids,
            mtp_per_depth_valid_masks,
            **kwargs,
        ):
            self.mtp_forward_inputs = (
                mtp_per_depth_input_ids,
                mtp_per_depth_position_ids,
                mtp_per_depth_valid_masks,
            )
            return super().forward(input_ids, **kwargs)

    model = MTPModel()
    mesh_context = SimpleNamespace(pp_size=1, cp_size=2, device_mesh=_CPMesh(size=2, rank=0))
    datum = Datum(
        model_inputs={
            "input_ids": torch.arange(10, 18).unsqueeze(0),
            "seq_lens_padded": torch.tensor([[4, 4]], dtype=torch.int32),
        },
        loss_fn_inputs={
            "labels": torch.arange(20, 28).unsqueeze(0),
            "weights": torch.ones(1, 8),
        },
    )

    engine = Engine(
        model,
        device="cpu",
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
        mtp_ignore_index=-7,
    )
    engine._dp_group_and_size = lambda: (None, 1)
    engine._gradient_group_and_size = lambda _group, _size: (None, 1)
    captured_loss_inputs = {}

    def loss_fn(output, loss_inputs):
        captured_loss_inputs.update(loss_inputs)
        return output

    result = getattr(engine, execution)([datum], loss_fn)

    input_ids, position_ids, valid_masks = model.mtp_forward_inputs

    assert [value.tolist() for value in input_ids] == [
        [[11, 12, 13, 0]],
        [[12, 13, 0, 0]],
    ]
    assert [value.tolist() for value in position_ids] == [
        [[[1, 2, 3, 0]], [[101, 102, 103, 0]]],
        [[[2, 3, 0, 0]], [[102, 103, 0, 0]]],
    ]
    assert [value.tolist() for value in valid_masks] == [
        [[True, True, True, False]],
        [[True, True, False, False]],
    ]
    assert captured_loss_inputs["labels"].tolist() == [[20, 21, 22, 23]]
    assert [value.tolist() for value in captured_loss_inputs["mtp_per_depth_targets"]] == [
        [[21, 22, 23, -7]],
        [[22, 23, -7, -7]],
    ]
    if execution == "forward":
        assert result.loss_sum.item() == pytest.approx(46)
        assert result.weight_sum.item() == pytest.approx(8)
        assert model.weight.grad is None
    else:
        assert result.loss.item() == pytest.approx(46 / 8)
        assert result.loss_sum.item() == pytest.approx(46)
        assert result.weight_sum.item() == pytest.approx(8)
        assert model.weight.grad.item() == pytest.approx(46 / 8)


def test_context_parallel_rejects_mtp_without_model_capability():
    class UnsupportedMTPModel(_DistributedCPModel):
        supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=False, supports_mtp_cp_pp=False)

    datum = Datum(
        model_inputs={"input_ids": torch.arange(8).unsqueeze(0)},
        loss_fn_inputs={"labels": torch.arange(8).unsqueeze(0), "weights": torch.ones(1, 8)},
    )
    engine = Engine(
        UnsupportedMTPModel(),
        device="cpu",
        mesh_context=SimpleNamespace(pp_size=1, cp_size=2, device_mesh=_CPMesh(size=2, rank=0)),
        collate_fn=collate_prebatched,
    )

    with pytest.raises(NotImplementedError, match="does not support MTP with context parallelism"):
        engine._prepare_batch([datum], num_pipeline_microbatches=1)


def test_context_pipeline_parallel_rejects_mtp_without_combined_capability():
    class CPOnlyMTPModel(_DistributedCPModel):
        supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=True, supports_mtp_cp_pp=False)

        def prepare_mtp_inputs_for_cp(self, batch, *, ignore_index):
            return prepare_mtp_context_parallel_inputs(batch, num_depths=1, ignore_index=ignore_index)

    model = CPOnlyMTPModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=1)
    engine = Engine(
        pipeline,
        device="cpu",
        mesh_context=SimpleNamespace(pp_size=2, cp_size=2, device_mesh=_CPMesh(size=2, rank=0)),
        collate_fn=collate_prebatched,
    )

    with pytest.raises(NotImplementedError, match="MTP with context and pipeline parallelism"):
        engine._validate_parallelism()
    assert model.forward_calls == 0


def test_weights_mask_loss_and_denominator():
    model = ScaleModel()
    result = Engine(model, device="cpu").forward_backward(
        [_datum([1, 100], [1.0, 0.0]), _datum([3, 5], [0.5, 1.0])],
        _identity_loss,
    )

    assert result.loss.item() == pytest.approx(3.0)
    assert result.loss_sum.item() == pytest.approx(7.5)
    assert result.weight_sum.item() == pytest.approx(2.5)
    assert model.weight.grad.item() == pytest.approx(3.0)


def test_fractional_weight_sum_below_one_is_not_clamped():
    model = ScaleModel()
    result = Engine(model, device="cpu").forward_backward(
        [_datum([2, 4], [0.2, 0.3])],
        _identity_loss,
    )

    assert result.loss.item() == pytest.approx(3.2)
    assert result.loss_sum.item() == pytest.approx(1.6)
    assert result.weight_sum.item() == pytest.approx(0.5)
    assert model.weight.grad.item() == pytest.approx(3.2)


def test_loss_fn_outputs_follow_datum_order_and_are_detached():
    model = ScaleModel()

    def loss_with_outputs(output, _loss_inputs):
        return output, [{"first_token": row.flatten()[0], "model_value": row.sum()} for row in output]

    result = Engine(model, device="cpu", microbatch_size=2).forward_backward(
        [_datum([1, 2]), _datum([3]), _datum([4])],
        loss_with_outputs,
    )

    assert [item["first_token"].item() for item in result.loss_fn_outputs] == [1, 3, 4]
    assert all(not item["model_value"].requires_grad for item in result.loss_fn_outputs)


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
def test_typed_batch_outputs_split_token_fields_into_datum_records(execution):
    """The callback describes one collated token tensor; Engine owns Datum splitting."""
    datums = [_datum([1, 2]), _datum([3, 4, 5])]

    def loss_with_outputs(output, _loss_inputs):
        token_probe = torch.stack((output, output + 100), dim=-1)
        return output, LossFnOutputBatch(
            per_token={"token_probe": PerTokenOutput(token_probe, fill_value=-1.0)},
            per_datum=[{"sample_id": torch.tensor(11)}, {"sample_id": torch.tensor(22)}],
        )

    result = getattr(Engine(ScaleModel(), device="cpu", microbatch_size=2), execution)(datums, loss_with_outputs)
    outputs = result.loss_fn_outputs

    assert [item["sample_id"].item() for item in outputs] == [11, 22]
    torch.testing.assert_close(outputs[0]["token_probe"], torch.tensor([[1.0, 101.0], [2.0, 102.0]]))
    torch.testing.assert_close(
        outputs[1]["token_probe"],
        torch.tensor([[3.0, 103.0], [4.0, 104.0], [5.0, 105.0]]),
    )
    assert all(not item["token_probe"].requires_grad for item in outputs)


def test_raw_thd_multirow_outputs_restore_trailing_features_in_datum_order():
    """Raw [B, S] THD coordinates flatten back to one routed token stream."""
    datums = [_datum([1, 2, 3]), _datum([4, 5])]

    def raw_thd_collate(_datums):
        token_rows = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        weights = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        return (
            {
                "input_ids": token_rows,
                "position_ids": torch.arange(4).expand(2, -1),
                "seq_lens": torch.tensor([[3], [2]], dtype=torch.int32),
                "seq_lens_padded": torch.tensor([[4], [4]], dtype=torch.int32),
                "qkv_format": "thd",
            },
            CollatedLossInputs(
                {"weights": weights},
                layouts={"weights": LossInputLayout.PER_TOKEN},
                item_to_datum=(0, 1),
            ),
        )

    def loss_with_outputs(output, loss_inputs):
        assert output.shape == loss_inputs["weights"].shape == (8,)
        probe = torch.stack((output, output + 100), dim=-1)
        return output, LossFnOutputBatch(per_token={"probe": PerTokenOutput(probe)})

    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    result = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=raw_thd_collate,
    ).forward(datums, loss_with_outputs)

    torch.testing.assert_close(
        result.loss_fn_outputs[0]["probe"], torch.tensor([[1.0, 101.0], [2.0, 102.0], [3.0, 103.0]])
    )
    torch.testing.assert_close(result.loss_fn_outputs[1]["probe"], torch.tensor([[4.0, 104.0], [5.0, 105.0]]))


def test_typed_batch_outputs_follow_left_and_noncontiguous_attention_masks():
    datums = [_datum([10, 11]), _datum([20, 21, 22])]

    def masked_collate(_datums):
        input_ids = torch.tensor(
            [
                [99, 99, 10, 11, 99],
                [20, 99, 21, 99, 22],
            ]
        )
        attention_mask = torch.tensor(
            [
                [False, False, True, True, False],
                [True, False, True, False, True],
            ]
        )
        return (
            {"input_ids": input_ids, "attention_mask": attention_mask},
            CollatedLossInputs(
                {"weights": attention_mask.to(torch.float32)},
                layouts={"weights": LossInputLayout.PER_TOKEN},
                item_to_datum=(0, 1),
            ),
        )

    def loss_with_outputs(output, _loss_inputs):
        probe = torch.stack((output, output + 100), dim=-1)
        return output, LossFnOutputBatch(per_token={"probe": PerTokenOutput(probe)})

    result = Engine(
        ScaleModel(),
        device="cpu",
        microbatch_size=2,
        collate_fn=masked_collate,
    ).forward(datums, loss_with_outputs)

    torch.testing.assert_close(result.loss_fn_outputs[0]["probe"], torch.tensor([[10.0, 110.0], [11.0, 111.0]]))
    torch.testing.assert_close(
        result.loss_fn_outputs[1]["probe"],
        torch.tensor([[20.0, 120.0], [21.0, 121.0], [22.0, 122.0]]),
    )


def test_padded_output_routing_keeps_attention_mask_compact():
    mask = torch.tensor([[False, True, True], [True, False, True]])
    real_lengths, padded_lengths, token_mask = engine_module._output_sequence_lengths(
        [_datum([1, 2]), _datum([3, 4])],
        {"attention_mask": mask},
        {"weights": mask.to(torch.float32)},
        (0, 1),
        is_thd=False,
    )

    assert real_lengths == (2, 2)
    assert padded_lengths == (3, 3)
    assert isinstance(token_mask, torch.Tensor)
    assert token_mask.dtype is torch.bool
    torch.testing.assert_close(token_mask, mask)


def test_legacy_per_datum_output_tensors_remain_opaque():
    """A vector whose length resembles a token axis must not opt into restoration by shape."""
    opaque = torch.tensor([91.0, 92.0, 93.0])

    result = Engine(ScaleModel(), device="cpu", microbatch_size=2).forward(
        [_datum([1, 2]), _datum([3, 4, 5])],
        lambda output, _inputs: (output, [{"opaque": opaque.clone()} for _ in output]),
    )

    assert len(result.loss_fn_outputs[0]["opaque"]) == 3
    assert len(result.loss_fn_outputs[1]["opaque"]) == 3
    assert all(torch.equal(item["opaque"], opaque) for item in result.loss_fn_outputs)


def test_typed_batch_output_rejects_a_mismatched_token_shape():
    def loss_with_bad_shape(output, _loss_inputs):
        return output, LossFnOutputBatch(
            per_token={"bad_probe": PerTokenOutput(output[:, :-1])},
            per_datum=[{}, {}],
        )

    with pytest.raises(ValueError, match=r"bad_probe.*shape|shape.*bad_probe"):
        Engine(ScaleModel(), device="cpu", microbatch_size=2).forward(
            [_datum([1, 2]), _datum([3, 4])],
            loss_with_bad_shape,
        )


def test_output_only_parse_error_is_reported_before_single_rank_backward():
    model = ScaleModel()

    with pytest.raises(ValueError, match="outputs must be a sequence of mappings"):
        Engine(model, device="cpu").forward_backward(
            [_datum([1, 2])],
            lambda output, _inputs: (output, "not-records"),
        )

    assert model.weight.grad is None


def test_typed_batch_output_rejects_explicit_empty_per_datum_records():
    def loss_with_empty_records(output, _loss_inputs):
        return output, LossFnOutputBatch(
            per_token={"probe": PerTokenOutput(output)},
            per_datum=[],
        )

    with pytest.raises(ValueError, match=r"per_datum contains 0 records, expected 2"):
        Engine(ScaleModel(), device="cpu", microbatch_size=2).forward(
            [_datum([1]), _datum([2])],
            loss_with_empty_records,
        )


@pytest.mark.parametrize(
    ("cu_seqlens", "cu_seqlens_padded", "match"),
    [
        (
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            "0 <= real_length <= padded_length",
        ),
        (
            torch.tensor([0, 4], dtype=torch.int32),
            torch.tensor([0, 5], dtype=torch.int32),
            r"padded lengths sum to 5.*loss weights.*4",
        ),
    ],
    ids=("real-exceeds-padded", "padded-sum-mismatch"),
)
def test_typed_batch_output_rejects_invalid_final_thd_lengths_before_forward(
    cu_seqlens,
    cu_seqlens_padded,
    match,
):
    model = ScaleModel()

    def bad_thd_collate(_datums):
        return (
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "position_ids": torch.arange(4),
                "cu_seqlens": cu_seqlens,
                "cu_seqlens_padded": cu_seqlens_padded,
                "max_seqlen": torch.tensor(4, dtype=torch.int32),
                "qkv_format": "thd",
            },
            CollatedLossInputs(
                {"weights": torch.ones(4)},
                layouts={"weights": LossInputLayout.PER_TOKEN},
                item_to_datum=(0,),
            ),
        )

    with pytest.raises(ValueError, match=match):
        Engine(model, device="cpu", collate_fn=bad_thd_collate).forward(
            [_datum([1, 2, 3, 4])],
            _identity_loss,
        )
    assert model.forward_calls == 0


def test_typed_batch_output_rejects_per_token_and_per_datum_key_collision():
    def loss_with_duplicate_key(output, _loss_inputs):
        return output, LossFnOutputBatch(
            per_token={"score": PerTokenOutput(output)},
            per_datum=[{"score": torch.tensor(1.0)}, {"score": torch.tensor(2.0)}],
        )

    with pytest.raises(ValueError, match=r"score.*(?:both|conflict|duplicate)|(?:both|conflict|duplicate).*score"):
        Engine(ScaleModel(), device="cpu", microbatch_size=2).forward(
            [_datum([1, 2]), _datum([3, 4])],
            loss_with_duplicate_key,
        )


def test_loss_fn_outputs_must_align_with_datums():
    model = ScaleModel()
    with pytest.raises(ValueError, match="one mapping per Datum"):
        Engine(model, device="cpu", microbatch_size=2).forward_backward(
            [_datum([1]), _datum([2])],
            lambda output, _inputs: (output, [{"only": "one"}]),
        )
    assert model.weight.grad is None


def test_loss_fn_outputs_must_be_consistent_across_the_window():
    calls = 0

    def inconsistent_outputs(output, _inputs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return output, [{"value": output.sum()}]
        return output

    with pytest.raises(ValueError, match="every microbatch or none"):
        Engine(ScaleModel(), device="cpu").forward_backward(
            [_datum([1]), _datum([2])],
            inconsistent_outputs,
        )


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.output = nn.Linear(4, 8)

    def forward(self, input_ids, **_):
        return self.output(self.embedding(input_ids))


def test_raw_output_and_loss_inputs_support_an_rl_loss_callback():
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([1, 2, 3])},
        loss_fn_inputs={
            "target_tokens": torch.tensor([2, 3, 4]),
            "weights": torch.tensor([1.0, 1.0, 0.0]),
            "logprobs": torch.tensor([-1.0, -1.0, 0.0]),
            "advantages": torch.tensor([0.5, -0.25, 0.0]),
        },
    )
    model = TinyLM()

    def policy_loss(logits, inputs):
        new_logprobs = -F.cross_entropy(
            logits.flatten(0, 1),
            inputs["target_tokens"].flatten(),
            reduction="none",
        ).view_as(inputs["weights"])
        ratio = torch.exp(new_logprobs - inputs["logprobs"])
        losses = -(ratio * inputs["advantages"])
        return losses, [{"policy_sum": (losses * inputs["weights"]).sum()}]

    result = Engine(model, device="cpu").forward_backward([datum], policy_loss)

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.loss_fn_outputs[0]["policy_sum"])
    assert not result.loss_fn_outputs[0]["policy_sum"].requires_grad
    assert model.embedding.weight.grad is not None
    assert model.output.weight.grad is not None


def test_lifecycle_marks_only_the_last_microbatch_for_sync(monkeypatch):
    events = []

    monkeypatch.setattr(
        engine_module, "prepare_for_grad_accumulation", lambda *_args, **_kwargs: events.append("prepare")
    )
    monkeypatch.setattr(engine_module, "prepare_after_first_microbatch", lambda: events.append("after_first"))
    monkeypatch.setattr(engine_module, "prepare_for_final_backward", lambda *_args, **_kwargs: events.append("final"))

    @contextmanager
    def sync_context(_model, is_last, _defer):
        events.append(f"sync:{is_last}")
        yield

    monkeypatch.setattr(engine_module, "get_sync_ctx", sync_context)

    Engine(ScaleModel(), device="cpu").forward_backward(
        [_datum([1]), _datum([2])],
        _identity_loss,
    )

    assert events == ["prepare", "sync:False", "after_first", "final", "sync:True"]


def test_pipeline_window_uses_schedule_microbatches_and_global_normalization():
    active = False
    context_events = []
    backward_calls = 0

    @contextmanager
    def forward_context(_model_inputs):
        nonlocal active
        assert not active
        active = True
        context_events.append("enter")
        try:
            yield
        finally:
            active = False
            context_events.append("exit")

    class PipelineModel(ScaleModel):
        def forward(self, input_ids, **kwargs):
            assert active
            return super().forward(input_ids, **kwargs)

    model = PipelineModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)

    def check_backward_context(grad):
        nonlocal backward_calls
        assert active
        backward_calls += 1
        return grad

    model.weight.register_hook(check_backward_context)

    def loss_fn(output, inputs):
        assert active
        assert output.shape == inputs["weights"].shape == (1, 2)
        return output

    result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        collate_fn=collate_prebatched,
        context_fn=forward_context,
    ).forward_backward(
        [
            _datum([[1, 2], [3, 4]]),
            _datum([[5, 6], [7, 8]]),
        ],
        loss_fn,
    )

    assert result.loss.item() == pytest.approx(4.5)
    assert model.weight.grad.item() == pytest.approx(4.5)
    assert result.loss_fn_outputs == []
    assert pipeline.step_calls == 2
    # The fake schedule performs and counts every backward, then returns None.
    # A second Engine-owned backward would either fail or change these counts.
    assert pipeline.backward_calls == backward_calls == 4
    assert model.forward_calls == 4
    assert pipeline.updated_seq_lens == [2, 2]
    assert [item["input_ids"].shape for call in pipeline.prepared_inputs for item in call] == [
        (1, 2),
        (1, 2),
        (1, 2),
        (1, 2),
    ]
    scaled_losses = torch.stack(pipeline.callback_losses)
    torch.testing.assert_close(
        scaled_losses,
        torch.tensor([3 / 8, 7 / 8, 11 / 8, 15 / 8], dtype=scaled_losses.dtype),
    )
    assert context_events == ["enter", "exit", "enter", "exit"]
    assert not active


def test_pipeline_forward_uses_eval_microbatches_without_backward_and_orders_outputs():
    model = ScaleModel()
    other_part = ScaleModel()
    pipeline = _FakeAutoPipeline(
        model,
        parts=[model, other_part],
        num_microbatches=2,
        callback_order=[1, 0],
    )

    result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=2,
    ).forward(
        [_datum([1, 2]), _datum([3, 4])],
        lambda output, _inputs: (output, [{"first_token": output.flatten()[0]}]),
    )

    assert result.loss_sum.item() == pytest.approx(10.0)
    assert result.weight_sum.item() == pytest.approx(4.0)
    assert [item["first_token"].item() for item in result.loss_fn_outputs] == [1.0, 3.0]
    assert pipeline.eval_calls == 1
    assert pipeline.step_calls == 0
    assert pipeline.backward_calls == 0
    assert model.weight.grad is None
    assert model.forward_calls == 2
    assert not model.training
    assert not other_part.training
    torch.testing.assert_close(torch.stack(pipeline.callback_losses), torch.tensor([7.0, 3.0]))


def test_forward_does_not_apply_backward_only_parallelism_restrictions():
    model = ScaleModel()
    model.calculate_per_token_loss = True
    eager_result = Engine(model, device="cpu").forward([_datum([1])], _identity_loss)

    pipeline_model = ScaleModel()
    pipeline = _FakeAutoPipeline(pipeline_model, num_microbatches=1, scale_grads=True)
    pipeline_result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
    ).forward([_datum([2])], _identity_loss)

    assert eager_result.loss_sum.item() == 1.0
    assert pipeline_result.loss_sum.item() == 2.0
    assert pipeline.eval_calls == 1
    assert pipeline.step_calls == 0


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
def test_pipeline_stage_metadata_prevents_real_forward_from_resetting_media_cursor(execution):
    class MediaModel(ScaleModel):
        def __init__(self):
            super().__init__()
            self.consumed_media = []

        def forward(self, input_ids, **kwargs):
            chunk = self._vlm_pixel_values_chunks[self._vlm_chunk_idx]
            self._vlm_chunk_idx += 1
            self.consumed_media.append(int(chunk.item()))
            assert int(input_ids.item()) == int(chunk.item())
            return super().forward(input_ids, **kwargs)

    model = MediaModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    stage = SimpleNamespace(is_first=True, _user_meta=None)
    schedule = SimpleNamespace(_stage_forward_initialized=False)
    pipeline._info = SimpleNamespace(
        has_first_stage=True,
        has_last_stage=True,
        stages=[stage],
        schedule=schedule,
    )
    original_update_seq_len = pipeline.update_seq_len

    def update_seq_len(seq_len, *, microbatch_size=None, input_tensor=None):
        original_update_seq_len(seq_len, microbatch_size=microbatch_size, input_tensor=input_tensor)
        stage._user_meta = SimpleNamespace(inputs=(object(),), outputs=(object(),))

    pipeline.update_seq_len = update_seq_len

    @contextmanager
    def batch_context(model_inputs):
        with stage_vlm_media_for_pp(pipeline, [model], model_inputs):
            yield

    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1], [2]]),
            VLM_PP_MEDIA_KEY: {
                "pixel_values": [torch.tensor([1.0]), torch.tensor([2.0])],
                "image_grid_hws": [torch.ones(1), torch.ones(1)],
            },
        },
        loss_fn_inputs={"weights": torch.ones(2, 1)},
    )

    engine = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        collate_fn=collate_prebatched,
        context_fn=batch_context,
    )
    getattr(engine, execution)([datum], _identity_loss)

    assert model.consumed_media == [1, 2]
    assert pipeline.updated_seq_lens == [1]
    assert pipeline.step_calls == int(execution == "forward_backward")
    assert pipeline.eval_calls == int(execution == "forward")


def test_pipeline_lifecycle_and_moe_scale_cover_outer_and_inner_microbatches(monkeypatch):
    events = []
    model = ScaleModel()
    other_part = ScaleModel()
    model.eval()
    other_part.eval()
    pipeline = _FakeAutoPipeline(
        model,
        parts=[model, other_part],
        num_microbatches=2,
        events=events,
    )

    def prepare(parts, *, pp_enabled):
        assert parts == [model, other_part]
        assert pp_enabled is True
        events.append("prepare")

    def prepare_final(parts, *, pp_enabled):
        assert parts == [model, other_part]
        assert pp_enabled is True
        events.append("final")

    monkeypatch.setattr(engine_module, "prepare_for_grad_accumulation", prepare)
    monkeypatch.setattr(engine_module, "prepare_for_final_backward", prepare_final)
    monkeypatch.setattr(engine_module, "prepare_after_first_microbatch", lambda: events.append("after_first"))
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)

    Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        collate_fn=collate_prebatched,
    ).forward_backward(
        [
            _datum([[1], [2]]),
            _datum([[3], [4]]),
        ],
        _identity_loss,
    )

    assert events == ["prepare", "step", "after_first", "final", "step"]
    assert model.training
    assert other_part.training
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(0.25)


def test_pipeline_outputs_follow_logical_microbatch_order():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])

    result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=2,
    ).forward_backward(
        [_datum([1]), _datum([2])],
        lambda output, _inputs: (output, [{"metric": output.sum()}]),
    )

    assert pipeline.step_calls == 1
    assert pipeline.backward_calls == 2
    assert [item["metric"].item() for item in result.loss_fn_outputs] == [1.0, 2.0]


def _packed_layout_datums(lengths: list[int]) -> list[Datum]:
    """Build flat Datums whose three loss layouts are easy to distinguish."""
    datums = []
    token_start = 1
    for datum_index, length in enumerate(lengths):
        input_ids = torch.arange(token_start, token_start + length)
        datums.append(
            Datum(
                model_inputs={"input_ids": input_ids},
                loss_fn_inputs={
                    "weights": torch.ones(length),
                    "advantages": input_ids.to(torch.float32) * 10,
                    "old_logprobs": -input_ids.to(torch.float32),
                    "sample_id": torch.tensor((datum_index + 1) * 11),
                    # Its leading extent deliberately collides with PP=2. A
                    # shape-based splitter would silently turn this into one
                    # value per microbatch instead of replicating it intact.
                    "global_coefficients": torch.tensor([701.0, 709.0]),
                },
                loss_fn_input_layouts={
                    "weights": LossInputLayout.PER_TOKEN,
                    "advantages": LossInputLayout.PER_TOKEN,
                    "old_logprobs": LossInputLayout.PER_TOKEN,
                    "sample_id": LossInputLayout.PER_DATUM,
                    "global_coefficients": LossInputLayout.REPLICATED,
                },
            )
        )
        token_start += length
    return datums


def _final_thd_layout_collate(datums: list[Datum]):
    """Convert canonical packed output to final THD without dropping layout metadata."""
    model_inputs, loss_inputs = collate_datums(datums, packed=True)
    lengths = torch.tensor([datum.seq_len for datum in datums], dtype=torch.int32)
    final_model_inputs = {
        "input_ids": model_inputs["input_ids"].reshape(-1),
        "position_ids": model_inputs["position_ids"].reshape(-1),
        "cu_seqlens": F.pad(lengths.cumsum(0), (1, 0)).to(torch.int32),
        "max_seqlen": lengths.max(),
        "qkv_format": "thd",
    }
    final_loss_inputs = {
        name: value.reshape(-1) if loss_inputs.layouts[name] is LossInputLayout.PER_TOKEN else value
        for name, value in loss_inputs.items()
    }
    return final_model_inputs, CollatedLossInputs(
        final_loss_inputs,
        layouts=loss_inputs.layouts,
        item_to_datum=loss_inputs.item_to_datum,
    )


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
@pytest.mark.parametrize(
    ("lengths", "collate_fn"),
    [
        pytest.param([2, 2, 2, 2], partial(collate_datums, packed=True), id="raw-thd-2-plus-2"),
        pytest.param([1, 1, 2, 4], _final_thd_layout_collate, id="final-thd-3-plus-1"),
    ],
)
def test_pipeline_typed_batch_outputs_split_packed_stream_in_datum_order(execution, lengths, collate_fn):
    """Packed callbacks return one stream; Engine restores records after logical PP ordering."""
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])
    datums = _packed_layout_datums(lengths)

    def loss_with_outputs(output, loss_inputs):
        probe = torch.stack((loss_inputs["advantages"], loss_inputs["old_logprobs"]), dim=-1)
        return output, LossFnOutputBatch(
            per_token={"rl_probe": PerTokenOutput(probe, fill_value=-999.0)},
            per_datum=[{"sample_id": sample_id} for sample_id in loss_inputs["sample_id"]],
        )

    result = getattr(
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            microbatch_size=4,
            collate_fn=collate_fn,
        ),
        execution,
    )(datums, loss_with_outputs)
    outputs = result.loss_fn_outputs

    assert [item["sample_id"].item() for item in outputs] == [11, 22, 33, 44]
    for datum, item in zip(datums, outputs):
        expected = torch.stack(
            (datum.loss_fn_inputs["advantages"], datum.loss_fn_inputs["old_logprobs"]),
            dim=-1,
        )
        torch.testing.assert_close(item["rl_probe"], expected)
        assert not item["rl_probe"].requires_grad


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
def test_pipeline_raw_thd_routes_explicit_loss_layouts_and_outputs_in_datum_order(execution):
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])
    datums = _packed_layout_datums([2, 2, 2, 2])
    seen = []

    def loss_with_outputs(output, loss_inputs):
        torch.testing.assert_close(loss_inputs["global_coefficients"], torch.tensor([701.0, 709.0]))
        assert loss_inputs["global_coefficients"].shape == (2,)
        torch.testing.assert_close(loss_inputs["advantages"], output * 10)
        torch.testing.assert_close(loss_inputs["old_logprobs"], -output)
        sample_ids = loss_inputs["sample_id"].clone()
        seen.append(sample_ids.tolist())
        return output, [{"sample_id": sample_id} for sample_id in sample_ids]

    engine = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=4,
        collate_fn=partial(collate_datums, packed=True),
    )
    result = getattr(engine, execution)(datums, loss_with_outputs)

    assert seen == [[33, 44], [11, 22]]
    if execution == "forward":
        assert result.loss_sum.item() == pytest.approx(36.0)
        assert result.weight_sum.item() == pytest.approx(8.0)
        assert pipeline.eval_calls == 1
        assert pipeline.step_calls == 0
        assert pipeline.backward_calls == 0
        assert model.weight.grad is None
    else:
        assert result.loss.item() == pytest.approx(4.5)
        assert result.loss_sum.item() == pytest.approx(36.0)
        assert result.weight_sum.item() == pytest.approx(8.0)
        assert pipeline.eval_calls == 0
        assert pipeline.step_calls == 1
        assert pipeline.backward_calls == 2
        assert model.weight.grad.item() == pytest.approx(4.5)
    outputs = result.loss_fn_outputs
    assert [item["sample_id"].item() for item in outputs] == [11, 22, 33, 44]


def test_pipeline_packed_legacy_collater_metadata_stripping_fails_closed():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    datums = _packed_layout_datums([2, 2, 2, 2])

    def strip_layout_metadata(items):
        model_inputs, loss_inputs = collate_datums(items, packed=True)
        return model_inputs, dict(loss_inputs)

    with pytest.raises(NotImplementedError, match="item_to_datum metadata"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            microbatch_size=4,
            collate_fn=strip_layout_metadata,
        ).forward(datums, _identity_loss)

    assert pipeline.eval_calls == 0


def test_custom_collater_cannot_strip_loss_input_pad_values():
    model = ScaleModel()
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2]])},
        loss_fn_inputs={
            "weights": torch.ones(1, 2),
            "routed_experts": torch.tensor([[[[0]], [[-1]]]], dtype=torch.int16),
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    def strip_pad_metadata(items):
        """Return prebatched tensors while intentionally dropping padding metadata.

        Args:
            items: One prebatched Datum containing ``input_ids [B, S]`` and
                ``routed_experts [B, S, G, K]``.

        Returns:
            Model inputs and a plain loss-input mapping without pad metadata.
        """
        model_inputs, loss_inputs = collate_prebatched(items)
        return model_inputs, dict(loss_inputs)

    with pytest.raises(ValueError, match="CollatedLossInputs.*loss_fn_input_pad_values"):
        Engine(model, device="cpu", collate_fn=strip_pad_metadata).forward([datum], _identity_loss)

    assert model.forward_calls == 0


@pytest.mark.parametrize("metadata_error", ["missing_pad_value", "changed_pad_value", "changed_layout"])
def test_typed_custom_collater_cannot_change_datum_metadata(metadata_error):
    model = ScaleModel()
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2]])},
        loss_fn_inputs={
            "weights": torch.ones(1, 2),
            "routed_experts": torch.tensor([[[[0]], [[-1]]]], dtype=torch.int16),
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    def change_metadata(items):
        """Return prebatched tensors with deliberately inconsistent metadata.

        Args:
            items: One prebatched Datum containing ``input_ids [B, S]`` and
                ``routed_experts [B, S, G, K]``.

        Returns:
            Model inputs and typed loss inputs whose layout or padding
            metadata disagrees with the Datum declaration.
        """
        model_inputs, loss_inputs = collate_prebatched(items)
        layouts = dict(loss_inputs.layouts)
        pad_values = dict(loss_inputs.pad_values)
        if metadata_error == "missing_pad_value":
            pad_values.clear()
        elif metadata_error == "changed_pad_value":
            pad_values["routed_experts"] = 0
        else:
            layouts["routed_experts"] = LossInputLayout.REPLICATED
            pad_values.clear()
        return model_inputs, CollatedLossInputs(
            loss_inputs,
            layouts=layouts,
            item_to_datum=loss_inputs.item_to_datum,
            pad_values=pad_values,
        )

    with pytest.raises(ValueError, match="CollatedLossInputs (pad value|layout).+disagrees"):
        Engine(model, device="cpu", collate_fn=change_metadata).forward([datum], _identity_loss)

    assert model.forward_calls == 0


def test_pipeline_final_thd_routes_three_plus_one_datums_and_restores_output_order():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])
    datums = _packed_layout_datums([1, 1, 2, 4])
    seen = []

    def loss_with_outputs(output, loss_inputs):
        torch.testing.assert_close(loss_inputs["global_coefficients"], torch.tensor([701.0, 709.0]))
        assert loss_inputs["global_coefficients"].shape == (2,)
        torch.testing.assert_close(loss_inputs["advantages"], output * 10)
        torch.testing.assert_close(loss_inputs["old_logprobs"], -output)
        sample_ids = loss_inputs["sample_id"].clone()
        seen.append(sample_ids.tolist())
        return output, [{"sample_id": sample_id} for sample_id in sample_ids]

    result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=4,
        collate_fn=_final_thd_layout_collate,
    ).forward(datums, loss_with_outputs)

    assert result.loss_sum.item() == pytest.approx(36.0)
    assert result.weight_sum.item() == pytest.approx(8.0)
    assert seen == [[44], [11, 22, 33]]
    assert [item["sample_id"].item() for item in result.loss_fn_outputs] == [11, 22, 33, 44]


def test_pipeline_final_thd_rejects_wrong_outputs_even_when_window_total_matches():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])
    datums = _packed_layout_datums([1, 1, 2, 4])

    with pytest.raises(ValueError, match="returned 2 outputs for microbatch 0, expected 3"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            microbatch_size=4,
            collate_fn=_final_thd_layout_collate,
        ).forward(
            datums,
            lambda output, _loss_inputs: (output, [{"wrong": 0}, {"wrong": 1}]),
        )


def test_engine_rejects_collater_item_reordering_before_execution():
    datums = _packed_layout_datums([2, 2])

    def reordered_collate(items):
        model_inputs, loss_inputs = collate_datums(items)
        return model_inputs, CollatedLossInputs(
            loss_inputs,
            layouts=loss_inputs.layouts,
            item_to_datum=(1, 0),
        )

    with pytest.raises(ValueError, match="preserve outer Datum order"):
        Engine(
            ScaleModel(),
            device="cpu",
            microbatch_size=2,
            collate_fn=reordered_collate,
        ).forward(datums, _identity_loss)


def test_engine_keeps_weights_as_a_per_token_contract():
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([1])},
        loss_fn_inputs={"weights": torch.tensor([1.0])},
        loss_fn_input_layouts={"weights": LossInputLayout.PER_DATUM},
    )

    with pytest.raises(ValueError, match="weights.*PER_TOKEN"):
        Engine(ScaleModel(), device="cpu").forward([datum], _identity_loss)


def test_pipeline_single_microbatch_validates_thd_item_count():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=1)
    datum = _datum([1, 2])

    def two_sequence_collate(_items):
        return (
            {
                "input_ids": torch.tensor([1, 2]),
                "position_ids": torch.tensor([0, 0]),
                "cu_seqlens": torch.tensor([0, 1, 2], dtype=torch.int32),
                "max_seqlen": torch.tensor(1, dtype=torch.int32),
                "qkv_format": "thd",
            },
            CollatedLossInputs(
                {"weights": torch.ones(2)},
                layouts={"weights": LossInputLayout.PER_TOKEN},
                item_to_datum=(0,),
            ),
        )

    with pytest.raises(ValueError, match="item_to_datum does not match"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            collate_fn=two_sequence_collate,
        ).forward([datum], _identity_loss)


def test_pipeline_output_sync_finds_last_stage_on_physical_rank_zero(monkeypatch):
    pipeline = _FakeAutoPipeline(ScaleModel(), has_last_stage=False)
    engine = Engine(pipeline, device="cpu", mesh_context=_pipeline_mesh_context())
    pp_group = object()
    engine._pp_group_and_size = lambda: (pp_group, 2)
    expected = [{"metric": torch.tensor(7.0)}]

    def fake_all_gather_into_tensor(gathered, local, *, group):
        torch.testing.assert_close(local, torch.tensor([0, 0], dtype=torch.int64))
        assert group is pp_group
        gathered.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.int64))

    def fake_broadcast_object_list(objects, *, src, group, device):
        assert objects == [None]
        assert src == 11
        assert group is pp_group
        assert device == torch.device("cpu")
        objects[0] = pickle.dumps(expected)

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(dist, "get_rank", lambda *, group: 1)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, group_rank: 11 if group_rank == 0 else 12)
    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast_object_list)

    result = engine._broadcast_pipeline_outputs([], serialized_outputs=None)

    assert result == expected


def test_pipeline_output_sync_skips_object_broadcast_without_outputs(monkeypatch):
    pipeline = _FakeAutoPipeline(ScaleModel(), has_last_stage=False)
    engine = Engine(pipeline, device="cpu", mesh_context=_pipeline_mesh_context())
    pp_group = object()
    engine._pp_group_and_size = lambda: (pp_group, 2)

    def fake_all_gather_into_tensor(gathered, local, *, group):
        torch.testing.assert_close(local, torch.tensor([0, 0], dtype=torch.int64))
        assert group is pp_group
        gathered.copy_(torch.tensor([1, 0, 0, 0], dtype=torch.int64))

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(
        dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: pytest.fail("empty pipeline outputs must not use an object collective"),
    )

    assert engine._broadcast_pipeline_outputs([], serialized_outputs=None) == []


def test_pipeline_prebatched_outputs_require_one_inner_microbatch():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    datum = _datum([[1], [2]])

    with pytest.raises(ValueError, match="prebatched Datum may return outputs only"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            collate_fn=collate_prebatched,
        ).forward_backward(
            [datum],
            lambda output, _inputs: (output, [{"metric": output.sum()}]),
        )


def test_pipeline_prebatched_per_datum_field_without_item_mapping_is_rejected():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    datum = Datum(
        model_inputs={
            "input_ids": torch.arange(1, 9),
            "position_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            "cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32),
            "max_seqlen": torch.tensor(4, dtype=torch.int32),
            "qkv_format": "thd",
        },
        loss_fn_inputs={
            "weights": torch.ones(8),
            "sample_id": torch.tensor([17]),
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "sample_id": LossInputLayout.PER_DATUM,
        },
    )

    with pytest.raises(NotImplementedError, match="item_to_datum metadata"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            collate_fn=collate_prebatched,
        ).forward([datum], _identity_loss)

    assert pipeline.eval_calls == 0


def test_pipeline_final_thd_embeddings_use_the_token_axis_for_sequence_length():
    pipeline = _FakeAutoPipeline(ScaleModel(), num_microbatches=1)
    embeddings = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    datum = Datum(
        model_inputs={
            "inputs_embeds": embeddings,
            "position_ids": torch.arange(4),
            "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
            "max_seqlen": torch.tensor(4, dtype=torch.int32),
            "qkv_format": "thd",
        },
        loss_fn_inputs={"weights": torch.ones(4)},
    )

    Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        collate_fn=collate_prebatched,
    ).forward_backward([datum], lambda output, _inputs: output.sum())

    assert pipeline.updated_seq_lens == [4]


def test_pipeline_requires_materialized_padded_microbatch_size_to_match_config():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, pp_microbatch_size=2)

    with pytest.raises(ValueError, match="materialized pipeline microbatch has batch size 1"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            collate_fn=collate_prebatched,
        ).forward_backward([_datum([[1], [2]])], _identity_loss)

    assert pipeline.step_calls == 0
    assert model.forward_calls == 0


def test_pipeline_te_thd_keeps_arbitrary_loss_fields_aligned_through_cp(monkeypatch):
    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(_cu_seqlens, total_tokens, _cp_size, _cp_rank):
            assert total_tokens == 4
            return torch.tensor([0, 3])

    monkeypatch.setitem(sys.modules, "transformer_engine_torch", MockTex)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    mesh = _CPMesh(size=2, rank=0)
    mesh_context = SimpleNamespace(pp_size=2, cp_size=2, device_mesh=mesh, process_group=None)
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "position_ids": torch.arange(4).expand(2, -1),
            "seq_lens": torch.tensor([[4], [4]]),
            "seq_lens_padded": torch.tensor([[4], [4]]),
            "qkv_format": "thd",
        },
        loss_fn_inputs={
            "weights": torch.ones(2, 4),
            "advantages": torch.arange(1, 9, dtype=torch.float32).view(2, 4) * 10,
            "old_logprobs": -torch.arange(1, 9, dtype=torch.float32).view(2, 4),
        },
    )
    engine = Engine(
        pipeline,
        device="cpu",
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
    )
    # This CPU test exercises the exact THD layout only; real CP collectives are
    # covered by the distributed tests below.
    engine._dp_group_and_size = lambda: (None, 1)
    engine._gradient_group_and_size = lambda _group, _size: (None, 1)
    seen = []

    def loss_fn(output, inputs):
        seen.append((output.detach().clone(), inputs["advantages"].clone(), inputs["old_logprobs"].clone()))
        assert output.shape == inputs["weights"].shape == (1, 2)
        return output

    result = engine.forward_backward([datum], loss_fn)

    assert result.loss.item() == pytest.approx(18 / 8)
    assert len(pipeline.prepared_inputs) == 1
    assert [item["input_ids"].shape for item in pipeline.prepared_inputs[0]] == [(1, 2), (1, 2)]
    torch.testing.assert_close(seen[0][0], torch.tensor([[1.0, 4.0]]))
    torch.testing.assert_close(seen[0][1], torch.tensor([[10.0, 40.0]]))
    torch.testing.assert_close(seen[0][2], torch.tensor([[-1.0, -4.0]]))
    torch.testing.assert_close(seen[1][0], torch.tensor([[5.0, 8.0]]))
    torch.testing.assert_close(seen[1][1], torch.tensor([[50.0, 80.0]]))
    torch.testing.assert_close(seen[1][2], torch.tensor([[-5.0, -8.0]]))


def test_eager_te_thd_batch_context_preserves_trailing_routes_through_cp(monkeypatch):
    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(_cu_seqlens, total_tokens, _cp_size, _cp_rank):
            assert total_tokens == 8
            return torch.tensor([0, 3, 4, 7])

    monkeypatch.setitem(sys.modules, "transformer_engine_torch", MockTex)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)

    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    mesh = _CPMesh(size=2, rank=0)
    mesh_context = SimpleNamespace(pp_size=1, cp_size=2, device_mesh=mesh, process_group=None)
    routes = torch.tensor(
        [
            [[[0, 1]], [[-1, -1]], [[4, 5]], [[2, 3]]],
            [[[4, 5]], [[0, 1]], [[2, 3]], [[-1, -1]]],
        ],
        dtype=torch.int16,
    )
    expected_routes = routes.reshape(8, 1, 2).index_select(0, torch.tensor([0, 3, 4, 7]))
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            "position_ids": torch.arange(4).expand(2, -1),
            "seq_lens": torch.tensor([[4], [4]]),
            "seq_lens_padded": torch.tensor([[4], [4]]),
            "qkv_format": "thd",
        },
        loss_fn_inputs={"weights": torch.ones(2, 4), "routed_experts": routes},
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    @contextmanager
    def batch_context(model_inputs, loss_fn_inputs):
        """Check final THD token order after a fake two-way CP partition.

        Args:
            model_inputs: CP-local final THD mapping with ``input_ids [tokens]``.
            loss_fn_inputs: CP-local side channels with
                ``routed_experts [tokens, layers, topk]``.

        Yields:
            ``None`` while the eager forward and backward execute.
        """
        torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([1, 4, 5, 8]))
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], expected_routes)
        yield

    def loss_fn(output, loss_fn_inputs):
        """Return one loss per CP-local THD token.

        Args:
            output: CP-local model values with shape ``[tokens]``.
            loss_fn_inputs: CP-local side channels with leading ``[tokens]``.

        Returns:
            Per-token losses with shape ``[tokens]``.
        """
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], expected_routes)
        return output

    engine = Engine(
        model,
        device="cpu",
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
        batch_context_fn=batch_context,
    )
    engine._dp_group_and_size = lambda: (None, 1)
    engine._gradient_group_and_size = lambda _group, _size: (None, 1)

    result = engine.forward_backward([datum], loss_fn)

    assert result.loss_sum.item() == pytest.approx(18.0)
    torch.testing.assert_close(expected_routes[-1], torch.tensor([[-1, -1]], dtype=torch.int16))


def test_pipeline_thd_rejects_nonzero_per_token_pad_sentinel_before_schedule():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "position_ids": torch.arange(2).expand(2, -1),
            "seq_lens": torch.tensor([[2], [2]]),
            "seq_lens_padded": torch.tensor([[2], [2]]),
            "qkv_format": "thd",
        },
        loss_fn_inputs={
            "weights": torch.ones(2, 2),
            "routed_experts": torch.zeros(2, 2, 1, 1, dtype=torch.int16),
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    with pytest.raises(NotImplementedError, match="packed pipeline.*nonzero PER_TOKEN pad sentinels"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            collate_fn=collate_prebatched,
        ).forward_backward([datum], _identity_loss)

    assert pipeline.step_calls == 0
    assert model.forward_calls == 0


def test_pipeline_rejects_schedule_gradient_scaling_before_forward():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model, scale_grads=True)

    with pytest.raises(ValueError, match="scale_grads_in_schedule=False"):
        Engine(pipeline, device="cpu", mesh_context=_pipeline_mesh_context()).forward_backward(
            [_datum([1])], _identity_loss
        )

    assert pipeline.step_calls == 0
    assert model.forward_calls == 0
    assert model.weight.grad is None


def test_pipeline_groups_multiple_flat_datums_into_one_outer_batch():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model)

    result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=2,
    ).forward_backward([_datum([1]), _datum([2])], _identity_loss)

    assert result.loss.item() == pytest.approx(1.5)
    assert pipeline.step_calls == 1
    assert model.forward_calls == 2


def test_pipeline_requires_mesh_context_before_forward():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model)

    with pytest.raises(ValueError, match="requires mesh_context"):
        Engine(pipeline, device="cpu").forward_backward([_datum([1])], _identity_loss)

    assert pipeline.step_calls == 0
    assert model.forward_calls == 0
    assert model.weight.grad is None


def test_forward_context_covers_forward_loss_and_backward():
    active = False

    @contextmanager
    def forward_context(_model_inputs):
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    class ContextModel(ScaleModel):
        def forward(self, input_ids, **kwargs):
            assert active
            return super().forward(input_ids, **kwargs)

    model = ContextModel()
    model.weight.register_hook(lambda grad: grad if active else pytest.fail("context ended before backward"))

    def loss_fn(output, _inputs):
        assert active
        return output

    Engine(model, device="cpu", context_fn=forward_context).forward_backward([_datum([1, 2])], loss_fn)
    assert not active


@pytest.mark.parametrize("execution", ["forward", "forward_backward"])
def test_batch_context_receives_prepared_side_inputs_without_forwarding_them_to_model(execution):
    active = False
    legacy_active = False
    events = []
    routed_experts = torch.tensor(
        [[[[0], [2]], [[1], [-1]], [[3], [0]]]],
        dtype=torch.int16,
    )

    @contextmanager
    def legacy_context(model_inputs):
        """Model-only context remains outermost for FP8-style callers.

        Args:
            model_inputs: Prepared model mapping with ``input_ids [B, S]``.

        Yields:
            ``None`` while both the batch context and model execution run.
        """
        nonlocal legacy_active
        assert not legacy_active
        assert not active
        torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[1, 2, 3]]))
        legacy_active = True
        events.append("legacy_enter")
        try:
            yield
        finally:
            assert not active
            events.append("legacy_exit")
            legacy_active = False

    @contextmanager
    def batch_context(model_inputs, loss_fn_inputs):
        """Expose prepared routing tensors only through the batch context.

        Args:
            model_inputs: Prepared model mapping with ``input_ids`` shaped ``[B, S]``.
            loss_fn_inputs: Prepared loss mapping with ``routed_experts`` shaped
                ``[B, S, G, K]``.

        Yields:
            None while the eager model, loss, and optional backward execute.
        """
        nonlocal active
        assert legacy_active
        assert not active
        assert "routed_experts" not in model_inputs
        torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[1, 2, 3]]))
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], routed_experts)
        active = True
        events.append("context_enter")
        try:
            yield
        finally:
            events.append("context_exit")
            active = False

    class ContextModel(ScaleModel):
        def forward(self, input_ids, **kwargs):
            """Scale prepared token ids without receiving loss-only routes.

            Args:
                input_ids: Prepared token ids with shape ``[B, S]``.
                **kwargs: Additional model inputs; ``routed_experts`` must be absent.

            Returns:
                Per-token values with shape ``[B, S]``.
            """
            assert active
            assert "routed_experts" not in kwargs
            events.append("model")
            return super().forward(input_ids, **kwargs)

    model = ContextModel()

    def record_backward(grad):
        """Record backward while preserving the scalar parameter gradient.

        Args:
            grad: Scalar gradient for the toy model weight.

        Returns:
            The unchanged scalar gradient.
        """
        if not active or not legacy_active:
            pytest.fail("legacy or batch context ended before backward")
        events.append("backward")
        return grad

    model.weight.register_hook(record_backward)
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        loss_fn_inputs={
            "weights": torch.ones(1, 3),
            "routed_experts": routed_experts,
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    def loss_fn(output, loss_fn_inputs):
        """Return prepared per-token values as losses after checking routes.

        Args:
            output: Model values with shape ``[B, S]``.
            loss_fn_inputs: Loss mapping with routes shaped ``[B, S, G, K]``.

        Returns:
            Per-token losses with shape ``[B, S]``.
        """
        assert active
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], routed_experts)
        events.append("loss")
        return output

    result = getattr(
        Engine(
            model,
            device="cpu",
            collate_fn=collate_prebatched,
            context_fn=legacy_context,
            batch_context_fn=batch_context,
        ),
        execution,
    )([datum], loss_fn)

    assert result.loss_sum.item() == pytest.approx(6.0)
    expected_events = ["legacy_enter", "context_enter", "model", "loss"]
    if execution == "forward_backward":
        expected_events.append("backward")
    expected_events.append("context_exit")
    expected_events.append("legacy_exit")
    assert events == expected_events
    assert not active
    assert not legacy_active


def test_batch_context_covers_activation_checkpoint_recompute():
    active = False
    routed_experts = torch.tensor([[[[1]], [[-1]], [[2]]]], dtype=torch.int16)

    @contextmanager
    def batch_context(_model_inputs, loss_fn_inputs):
        """Keep replay side inputs installed through checkpoint recomputation.

        Args:
            _model_inputs: Prepared model mapping with token axes ``[B, S]``.
            loss_fn_inputs: Prepared loss mapping with routes shaped ``[B, S, G, K]``.

        Yields:
            None while forward, loss, backward, and recomputation execute.
        """
        nonlocal active
        torch.testing.assert_close(loss_fn_inputs["routed_experts"], routed_experts)
        active = True
        try:
            yield
        finally:
            active = False

    class CheckpointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.block_calls = 0

        def _block(self, value):
            """Apply a recomputed nonlinear transform to one token batch.

            Args:
                value: Floating-point token values with shape ``[B, S]``.

            Returns:
                Nonlinear activations with shape ``[B, S]``.
            """
            assert active
            self.block_calls += 1
            return torch.sin(value * self.weight)

        def forward(self, input_ids, **kwargs):
            """Checkpoint a token transform without forwarding replay side inputs.

            Args:
                input_ids: Prepared token ids with shape ``[B, S]``.
                **kwargs: Additional model inputs; ``routed_experts`` must be absent.

            Returns:
                Checkpointed activations with shape ``[B, S]``.
            """
            assert active
            assert "routed_experts" not in kwargs
            return checkpoint(self._block, input_ids.to(torch.float32), use_reentrant=False)

    model = CheckpointModel()

    def assert_checkpoint_backward_context(grad):
        """Require the batch context while propagating a scalar checkpoint gradient.

        Args:
            grad: Scalar gradient for the checkpointed model weight.

        Returns:
            The unchanged scalar gradient.
        """
        if not active:
            pytest.fail("context ended before backward")
        return grad

    model.weight.register_hook(assert_checkpoint_backward_context)
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
        loss_fn_inputs={
            "weights": torch.ones(1, 3),
            "routed_experts": routed_experts,
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "routed_experts": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    def loss_fn(output, _loss_fn_inputs):
        """Build per-token nonlinear losses that force checkpoint recomputation.

        Args:
            output: Checkpointed activations with shape ``[B, S]``.
            _loss_fn_inputs: Prepared loss mapping with token axes ``[B, S]``.

        Returns:
            Squared per-token losses with shape ``[B, S]``.
        """
        return output.square()

    Engine(
        model,
        device="cpu",
        collate_fn=collate_prebatched,
        batch_context_fn=batch_context,
    ).forward_backward([datum], loss_fn)

    assert model.block_calls == 2
    assert not active


def test_pipeline_batch_context_tracks_each_microbatch_through_checkpoint_backward():
    active_route = []
    block_routes = []
    context_routes = []

    class CheckpointedPipelineModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def block(self, values, weight):
            assert len(active_route) == 1
            block_routes.append(active_route[0])
            return values * weight

        def forward(self, input_ids, **_kwargs):
            expected_route = int((input_ids[0, 0] - 1) // 2)
            assert active_route == [expected_route]
            return checkpoint(self.block, input_ids.float(), self.weight, use_reentrant=False)

    model = CheckpointedPipelineModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])

    @contextmanager
    def batch_context(_model_inputs, loss_fn_inputs):
        route = int(loss_fn_inputs["route_id"].flatten()[0])
        assert not active_route
        active_route.append(route)
        context_routes.append(route)
        try:
            yield
        finally:
            active_route.clear()

    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2], [3, 4]])},
        loss_fn_inputs={
            "weights": torch.ones(2, 2),
            "route_id": torch.tensor([[0, 0], [1, 1]]),
        },
        loss_fn_input_layouts={
            "weights": LossInputLayout.PER_TOKEN,
            "route_id": LossInputLayout.PER_TOKEN,
        },
    )

    def loss_fn(output, loss_fn_inputs):
        route = int(loss_fn_inputs["route_id"].flatten()[0])
        assert active_route == [route]
        return output

    result = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        collate_fn=collate_prebatched,
        batch_context_fn=batch_context,
    ).forward_backward([datum], loss_fn)

    assert result.loss.item() == pytest.approx(2.5)
    assert model.weight.grad.item() == pytest.approx(2.5)
    assert block_routes == [1, 1, 0, 0]
    assert context_routes == [1, 1, 1, 0, 0, 0]
    assert not active_route


def test_window_sets_the_same_moe_aux_scale_as_the_recipes(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)

    Engine(ScaleModel(), device="cpu").forward_backward(
        [_datum([1]), _datum([2])],
        _identity_loss,
    )

    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(0.5)


class TinyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text = nn.Embedding(8, 1)
        self.vision = nn.Linear(1, 1, bias=False)

    def forward(self, input_ids, pixel_values):
        pixels = torch.stack(pixel_values)
        return self.text(input_ids).mean(dim=1).squeeze(-1) + self.vision(pixels).squeeze(-1)


def _vlm_collate(datums):
    return (
        {
            "input_ids": torch.stack([datum.model_inputs["input_ids"] for datum in datums]),
            "pixel_values": [datum.model_inputs["pixel_values"] for datum in datums],
        },
        {"weights": torch.stack([datum.loss_fn_inputs["weights"] for datum in datums])},
    )


def test_model_specific_collater_keeps_multimodal_inputs_and_gradients():
    datums = [
        Datum(
            model_inputs={"input_ids": torch.tensor([1, 2]), "pixel_values": torch.tensor([0.5])},
            loss_fn_inputs={"weights": torch.tensor(1.0)},
        ),
        Datum(
            model_inputs={"input_ids": torch.tensor([3, 4]), "pixel_values": torch.tensor([1.5])},
            loss_fn_inputs={"weights": torch.tensor(1.0)},
        ),
    ]
    model = TinyVLM()
    Engine(model, device="cpu", microbatch_size=2, collate_fn=_vlm_collate).forward_backward(
        datums,
        lambda output, _inputs: output,
    )

    assert model.text.weight.grad is not None
    assert model.text.weight.grad.abs().sum() > 0
    assert model.vision.weight.grad is not None
    assert model.vision.weight.grad.abs().sum() > 0


def test_explicit_microbatch_sizes_preserve_packed_vlm_group_boundaries():
    processor = SimpleNamespace(
        image_token_id=99,
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    datums = []
    for sample_id in (10, 20, 30, 40):
        input_ids = torch.tensor([sample_id, 99, sample_id + 1])
        datums.append(
            Datum(
                model_inputs={
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "pixel_values": torch.tensor([[float(sample_id), float(sample_id) + 0.5]]),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                loss_fn_inputs={
                    "labels": input_ids[1:].clone(),
                    "weights": torch.ones(2),
                    "routed_experts": torch.tensor([sample_id, sample_id + 1], dtype=torch.int16).view(2, 1, 1),
                },
                loss_fn_input_layouts={
                    "labels": LossInputLayout.PER_TOKEN,
                    "weights": LossInputLayout.PER_TOKEN,
                    "routed_experts": LossInputLayout.PER_TOKEN,
                },
                loss_fn_input_pad_values={"labels": -100, "routed_experts": -1},
            )
        )

    collate_groups = []
    collated_routes = []
    prepared_groups = []
    canonical_collate = partial(
        collate_vlm_datums,
        processor=processor,
        packed=True,
        sequence_alignment=4,
    )

    def recording_collate(group):
        """Pack one explicit VLM group of processor-ready ``[S]`` Datums."""
        collate_groups.append(tuple(int(datum.model_inputs["input_ids"][0]) for datum in group))
        model_inputs, loss_inputs = canonical_collate(group)
        collated_routes.append(loss_inputs["routed_experts"].detach().clone())
        return model_inputs, loss_inputs

    @contextmanager
    def record_prepared_group(model_inputs, loss_inputs):
        """Capture final THD ``[T]`` metadata, media, and ``[T, L, K]`` routes."""
        prepared_groups.append(
            {
                "cu_seqlens": loss_inputs["cu_seqlens"].detach().clone(),
                "pixel_values": model_inputs["pixel_values"].detach().clone(),
                "routed_experts": loss_inputs["routed_experts"].detach().clone(),
            }
        )
        yield

    def loss_with_routes(output, loss_inputs):
        """Return ``[T]`` losses and restorable ``[T, L, K]`` routes."""
        return output, LossFnOutputBatch(
            per_token={
                "routes": PerTokenOutput(loss_inputs["routed_experts"], fill_value=-1),
            }
        )

    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    result = Engine(
        model,
        device="cpu",
        microbatch_size=1,
        collate_fn=recording_collate,
        batch_context_fn=record_prepared_group,
    ).forward(datums, loss_with_routes, microbatch_sizes=[1, 3])

    assert collate_groups == [(10,), (20, 30, 40)]
    assert [group["cu_seqlens"].numel() - 1 for group in prepared_groups] == [1, 3]
    assert [group["pixel_values"].shape[0] for group in prepared_groups] == [1, 3]
    assert collated_routes[0].flatten().tolist() == [10, 11, -1, -1]
    assert collated_routes[1].flatten().tolist() == [
        20,
        21,
        -1,
        -1,
        30,
        31,
        -1,
        -1,
        40,
        41,
        -1,
        -1,
    ]
    assert [record["routes"].flatten().tolist() for record in result.loss_fn_outputs] == [
        [10, 11],
        [20, 21],
        [30, 31],
        [40, 41],
    ]


@pytest.mark.parametrize("packed", [False, True])
def test_vlm_datum_collater_restores_prediction_aligned_outputs_without_padding(packed):
    """VLM outputs use each Datum's shifted ``S-1`` token length."""
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = []
    for input_ids, routes in (
        ([1, 2, 3, 4], [10, 11, 12]),
        ([5, 6, 7], [20, 21]),
    ):
        input_ids = torch.tensor(input_ids)
        weights = torch.cat((torch.zeros(1), torch.ones(input_ids.numel() - 1)))
        datums.append(
            Datum(
                model_inputs={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
                loss_fn_inputs={
                    "labels": input_ids.clone(),
                    "weights": weights,
                    "routed_experts": torch.tensor(routes),
                },
                loss_fn_input_layouts={
                    "labels": LossInputLayout.PER_TOKEN,
                    "weights": LossInputLayout.PER_TOKEN,
                    "routed_experts": LossInputLayout.PER_TOKEN,
                },
                loss_fn_input_pad_values={"labels": -100, "routed_experts": -1},
            )
        )

    def loss_with_routes(output, loss_inputs):
        """Return model loss and prediction-aligned route ids.

        Args:
            output: Model values with shape ``[batch, prediction_tokens]``.
            loss_inputs: Collated loss tensors on the same token axes.

        Returns:
            The per-token loss and typed route output batch.
        """
        return output, LossFnOutputBatch(
            per_token={"routes": PerTokenOutput(loss_inputs["routed_experts"], fill_value=-1)}
        )

    model = ScaleModel()
    if packed:
        model.backend = SimpleNamespace(attn="te")
    result = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=partial(
            collate_vlm_datums,
            processor=processor,
            packed=packed,
            sequence_alignment=4 if packed else 1,
        ),
    ).forward(datums, loss_with_routes)

    torch.testing.assert_close(result.loss_fn_outputs[0]["routes"], torch.tensor([10, 11, 12]))
    torch.testing.assert_close(result.loss_fn_outputs[1]["routes"], torch.tensor([20, 21]))


def test_prebatched_datum_keeps_existing_recipe_batch_layout():
    model = ScaleModel()
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2], [3, 4]])},
        loss_fn_inputs={"weights": torch.tensor([[1.0, 1.0], [1.0, 0.0]])},
    )

    result = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward([datum], _identity_loss)

    assert result.loss.item() == pytest.approx(2.0)
    assert model.weight.grad.item() == pytest.approx(2.0)


def test_prebatched_datum_keeps_vlm_media_layout():
    model = TinyVLM()
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "pixel_values": [torch.tensor([0.5]), torch.tensor([1.5])],
        },
        loss_fn_inputs={"weights": torch.ones(2)},
    )

    result = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward([datum], _identity_loss)

    assert torch.isfinite(result.loss)
    assert model.text.weight.grad is not None
    assert model.vision.weight.grad is not None


def test_scalar_loss_is_a_local_weighted_sum_numerator():
    model = ScaleModel()

    result = Engine(model, device="cpu").forward_backward(
        [_datum([1, 100], [1.0, 0.0]), _datum([3, 5], [0.5, 1.0])],
        lambda output, inputs: (output * inputs["weights"]).sum(),
    )

    assert result.loss.item() == pytest.approx(3.0)
    assert model.weight.grad.item() == pytest.approx(3.0)


def test_zero_weights_run_graph_connected_zero_backward():
    model = ScaleModel()
    result = Engine(model, device="cpu").forward_backward(
        [_datum([1, 2], [0.0, 0.0])],
        _identity_loss,
    )
    assert result.loss.item() == 0
    assert result.loss_sum.item() == 0
    assert result.weight_sum.item() == 0
    assert model.forward_calls == 1
    assert model.weight.grad.item() == 0


def test_pipeline_parallelism_fails_before_forward():
    model = ScaleModel()
    mesh_context = SimpleNamespace(pp_size=2, cp_size=1)

    with pytest.raises(NotImplementedError, match="pipeline"):
        Engine(model, device="cpu", mesh_context=mesh_context).forward_backward(
            [_datum([1])],
            _identity_loss,
        )

    assert model.forward_calls == 0


@pytest.mark.parametrize(
    ("declared_mode", "expected_pre_collective_grad"),
    [
        pytest.param(None, 14.0, id="undeclared-averaged"),
        pytest.param(False, 14.0, id="explicit-averaged"),
        pytest.param(True, 3.5, id="summed"),
    ],
)
def test_gradient_reduction_mode_controls_main_loss_scale(
    monkeypatch,
    declared_mode,
    expected_pre_collective_grad,
):
    model = ScaleModel()
    if declared_mode is not None:
        model.calculate_per_token_loss = declared_mode
    engine = Engine(model, device="cpu")
    _configure_fake_gradient_group(engine, monkeypatch, group_size=4)

    result = engine.forward_backward([_datum([2, 4], [1.0, 3.0])], _identity_loss)

    assert result.loss_sum.item() == pytest.approx(14.0)
    assert result.weight_sum.item() == pytest.approx(4.0)
    assert result.loss.item() == pytest.approx(3.5)
    assert model.weight.grad.item() == pytest.approx(expected_pre_collective_grad)


def test_gradient_reduction_mode_finds_summed_backend_below_ordinary_wrapper(monkeypatch):
    model = ScaleModel()
    model.distributed_backend = nn.Identity()
    model.distributed_backend.calculate_per_token_loss = True
    engine = Engine(model, device="cpu")
    _configure_fake_gradient_group(engine, monkeypatch, group_size=8)

    engine.forward_backward([_datum([2, 4], [1.0, 3.0])], _identity_loss)

    assert model.weight.grad.item() == pytest.approx(3.5)


def test_summed_gradient_mode_zero_weight_window_keeps_graph_connected_zero(monkeypatch):
    model = ScaleModel()
    model.calculate_per_token_loss = True
    engine = Engine(model, device="cpu")
    _configure_fake_gradient_group(engine, monkeypatch, group_size=4)

    result = engine.forward_backward([_datum([100, 200], [0.0, 0.0])], _identity_loss)

    assert result.loss.item() == 0
    assert result.loss_sum.item() == 0
    assert result.weight_sum.item() == 0
    assert model.forward_calls == 1
    assert model.weight.grad.item() == 0


@pytest.mark.parametrize(
    ("summed_gradients", "expected_scale", "expected_local_aux_grad"),
    [
        pytest.param(False, 2.0 / 3.0, 2.0, id="averaged"),
        pytest.param(True, 1.0 / 12.0, 0.25, id="summed"),
    ],
)
def test_moe_aux_scale_uses_general_gradient_reduction_formula(
    monkeypatch,
    summed_gradients,
    expected_scale,
    expected_local_aux_grad,
):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    model = _MainAndAuxScaleModel()
    model.calculate_per_token_loss = summed_gradients
    engine = Engine(model, device="cpu")
    engine._cp_size = lambda: 2
    _configure_fake_gradient_group(engine, monkeypatch, group_size=8)

    engine.forward_backward([_datum([1]), _datum([2]), _datum([3])], _identity_loss)

    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(expected_scale)
    assert model.aux_weight.grad.item() == pytest.approx(expected_local_aux_grad)


@pytest.mark.parametrize(
    ("root_mode", "nested_mode", "error_type", "match"),
    [
        (True, False, ValueError, "mixes calculate_per_token_loss"),
        ("yes", None, TypeError, "must be boolean"),
    ],
)
def test_gradient_reduction_mode_rejects_ambiguous_declarations(root_mode, nested_mode, error_type, match):
    model = ScaleModel()
    model.calculate_per_token_loss = root_mode
    if nested_mode is not None:
        model.mode_probe = nn.Identity()
        model.mode_probe.calculate_per_token_loss = nested_mode

    with pytest.raises(error_type, match=match):
        Engine(model, device="cpu")


def test_loss_shape_must_exactly_match_weights():
    model = ScaleModel()
    with pytest.raises(ValueError, match="exactly the same shape"):
        Engine(model, device="cpu").forward_backward(
            [_datum([1, 2])],
            lambda output, _inputs: output[:, :1],
        )
    assert model.weight.grad is None


def test_collater_cannot_change_weight_sum():
    model = ScaleModel()

    def bad_collate(datums):
        model_inputs, loss_inputs = collate_datums(datums)
        loss_inputs["weights"].zero_()
        return model_inputs, loss_inputs

    with pytest.raises(ValueError, match="collate_fn changed"):
        Engine(model, device="cpu", collate_fn=bad_collate).forward_backward(
            [_datum([1, 2])],
            _identity_loss,
        )
    assert model.forward_calls == 0


def _distributed_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        # With an unwrapped model, forward-only execution adds no DP
        # collectives and keeps each rank's statistics local. DDP/FSDP wrappers
        # may impose their own aligned-call requirement.
        forward_window = [_datum([1])] if rank == 0 else [_datum([2]), _datum([3])]
        forward_result = Engine(ScaleModel(), device="cpu").forward(forward_window, _identity_loss)
        assert forward_result.loss_sum.item() == pytest.approx(1.0 if rank == 0 else 5.0)
        assert forward_result.weight_sum.item() == pytest.approx(1.0 if rank == 0 else 2.0)

        model = nn.parallel.DistributedDataParallel(ScaleModel())
        bad_window = [_datum([1])] if rank == 0 else [_datum([1]), _datum([2])]
        with pytest.raises(ValueError, match="same number of microbatches"):
            Engine(model, device="cpu").forward_backward(bad_window, _identity_loss)
        assert model.module.forward_calls == 0

        window = [_datum([1, 2]), _datum([3])] if rank == 0 else [_datum([4]), _datum([5, 6])]
        result = Engine(model, device="cpu").forward_backward(window, _identity_loss)
        assert result.loss.item() == pytest.approx(3.5)
        assert result.loss_sum.item() == pytest.approx(21.0)
        assert result.weight_sum.item() == pytest.approx(6.0)
        assert result.loss_fn_outputs == []
        assert model.module.weight.grad.item() == pytest.approx(3.5)

        # Dynamic token balancing keeps the number of forwards rank-symmetric,
        # but the number of Datums packed into each forward may differ by DP
        # rank. Those local partitions must still produce one global mean.
        variable_model = nn.parallel.DistributedDataParallel(ScaleModel())
        first_value = rank * 4 + 1
        variable_window = [_datum([value]) for value in range(first_value, first_value + 4)]
        variable_sizes = [1, 3] if rank == 0 else [2, 2]
        variable_result = Engine(variable_model, device="cpu", microbatch_size=1).forward_backward(
            variable_window,
            _identity_loss,
            microbatch_sizes=variable_sizes,
        )

        assert variable_model.module.forward_calls == 2
        assert variable_result.loss.item() == pytest.approx(4.5)
        assert variable_result.loss_sum.item() == pytest.approx(36.0)
        assert variable_result.weight_sum.item() == pytest.approx(8.0)
        assert variable_model.module.weight.grad.item() == pytest.approx(4.5)
    finally:
        dist.destroy_process_group()


def _hybridep_packed_equalization_worker(rank: int, world_size: int, init_file: str) -> None:
    """Compare unequal packed EP ranks after one physical-width equalization.

    Args:
        rank: Current Gloo process rank.
        world_size: Number of ranks in the shared DP/EP topology.
        init_file: Shared file-store path used to initialize the process group.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        mesh_context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(ep_size=world_size),
            world_size=world_size,
        )
        model = ScaleModel()
        model.backend = SimpleNamespace(attn="te")
        model.hybridep_probe = _UniformTokenDispatcherProbe()
        model = nn.parallel.DistributedDataParallel(model)
        model.backend = SimpleNamespace(attn="te")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        token_ids = torch.tensor([1, 2] if rank == 0 else [3, 4, 5, 6])
        routes = torch.arange(token_ids.numel(), dtype=torch.int16).view(1, -1, 1, 1)
        datum = Datum(
            model_inputs={
                "input_ids": token_ids.unsqueeze(0),
                "attention_mask": torch.ones(1, token_ids.numel(), dtype=torch.long),
                "position_ids": torch.arange(token_ids.numel()).unsqueeze(0),
                "seq_lens": torch.tensor([[token_ids.numel()]], dtype=torch.int32),
                "seq_lens_padded": torch.tensor([[token_ids.numel()]], dtype=torch.int32),
                "qkv_format": "thd",
            },
            loss_fn_inputs={
                "weights": torch.ones(1, token_ids.numel()),
                "labels": token_ids.unsqueeze(0).clone(),
                "routed_experts": routes,
            },
            loss_fn_input_layouts={
                "weights": LossInputLayout.PER_TOKEN,
                "labels": LossInputLayout.PER_TOKEN,
                "routed_experts": LossInputLayout.PER_TOKEN,
            },
            loss_fn_input_pad_values={"labels": -100, "routed_experts": -1},
        )

        @contextmanager
        def batch_context(model_inputs, loss_inputs):
            """Assert final THD tensors use the EP-wide four-token extent.

            Args:
                model_inputs: Final THD model tensors with token shape ``[4]``.
                loss_inputs: Final THD loss tensors with token-leading shape
                    ``[4, ...]``.

            Yields:
                None while forward, loss, and backward consume the batch.
            """
            assert model_inputs["input_ids"].shape == (4,)
            assert model_inputs["padding_mask"].shape == (4,)
            assert loss_inputs["routed_experts"].shape == (4, 1, 1)
            if rank == 0:
                torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([1, 2, 0, 0]))
                torch.testing.assert_close(model_inputs["padding_mask"], torch.tensor([False, False, True, True]))
                torch.testing.assert_close(loss_inputs["weights"], torch.tensor([1.0, 1.0, 0.0, 0.0]))
                torch.testing.assert_close(
                    loss_inputs["routed_experts"],
                    torch.tensor([[[0]], [[1]], [[-1]], [[-1]]], dtype=torch.int16),
                )
            else:
                assert not bool(model_inputs["padding_mask"].any())
                torch.testing.assert_close(loss_inputs["weights"], torch.ones(4))
            yield

        def loss_with_tokens(output, _loss_inputs):
            """Return the scaled tokens as both loss and a restorable stream.

            Args:
                output: Final THD model output with shape ``[4]``.
                _loss_inputs: Final THD loss mapping on the same token axis.

            Returns:
                Per-token losses and a typed token output batch.
            """
            return output, LossFnOutputBatch(per_token={"tokens": PerTokenOutput(output)})

        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
            optimizers=optimizer,
            max_grad_norm=None,
            batch_context_fn=batch_context,
        )
        result = engine.forward_backward([datum], loss_with_tokens)

        assert result.loss.item() == pytest.approx(3.5)
        assert result.loss_sum.item() == pytest.approx(21.0)
        assert result.weight_sum.item() == pytest.approx(6.0)
        assert model.module.weight.grad.item() == pytest.approx(3.5)
        torch.testing.assert_close(result.loss_fn_outputs[0]["tokens"], token_ids.to(torch.float32).unsqueeze(0))

        engine.optim_step()
        assert model.module.weight.item() == pytest.approx(0.65)
    finally:
        dist.destroy_process_group()


def _hybridep_padded_batch_mismatch_worker(rank: int, world_size: int, init_file: str) -> None:
    """Verify every EP rank rejects unequal padded batch sizes.

    Args:
        rank: Current Gloo process rank.
        world_size: Number of ranks in the shared DP/EP topology.
        init_file: Shared file-store path used to initialize the process group.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        mesh_context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(ep_size=world_size),
            world_size=world_size,
        )
        model = ScaleModel()
        model.hybridep_probe = _UniformTokenDispatcherProbe()
        datums = [_datum([1, 2, 3])] if rank == 0 else [_datum([4, 5]), _datum([6])]
        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            microbatch_size=len(datums),
        )
        with pytest.raises(NotImplementedError, match="same batch size"):
            engine.forward(datums, _identity_loss)

        width = 3 if rank == 0 else 2
        attention_mask = torch.ones((1, width), dtype=torch.long)
        if rank == 1:
            attention_mask = torch.ones((1, 1, width, width), dtype=torch.bool)
        prebatched = Datum(
            model_inputs={"input_ids": torch.arange(width).view(1, width), "attention_mask": attention_mask},
            loss_fn_inputs={"weights": torch.ones((1, width))},
            loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
        )
        engine = Engine(model, device="cpu", mesh_context=mesh_context, collate_fn=collate_prebatched)
        with pytest.raises(NotImplementedError, match="sequence-padding preflight"):
            engine.forward([prebatched], _identity_loss)

        equal_width = 2
        equal_datum = Datum(
            model_inputs={
                "input_ids": torch.tensor([[rank + 1, rank + 2]]),
                "attention_mask": torch.ones((1, 1, equal_width, equal_width), dtype=torch.bool),
            },
            loss_fn_inputs={"weights": torch.ones((1, equal_width))},
            loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
        )
        result = engine.forward([equal_datum], _identity_loss)
        assert result.loss_sum.item() == pytest.approx(2 * rank + 3)
        assert "padding_mask" not in equal_datum.model_inputs
    finally:
        dist.destroy_process_group()


def _hybridep_packed_cp_cross_axis_worker(rank: int, world_size: int, init_file: str) -> None:
    """Verify unequal DP packs reach one width through real EP then EP-shard groups.

    Args:
        rank: Current Gloo process rank.
        world_size: Four-rank ``dp=2, cp=2, ep=2`` topology size.
        init_file: Shared file-store path used to initialize the process group.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh_context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(dp_size=2, cp_size=2, ep_size=2),
            world_size=world_size,
        )
        assert mesh_context.moe_mesh["ep"].size() == 2
        assert mesh_context.moe_mesh["ep_shard"].size() == 2

        class MockTex:
            @staticmethod
            def thd_get_partitioned_indices(
                _cu_seqlens: torch.Tensor,
                total_tokens: int,
                cp_size: int,
                cp_rank: int,
            ) -> torch.Tensor:
                """Return TE's head-tail partition for one packed stream.

                Args:
                    _cu_seqlens: Tensor of shape ``[documents + 1]`` containing
                        packed cumulative padded sequence lengths.
                    total_tokens: Physical global token count.
                    cp_size: Context-parallel degree.
                    cp_rank: Rank within the context-parallel group.

                Returns:
                    Tensor of shape ``[local_tokens]`` containing global token
                    indices in TE head-tail order.
                """
                assert cp_size == 2
                assert total_tokens == 8
                chunk = total_tokens // (2 * cp_size)
                head = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk)
                tail_start = (2 * cp_size - cp_rank - 1) * chunk
                tail = torch.arange(tail_start, tail_start + chunk)
                return torch.cat((head, tail))

        sys.modules["transformer_engine_torch"] = MockTex

        model = ScaleModel()
        model.backend = SimpleNamespace(attn="te")
        model.hybridep_probe = _UniformTokenDispatcherProbe()
        model = nn.parallel.DistributedDataParallel(model)
        model.backend = SimpleNamespace(attn="te")
        dp_rank = get_flat_mesh(mesh_context.device_mesh, "dp").get_local_rank()
        cp_rank = mesh_context.device_mesh["cp"].get_local_rank()
        token_count = 4 if dp_rank == 0 else 8
        token_ids = torch.arange(1, token_count + 1)
        routes = torch.arange(token_count, dtype=torch.int16).view(1, token_count, 1, 1)
        datum = Datum(
            model_inputs={
                "input_ids": token_ids.unsqueeze(0),
                "attention_mask": torch.ones(1, token_count, dtype=torch.long),
                "position_ids": torch.arange(token_count).unsqueeze(0),
                "seq_lens": torch.tensor([[token_count]], dtype=torch.int32),
                "seq_lens_padded": torch.tensor([[token_count]], dtype=torch.int32),
                "qkv_format": "thd",
            },
            loss_fn_inputs={
                "weights": torch.ones(1, token_count),
                "labels": token_ids.unsqueeze(0).clone(),
                "routed_experts": routes,
            },
            loss_fn_input_layouts={
                "weights": LossInputLayout.PER_TOKEN,
                "labels": LossInputLayout.PER_TOKEN,
                "routed_experts": LossInputLayout.PER_TOKEN,
            },
            loss_fn_input_pad_values={"labels": -100, "routed_experts": -1},
        )

        @contextmanager
        def batch_context(model_inputs, loss_inputs):
            """Validate the MoE-stage target after the real TE-THD CP partition.

            Args:
                model_inputs: Final THD model mapping. ``input_ids``,
                    ``position_ids``, and ``padding_mask`` are Tensors of shape
                    ``[local_tokens]``; ``cu_seqlens`` is a Tensor of shape
                    ``[documents + 1]``; and ``max_seqlen`` is a scalar Tensor.
                loss_inputs: Final loss mapping. ``weights`` and ``labels`` are
                    Tensors of shape ``[local_tokens]``; ``routed_experts`` is a
                    Tensor of shape ``[local_tokens, top_k, route_fields]``,
                    where both ``top_k`` and ``route_fields`` are one here.

            Yields:
                None while forward and backward consume the prepared shard.
            """
            assert model_inputs["input_ids"].shape == (4,)
            assert model_inputs["padding_mask"].shape == (4,)
            assert loss_inputs["weights"].shape == (4,)
            assert loss_inputs["routed_experts"].shape == (4, 1, 1)
            if dp_rank == 0:
                expected_ids = torch.tensor([1, 2, 0, 0]) if cp_rank == 0 else torch.tensor([3, 4, 0, 0])
                torch.testing.assert_close(model_inputs["input_ids"], expected_ids)
                torch.testing.assert_close(loss_inputs["weights"], torch.tensor([1.0, 1.0, 0.0, 0.0]))
                torch.testing.assert_close(
                    loss_inputs["routed_experts"],
                    torch.tensor([[[0]], [[1]], [[-1]], [[-1]]], dtype=torch.int16)
                    if cp_rank == 0
                    else torch.tensor([[[2]], [[3]], [[-1]], [[-1]]], dtype=torch.int16),
                )
            else:
                assert not bool(model_inputs["padding_mask"].any())
                torch.testing.assert_close(loss_inputs["weights"], torch.ones(4))
            yield

        def loss_fn(output, _loss_inputs):
            """Return local ``[4]`` losses and a restorable THD token stream.

            Args:
                output: Tensor of shape ``[local_tokens]`` containing CP-local
                    model output.
                _loss_inputs: CP-local loss mapping. ``weights`` and ``labels``
                    are Tensors of shape ``[local_tokens]``; ``routed_experts``
                    is a Tensor of shape ``[local_tokens, top_k, route_fields]``.

            Returns:
                A tuple containing a Tensor of shape ``[local_tokens]`` with
                per-token losses and a typed output batch whose ``tokens``
                field contains a Tensor of shape ``[local_tokens]``.
            """
            return output, LossFnOutputBatch(per_token={"tokens": PerTokenOutput(output)})

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
            optimizers=optimizer,
            max_grad_norm=None,
            batch_context_fn=batch_context,
        )
        assert len(engine._hybridep_equalization_groups) == 2
        result = engine.forward_backward([datum], loss_fn)

        expected_mean = 46.0 / 12.0
        assert result.loss.item() == pytest.approx(expected_mean)
        assert result.loss_sum.item() == pytest.approx(46.0)
        assert result.weight_sum.item() == pytest.approx(12.0)
        assert model.module.weight.grad.item() == pytest.approx(expected_mean)
        torch.testing.assert_close(result.loss_fn_outputs[0]["tokens"], token_ids.to(torch.float32).unsqueeze(0))

        engine.optim_step()
        assert model.module.weight.item() == pytest.approx(1.0 - 0.1 * expected_mean)
    finally:
        dist.destroy_process_group()


def _hybridep_padded_cp_cross_axis_worker(rank: int, world_size: int, init_file: str) -> None:
    """Restore same-B, unequal-width DP batches after HybridEP plus CP.

    Args:
        rank: Current Gloo process rank.
        world_size: Four-rank ``dp=2, cp=2, ep=2`` topology size.
        init_file: Shared file-store path used to initialize the process group.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh_context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(dp_size=2, cp_size=2, ep_size=2),
            world_size=world_size,
        )
        base_model = _DistributedCPModel()
        base_model.hybridep_probe = _UniformTokenDispatcherProbe()
        gradient_group = get_flat_mesh(mesh_context.device_mesh, "dp_cp").get_group()
        model = _DDPWithCP(base_model, process_group=gradient_group)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dp_rank = get_flat_mesh(mesh_context.device_mesh, "dp").get_local_rank()
        cp_rank = mesh_context.device_mesh["cp"].get_local_rank()
        values = [[1, 2, 3], [4]] if dp_rank == 0 else [[5, 6], [7, 8]]
        datums = [_datum(tokens) for tokens in values]

        @contextmanager
        def batch_context(model_inputs, loss_inputs):
            """Check the real CP-local physical token rectangle.

            Args:
                model_inputs: CP-local model mapping with ``input_ids`` and
                    ``padding_mask`` shaped ``[physical_batch=2,
                    local_sequence=2]``.
                loss_inputs: CP-local loss mapping with ``weights`` shaped
                    ``[physical_batch=2, local_sequence=2]``.

            Yields:
                None while the CP-local forward consumes this shard.
            """
            assert model_inputs["input_ids"].shape == (2, 2)
            assert model_inputs["padding_mask"].shape == (2, 2)
            assert loss_inputs["weights"].shape == (2, 2)
            if dp_rank == 0:
                expected = torch.tensor([[1, 2], [4, 0]]) if cp_rank == 0 else torch.tensor([[3, 0], [0, 0]])
            else:
                expected = torch.tensor([[5, 6], [7, 8]]) if cp_rank == 0 else torch.zeros((2, 2), dtype=torch.long)
            torch.testing.assert_close(model_inputs["input_ids"], expected)
            yield

        def loss_with_tokens(output, _loss_inputs):
            """Return one typed token stream from the real CP-local shard.

            Args:
                output: CP-local values of shape ``[physical_batch=2,
                    local_sequence=2]``.
                _loss_inputs: CP-local loss mapping with matching ``weights``.

            Returns:
                Per-token losses and a typed token output with the same local
                physical axes.
            """
            return output, LossFnOutputBatch(per_token={"tokens": PerTokenOutput(output)})

        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            microbatch_size=len(datums),
            optimizers=optimizer,
            max_grad_norm=None,
            batch_context_fn=batch_context,
        )
        assert len(engine._hybridep_equalization_groups) == 2
        result = engine.forward_backward(datums, loss_with_tokens)

        assert result.loss.item() == pytest.approx(4.5)
        assert result.loss_sum.item() == pytest.approx(36.0)
        assert result.weight_sum.item() == pytest.approx(8.0)
        assert model.module.weight.grad.item() == pytest.approx(4.5)
        assert len(result.loss_fn_outputs) == len(datums)
        for record, tokens in zip(result.loss_fn_outputs, values):
            torch.testing.assert_close(record["tokens"], torch.tensor(tokens, dtype=torch.float32))

        engine.optim_step()
        assert model.module.weight.item() == pytest.approx(0.55)
    finally:
        dist.destroy_process_group()


def _batch_context_cp_worker(rank: int, world_size: int, init_file: str) -> None:
    """Validate prepared replay side inputs on a real two-rank Gloo CP mesh.

    Args:
        rank: Current Gloo process rank.
        world_size: Number of context-parallel ranks.
        init_file: Shared file-store path used to initialize the process group.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        mesh_context = MeshContext.build(
            MegatronFSDPConfig(),
            ParallelismSizes(dp_size=1, cp_size=world_size),
            world_size=world_size,
        )
        batch_context_active = False

        class ContextCPModel(_DistributedCPModel):
            def forward(self, input_ids, **kwargs):
                """Scale one real CP token shard while the batch context is active.

                Args:
                    input_ids: CP-local token ids with shape ``[B, S_local]``.
                    **kwargs: Additional model inputs; replay routes must be absent.

                Returns:
                    Per-token values with shape ``[B, S_local]``.
                """
                assert batch_context_active
                assert "routed_experts" not in kwargs
                return super().forward(input_ids, **kwargs)

        model = _DDPWithCP(ContextCPModel())
        token_ids = torch.arange(6, dtype=torch.int16)
        routed_experts = torch.stack((token_ids, token_ids + 10), dim=-1).unsqueeze(0).unsqueeze(-1)
        if rank == 0:
            expected_input_ids = torch.tensor([[1, 2, 3, 4]])
            expected_routes = routed_experts[:, :4]
        else:
            expected_input_ids = torch.tensor([[5, 6, 0, 0]])
            expected_routes = torch.cat(
                (
                    routed_experts[:, 4:],
                    torch.full((1, 2, 2, 1), -1, dtype=routed_experts.dtype),
                ),
                dim=1,
            )

        @contextmanager
        def batch_context(model_inputs, loss_fn_inputs):
            """Check rank-local tokens and routes around forward and backward.

            Args:
                model_inputs: CP-local model mapping with ``input_ids`` shaped
                    ``[B, S_local]``.
                loss_fn_inputs: CP-local loss mapping with routes shaped
                    ``[B, S_local, G, K]``.

            Yields:
                None while the DDP-wrapped CP model and loss execute.
            """
            nonlocal batch_context_active
            assert not batch_context_active
            assert "routed_experts" not in model_inputs
            torch.testing.assert_close(model_inputs["input_ids"], expected_input_ids)
            torch.testing.assert_close(loss_fn_inputs["routed_experts"], expected_routes)
            batch_context_active = True
            try:
                yield
            finally:
                batch_context_active = False

        datum = Datum(
            model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])},
            loss_fn_inputs={
                "weights": torch.ones(1, 6),
                "routed_experts": routed_experts,
            },
            loss_fn_input_layouts={
                "weights": LossInputLayout.PER_TOKEN,
                "routed_experts": LossInputLayout.PER_TOKEN,
            },
            loss_fn_input_pad_values={"routed_experts": -1},
        )

        def assert_distributed_backward_context(grad):
            """Require the batch context while propagating a scalar DDP gradient.

            Args:
                grad: Scalar gradient for the DDP-wrapped toy model weight.

            Returns:
                The unchanged scalar gradient.
            """
            if not batch_context_active:
                pytest.fail("batch context ended before backward")
            return grad

        model.module.weight.register_hook(assert_distributed_backward_context)

        def loss_fn(output, loss_fn_inputs):
            """Return CP-local token losses after checking the route shard.

            Args:
                output: CP-local model values with shape ``[B, S_local]``.
                loss_fn_inputs: CP-local loss mapping with routes shaped
                    ``[B, S_local, G, K]``.

            Returns:
                Per-token losses with shape ``[B, S_local]``.
            """
            assert batch_context_active
            torch.testing.assert_close(loss_fn_inputs["routed_experts"], expected_routes)
            return output

        result = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
            batch_context_fn=batch_context,
        ).forward_backward([datum], loss_fn)

        assert result.loss.item() == pytest.approx(3.5)
        assert result.loss_sum.item() == pytest.approx(21.0)
        assert result.weight_sum.item() == pytest.approx(6.0)
        assert model.module.weight.grad.item() == pytest.approx(3.5)
        assert not batch_context_active
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _context_parallel_worker(rank: int, world_size: int, init_file: str, dp_size: int) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        cp_size = world_size // dp_size
        mesh_context = MeshContext.build(
            MegatronFSDPConfig(),
            ParallelismSizes(dp_size=dp_size, cp_size=cp_size),
            world_size=world_size,
        )
        model = _DDPWithCP(_DistributedCPModel())
        if dp_size == 1:
            window = [
                Datum(
                    model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
                    loss_fn_inputs={"weights": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
                ),
                Datum(
                    model_inputs={"input_ids": torch.tensor([[5, 6, 7, 8]])},
                    loss_fn_inputs={"weights": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
                ),
            ]
        else:
            dp_rank = get_flat_mesh(mesh_context.device_mesh, "dp").get_local_rank()
            first = dp_rank * 4 + 1
            window = [
                Datum(
                    model_inputs={"input_ids": torch.arange(first, first + 4).unsqueeze(0)},
                    loss_fn_inputs={"weights": torch.ones(1, 4)},
                )
            ]

        result = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
        ).forward_backward(window, _identity_loss)

        assert result.loss.item() == pytest.approx(4.5)
        assert result.loss_sum.item() == pytest.approx(18.0 if dp_size == 1 else 36.0)
        assert result.weight_sum.item() == pytest.approx(4.0 if dp_size == 1 else 8.0)
        assert model.module.weight.grad.item() == pytest.approx(4.5)

        model.module.weight.grad = None
        forward_result = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
        ).forward(window, _identity_loss)
        expected_sum = (
            18.0 if dp_size == 1 else 10.0 + 16.0 * get_flat_mesh(mesh_context.device_mesh, "dp").get_local_rank()
        )
        assert forward_result.loss_sum.item() == pytest.approx(expected_sum)
        assert forward_result.weight_sum.item() == pytest.approx(4.0)
        assert model.module.weight.grad is None

        # One complete window must use its DP-only denominator while letting CP
        # ranks contribute disjoint numerators. DP ranks deliberately own
        # different amounts of supervision when dp_size=2.
        if dp_size == 1:
            window_a = [
                Datum(
                    model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
                    loss_fn_inputs={"weights": torch.tensor([[1.0, 0.0, 0.0, 0.0]])},
                )
            ]
            window_b = [
                Datum(
                    model_inputs={"input_ids": torch.tensor([[5, 6, 7, 8]])},
                    loss_fn_inputs={"weights": torch.tensor([[0.0, 1.0, 1.0, 1.0]])},
                )
            ]
            expected_a_sum, expected_a_weight = 1.0, 1.0
            expected_b_sum, expected_b_weight = 21.0, 3.0
        else:
            dp_rank = get_flat_mesh(mesh_context.device_mesh, "dp").get_local_rank()
            if dp_rank == 0:
                window_a = [
                    Datum(
                        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
                        loss_fn_inputs={"weights": torch.tensor([[1.0, 0.0, 0.0, 0.0]])},
                    )
                ]
                window_b = [
                    Datum(
                        model_inputs={"input_ids": torch.tensor([[5, 6, 7, 8]])},
                        loss_fn_inputs={"weights": torch.tensor([[0.0, 1.0, 1.0, 1.0]])},
                    )
                ]
            else:
                window_a = [
                    Datum(
                        model_inputs={"input_ids": torch.tensor([[9, 10, 11, 12]])},
                        loss_fn_inputs={"weights": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
                    )
                ]
                window_b = [
                    Datum(
                        model_inputs={"input_ids": torch.tensor([[13, 14, 15, 16]])},
                        loss_fn_inputs={"weights": torch.tensor([[0.0, 0.0, 0.0, 1.0]])},
                    )
                ]
            expected_a_sum, expected_a_weight = 20.0, 3.0
            expected_b_sum, expected_b_weight = 37.0, 4.0

        averaged_model = _DDPWithCP(_DistributedCPModel())
        averaged_optimizer = torch.optim.SGD(averaged_model.parameters(), lr=0.1)
        averaged_engine = Engine(
            averaged_model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
            optimizers=averaged_optimizer,
            max_grad_norm=None,
        )
        averaged_result = averaged_engine.forward_backward(window_a + window_b, _identity_loss)

        assert averaged_result.loss_sum.item() == pytest.approx(expected_a_sum + expected_b_sum)
        assert averaged_result.weight_sum.item() == pytest.approx(expected_a_weight + expected_b_weight)
        expected_global_mean = (expected_a_sum + expected_b_sum) / (expected_a_weight + expected_b_weight)
        assert averaged_result.loss.item() == pytest.approx(expected_global_mean)
        assert averaged_model.module.weight.grad.item() == pytest.approx(expected_global_mean)

        averaged_engine.optim_step()
        assert averaged_model.module.weight.item() == pytest.approx(1.0 - 0.1 * expected_global_mean)

        # Mimic MegatronFSDP calculate_per_token_loss=True with a SUM hook over
        # the same DP-CP gradient group. Engine must not compensate for an
        # average a second time, and the DP-only denominator must still produce
        # the identical normalized update under both CP-only and DP+CP meshes.
        summed_model = _DistributedCPModel()
        summed_model.calculate_per_token_loss = True
        gradient_group = get_flat_mesh(mesh_context.device_mesh, "dp_cp").get_group()

        def sum_gradient(gradient):
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=gradient_group)
            return gradient

        summed_model.weight.register_hook(sum_gradient)
        summed_optimizer = torch.optim.SGD(summed_model.parameters(), lr=0.1)
        summed_engine = Engine(
            summed_model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
            optimizers=summed_optimizer,
            max_grad_norm=None,
        )
        summed_result = summed_engine.forward_backward(window_a + window_b, _identity_loss)

        torch.testing.assert_close(summed_result.loss_sum, averaged_result.loss_sum)
        torch.testing.assert_close(summed_result.weight_sum, averaged_result.weight_sum)
        assert summed_model.weight.grad.item() == pytest.approx(expected_global_mean)

        summed_engine.optim_step()
        torch.testing.assert_close(summed_model.weight, averaged_model.module.weight)
    finally:
        dist.destroy_process_group()


def _mismatched_context_parallel_weights_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh_context = MeshContext.build(
            MegatronFSDPConfig(),
            ParallelismSizes(dp_size=1, cp_size=world_size),
            world_size=world_size,
        )
        model = _DDPWithCP(_DistributedCPModel())
        datum = Datum(
            model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
            loss_fn_inputs={"weights": torch.full((1, 4), float(rank + 1))},
        )

        with pytest.raises(ValueError, match="identical full-sequence weights"):
            Engine(
                model,
                device="cpu",
                mesh_context=mesh_context,
                collate_fn=collate_prebatched,
            ).forward_backward([datum], _identity_loss)

        assert model.module.forward_calls == 0
    finally:
        dist.destroy_process_group()


def _context_parallel_output_consensus_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        engine = Engine(
            ScaleModel(),
            device="cpu",
            mesh_context=SimpleNamespace(
                pp_size=1,
                cp_size=world_size,
                device_mesh=_CPMesh(size=world_size, rank=rank),
                process_group=None,
            ),
        )
        weights = torch.ones(2)
        valid = LossFnOutputBatch(
            per_token={"probe": PerTokenOutput(torch.tensor([1.0, 2.0]))},
            per_datum=[{}],
        )
        cases = [
            (None if rank == 0 else valid, "different loss output schemas"),
            (
                valid
                if rank == 0
                else LossFnOutputBatch(
                    per_token={"probe": PerTokenOutput(torch.tensor([1.0]))},
                    per_datum=[{}],
                ),
                "invalid loss_fn outputs",
            ),
            (
                valid
                if rank == 0
                else LossFnOutputBatch(
                    per_token={"probe": PerTokenOutput(torch.tensor([1.0, 2.0]))},
                    per_datum=[{}, {}],
                ),
                "invalid loss_fn outputs",
            ),
        ]
        for outputs, match in cases:
            with pytest.raises(ValueError, match=match):
                engine._validate_loss_fn_outputs_across_cp(
                    outputs,
                    weights,
                    expected_records=1,
                )
            dist.barrier()

        with pytest.raises(ValueError, match="invalid loss_fn outputs"):
            engine._validate_loss_fn_outputs_across_cp(
                valid,
                weights,
                expected_records=1,
                local_error=ValueError("rank-local output parse failure") if rank == 0 else None,
            )
    finally:
        dist.destroy_process_group()


def _data_parallel_output_error_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        model = nn.parallel.DistributedDataParallel(ScaleModel())

        def loss_with_rank_local_output_error(output, _loss_inputs):
            records = [{}, {}] if rank == 0 else [{}]
            return output, LossFnOutputBatch(
                per_token={"probe": PerTokenOutput(output)},
                per_datum=records,
            )

        expected = "per_datum contains 2 records" if rank == 0 else "another model-parallel rank"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            Engine(model, device="cpu").forward_backward(
                [_datum([1, 2])],
                loss_with_rank_local_output_error,
            )
        assert model.module.weight.grad is not None
    finally:
        dist.destroy_process_group()


def test_data_parallel_window_uses_global_numerator_and_denominator(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(2, str(tmp_path / "engine_dist_init")),
        nprocs=2,
        join=True,
    )


def test_hybridep_packed_ranks_equalize_physical_token_extents(tmp_path):
    mp.spawn(
        _hybridep_packed_equalization_worker,
        args=(2, str(tmp_path / "engine_hybridep_packed_init")),
        nprocs=2,
        join=True,
    )


def test_hybridep_padded_ranks_reject_unequal_batch_sizes(tmp_path):
    mp.spawn(
        _hybridep_padded_batch_mismatch_worker,
        args=(2, str(tmp_path / "engine_hybridep_padded_batch_mismatch_init")),
        nprocs=2,
        join=True,
    )


def test_hybridep_packed_cp_equalizes_across_ep_and_ep_shard_axes(tmp_path):
    mp.spawn(
        _hybridep_packed_cp_cross_axis_worker,
        args=(4, str(tmp_path / "engine_hybridep_packed_cp_cross_axis_init")),
        nprocs=4,
        join=True,
    )


def test_hybridep_padded_cp_equalizes_across_ep_and_ep_shard_axes(tmp_path):
    mp.spawn(
        _hybridep_padded_cp_cross_axis_worker,
        args=(4, str(tmp_path / "engine_hybridep_padded_cp_cross_axis_init")),
        nprocs=4,
        join=True,
    )


def test_batch_context_receives_real_context_parallel_route_shards(tmp_path):
    mp.spawn(
        _batch_context_cp_worker,
        args=(2, str(tmp_path / "engine_batch_context_cp_init")),
        nprocs=2,
        join=True,
    )


def test_context_parallel_window_uses_dp_denominator_and_dp_cp_gradient_sum(tmp_path):
    mp.spawn(
        _context_parallel_worker,
        args=(2, str(tmp_path / "engine_cp_init"), 1),
        nprocs=2,
        join=True,
    )


def test_data_and_context_parallel_composition_matches_global_reference(tmp_path):
    mp.spawn(
        _context_parallel_worker,
        args=(4, str(tmp_path / "engine_dp_cp_init"), 2),
        nprocs=4,
        join=True,
    )


def test_context_parallel_replicas_require_the_same_full_sequence_weights(tmp_path):
    mp.spawn(
        _mismatched_context_parallel_weights_worker,
        args=(2, str(tmp_path / "engine_cp_mismatch_init")),
        nprocs=2,
        join=True,
    )


def test_context_parallel_output_consensus_rejects_rank_local_contracts_without_hanging(tmp_path):
    mp.spawn(
        _context_parallel_output_consensus_worker,
        args=(2, str(tmp_path / "engine_cp_output_consensus_init")),
        nprocs=2,
        join=True,
    )


def test_data_parallel_output_errors_propagate_after_backward_without_hanging(tmp_path):
    mp.spawn(
        _data_parallel_output_error_worker,
        args=(2, str(tmp_path / "engine_dp_output_error_init")),
        nprocs=2,
        join=True,
    )
