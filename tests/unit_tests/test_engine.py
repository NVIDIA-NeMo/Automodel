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
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

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
)
from nemo_automodel.components.datasets.vlm.pp_media import VLM_PP_MEDIA_KEY, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.config import MegatronFSDPConfig
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

    def step_microbatches(self, model_inputs, *, loss_fn, losses, return_outputs):
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
            output = self.compute_model(primary, **inputs)
            scaled_loss = loss_fn(output, index)
            self.callback_losses.append(scaled_loss.detach())
            scaled_loss.backward()
            self.backward_calls += 1

    def eval_microbatches(self, model_inputs, *, loss_fn, losses, return_outputs):
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
            output = self.compute_model(primary, **inputs)
            loss = loss_fn(output, index)
            self.callback_losses.append(loss.detach())


def _pipeline_mesh_context():
    return SimpleNamespace(pp_size=2, cp_size=1, device_mesh=None, process_group=None)


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


def test_engine_and_datum_are_lazy_top_level_exports():
    assert PublicEngine is Engine
    assert PublicDatum is Datum
    assert PublicLossInputLayout is LossInputLayout
    assert PublicCollatedLossInputs is CollatedLossInputs


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


def test_planned_multi_call_matches_one_window_with_unequal_denominators():
    window_a = [_datum([2, 100], [1.0, 0.0])]
    window_b = [_datum([4, 8], [0.5, 1.5])]

    reference_model = ScaleModel()
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    reference_engine = Engine(
        reference_model,
        device="cpu",
        optimizers=reference_optimizer,
        max_grad_norm=None,
    )
    reference_result = reference_engine.forward_backward(window_a + window_b, _identity_loss)
    reference_grad = reference_model.weight.grad.detach().clone()
    reference_step = reference_engine.optim_step()

    planned_model = ScaleModel()
    planned_optimizer = torch.optim.SGD(planned_model.parameters(), lr=0.1)
    planned_engine = Engine(
        planned_model,
        device="cpu",
        optimizers=planned_optimizer,
        max_grad_norm=None,
    )
    planned_engine.begin_accumulation([window_a, window_b])
    result_a = planned_engine.forward_backward(window_a, _identity_loss)
    result_b = planned_engine.forward_backward(window_b, _identity_loss)
    planned_grad = planned_model.weight.grad.detach().clone()
    planned_step = planned_engine.optim_step()

    assert result_a.loss_sum.item() == pytest.approx(2.0)
    assert result_a.weight_sum.item() == pytest.approx(1.0)
    assert result_a.loss.item() == pytest.approx(2.0)
    assert result_b.loss_sum.item() == pytest.approx(14.0)
    assert result_b.weight_sum.item() == pytest.approx(2.0)
    assert result_b.loss.item() == pytest.approx(7.0)
    assert reference_result.loss_sum.item() == pytest.approx(16.0)
    assert reference_result.weight_sum.item() == pytest.approx(3.0)
    assert reference_result.loss.item() == pytest.approx(16.0 / 3.0)
    torch.testing.assert_close(planned_grad, reference_grad)
    torch.testing.assert_close(planned_step.grad_norm, reference_step.grad_norm)
    torch.testing.assert_close(planned_model.weight, reference_model.weight)


def test_explicit_one_window_accumulation_matches_implicit_call():
    implicit_window = [_datum([1, 9], [0.25, 0.75])]
    explicit_window = [_datum([1, 9], [0.25, 0.75])]

    implicit_model = ScaleModel()
    implicit_optimizer = torch.optim.SGD(implicit_model.parameters(), lr=0.1)
    implicit_engine = Engine(
        implicit_model,
        device="cpu",
        optimizers=implicit_optimizer,
        max_grad_norm=None,
    )
    implicit_result = implicit_engine.forward_backward(implicit_window, _identity_loss)

    explicit_model = ScaleModel()
    explicit_optimizer = torch.optim.SGD(explicit_model.parameters(), lr=0.1)
    explicit_engine = Engine(
        explicit_model,
        device="cpu",
        optimizers=explicit_optimizer,
        max_grad_norm=None,
    )
    explicit_engine.begin_accumulation([explicit_window])
    explicit_result = explicit_engine.forward_backward(explicit_window, _identity_loss)

    torch.testing.assert_close(explicit_result.loss, implicit_result.loss)
    torch.testing.assert_close(explicit_result.loss_sum, implicit_result.loss_sum)
    torch.testing.assert_close(explicit_result.weight_sum, implicit_result.weight_sum)
    torch.testing.assert_close(explicit_model.weight.grad, implicit_model.weight.grad)
    explicit_engine.optim_step()
    implicit_engine.optim_step()
    torch.testing.assert_close(explicit_model.weight, implicit_model.weight)


def test_begin_accumulation_requires_an_optimizer_before_forward():
    model = ScaleModel()
    engine = Engine(model, device="cpu")

    with pytest.raises(RuntimeError, match="optimizer"):
        engine.begin_accumulation([[_datum([1])]])

    assert model.forward_calls == 0
    assert model.weight.grad is None


def test_multiple_backward_calls_with_an_optimizer_require_an_explicit_plan():
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = Engine(model, device="cpu", optimizers=optimizer, max_grad_norm=None)

    engine.forward_backward([_datum([2])], _identity_loss)
    with pytest.raises(RuntimeError, match="begin_accumulation"):
        engine.forward_backward([_datum([6])], _identity_loss)

    torch.testing.assert_close(model.weight.grad, torch.tensor(2.0))
    engine.optim_step()
    torch.testing.assert_close(model.weight, torch.tensor(0.8))


def test_failed_implicit_backward_poisoned_gradients_cannot_be_reused():
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

    # The first microbatch already produced a partial gradient. It must never
    # be consumed or silently combined with a later optimizer window.
    torch.testing.assert_close(model.weight.grad, torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="failed"):
        engine.forward_backward([_datum([4])], _identity_loss)
    with pytest.raises(RuntimeError, match="failed"):
        engine.optim_step()
    with pytest.raises(RuntimeError, match="cleared gradients"):
        engine.begin_accumulation([[_datum([4])]])
    torch.testing.assert_close(model.weight, torch.tensor(1.0))


def test_planned_multi_call_uses_one_lifecycle_and_whole_step_moe_scale(monkeypatch):
    events = []

    @contextmanager
    def recording_sync_ctx(_model, is_optim_step, _defer_fsdp_grad_sync):
        events.append(f"sync-{is_optim_step}")
        yield

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
    monkeypatch.setattr(engine_module, "prepare_after_first_microbatch", lambda: events.append("after-first"))
    monkeypatch.setattr(engine_module, "get_sync_ctx", recording_sync_ctx)
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)

    model = _MainAndAuxScaleModel()
    engine = Engine(model, device="cpu", optimizers=torch.optim.SGD(model.parameters(), lr=0.1))
    # A zero-weight call still counts toward the whole-step MoE microbatch
    # average even though it contributes no main-loss numerator.
    window_a = [_datum([100], [0.0])]
    window_b = [_datum([3]), _datum([5])]

    engine.begin_accumulation([window_a, window_b])
    result_a = engine.forward_backward(window_a, _identity_loss)
    result_b = engine.forward_backward(window_b, _identity_loss)

    assert events == [
        "prepare",
        "sync-False",
        "after-first",
        "sync-False",
        "final",
        "sync-True",
    ]
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(1.0 / 3.0)
    assert result_a.loss_sum.item() == pytest.approx(0.0)
    assert result_a.weight_sum.item() == pytest.approx(0.0)
    assert result_a.loss.item() == pytest.approx(0.0)
    assert result_b.loss.item() == pytest.approx(4.0)
    assert model.main_weight.grad.item() == pytest.approx(4.0)
    assert model.aux_weight.grad.item() == pytest.approx(1.0)
    assert model.forward_calls == 3


def test_planned_accumulation_rejects_unfinished_extra_and_double_steps(monkeypatch):
    events = []

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("step")
            return super().step(closure)

        def zero_grad(self, set_to_none=True):
            events.append("zero")
            return super().zero_grad(set_to_none=set_to_none)

    class RecordingScheduler:
        def step(self, increment):
            assert increment == 1
            events.append("scheduler")

    model = ScaleModel()
    optimizer = RecordingSGD(model.parameters(), lr=0.1)
    engine = Engine(
        model,
        device="cpu",
        optimizers=optimizer,
        lr_schedulers=RecordingScheduler(),
        max_grad_norm=None,
    )
    real_finalize = engine_module.scale_grads_and_clip_grad_norm

    def recording_finalize(**kwargs):
        events.append("finalize")
        return real_finalize(**kwargs)

    monkeypatch.setattr(engine_module, "scale_grads_and_clip_grad_norm", recording_finalize)
    window_a = [_datum([1])]
    window_b = [_datum([3])]
    engine.begin_accumulation([window_a, window_b])
    engine.forward_backward(window_a, _identity_loss)

    with pytest.raises(RuntimeError):
        engine.optim_step()
    with pytest.raises(RuntimeError):
        engine.begin_accumulation([window_a, window_b])
    assert events == []

    engine.forward_backward(window_b, _identity_loss)
    with pytest.raises(RuntimeError):
        engine.forward_backward(window_b, _identity_loss)
    result = engine.optim_step()
    assert events == ["finalize", "step", "zero", "scheduler"]
    assert result.learning_rates == (0.1,)

    with pytest.raises(RuntimeError):
        engine.optim_step()
    assert events == ["finalize", "step", "zero", "scheduler"]

    # A new successful implicit window starts a new optimizer step.
    engine.forward_backward(window_a, _identity_loss)
    engine.optim_step()
    assert events == [
        "finalize",
        "step",
        "zero",
        "scheduler",
        "finalize",
        "step",
        "zero",
        "scheduler",
    ]


def test_planned_accumulation_failure_breaks_engine_before_optimizer_step():
    events = []

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("step")
            return super().step(closure)

    model = ScaleModel()
    optimizer = RecordingSGD(model.parameters(), lr=0.1)
    engine = Engine(model, device="cpu", optimizers=optimizer)
    window_a = [_datum([1])]
    window_b = [_datum([3])]
    engine.begin_accumulation([window_a, window_b])
    engine.forward_backward(window_a, _identity_loss)

    def failing_loss(_output, _loss_inputs):
        raise ValueError("loss callback failed")

    with pytest.raises(ValueError, match="loss callback failed"):
        engine.forward_backward(window_b, failing_loss)

    with pytest.raises(RuntimeError):
        engine.optim_step()
    with pytest.raises(RuntimeError):
        engine.begin_accumulation([window_a])
    with pytest.raises(RuntimeError):
        engine.forward_backward(window_b, _identity_loss)
    assert events == []
    torch.testing.assert_close(model.weight, torch.tensor(1.0))


@pytest.mark.parametrize("mismatch", ["datum_reference", "weights"])
def test_planned_accumulation_validates_declared_datums_before_forward(mismatch):
    model = ScaleModel()
    engine = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    window = [_datum([1], [1.0]), _datum([2], [1.0])]
    engine.begin_accumulation([window])

    actual_window = window
    if mismatch == "datum_reference":
        actual_window = [_datum([1], [1.0]), window[1]]
    else:
        window[0].loss_fn_inputs["weights"].mul_(2.0)

    with pytest.raises((RuntimeError, ValueError)):
        engine.forward_backward(actual_window, _identity_loss)
    assert model.forward_calls == 0


def test_planned_accumulation_rejects_microbatch_size_changes_before_forward():
    model = ScaleModel()
    engine = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    window = [_datum([1]), _datum([2])]
    engine.begin_accumulation([window])
    engine.microbatch_size = 1

    with pytest.raises(RuntimeError, match="microbatch_size changed"):
        engine.forward_backward(window, _identity_loss)

    assert model.forward_calls == 0


def test_planned_accumulation_rejects_nonfinal_partial_outer_microbatch():
    model = ScaleModel()
    engine = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    window_a = [_datum([1])]
    window_b = [_datum([2])]

    with pytest.raises(ValueError, match="microbatch|aligned|divisible"):
        engine.begin_accumulation([window_a, window_b])

    assert model.forward_calls == 0


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


def test_raw_thd_packed_collater_is_prepared_by_context_parallel_sharder():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    seen = {}

    def loss_fn(output, inputs):
        seen.update(inputs)
        assert inputs["weights"].shape == output.shape == (3,)
        return output

    result = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=partial(collate_datums, packed=True),
    ).forward_backward([_datum([1, 2]), _datum([3])], loss_fn)

    assert result.loss.item() == pytest.approx(2.0)
    assert "seq_lens" not in seen
    assert "seq_lens_padded" not in seen
    assert seen["cu_seqlens"].tolist() == [0, 2, 3]


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
            assert cp_context_active
            assert input_ids.tolist() == [[5, 6, 9, 9]]
            return super().forward(input_ids, *args, **kwargs)

    model = CPModel()
    mesh = _CPMesh(size=2, rank=1)
    mesh_context = SimpleNamespace(pp_size=1, cp_size=2, device_mesh=mesh)
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]])},
        loss_fn_inputs={
            "target_tokens": torch.tensor([[11, 12, 13, 14, 15, 16]]),
            "weights": torch.ones(1, 6),
            "advantages": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        },
    )
    engine = Engine(
        model,
        device="cpu",
        mesh_context=mesh_context,
        collate_fn=collate_prebatched,
        padding_token_id=9,
    )
    # This is a layout-only CPU test with a fake mesh. Distributed CP loss and
    # gradient scaling are covered separately with a real process group.
    engine._dp_group_and_size = lambda: (None, 1)
    engine._gradient_group_and_size = lambda _group, _size: (None, 1)
    model.weight.register_hook(
        lambda grad: grad if cp_context_active else pytest.fail("CP context ended before backward")
    )

    def loss_fn(output, inputs):
        assert cp_context_active
        assert inputs["target_tokens"].tolist() == [[15, 16, 0, 0]]
        assert inputs["weights"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
        torch.testing.assert_close(inputs["advantages"], torch.tensor([[0.5, 0.6, 0.0, 0.0]]))
        return output

    result = engine.forward_backward([datum], loss_fn)

    assert result.loss.item() == pytest.approx(11 / 6)
    assert model.weight.grad.item() == pytest.approx(11 / 6)
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(2.0)
    assert not cp_context_active


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


def test_pipeline_rejects_planned_multi_call_before_forward():
    model = ScaleModel()
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    engine = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
    )
    window_a = [_datum([1, 2])]
    window_b = [_datum([3, 4])]

    with pytest.raises(NotImplementedError, match="pipeline|PP"):
        engine.begin_accumulation([window_a, window_b])

    assert pipeline.step_calls == 0
    assert pipeline.backward_calls == 0
    assert model.forward_calls == 0
    assert model.weight.grad is None


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


def test_megatron_fsdp_per_token_loss_mode_fails_before_forward():
    model = ScaleModel()
    model.calculate_per_token_loss = True

    with pytest.raises(NotImplementedError, match="calculate_per_token_loss=True"):
        Engine(model, device="cpu").forward_backward([_datum([1])], _identity_loss)

    assert model.forward_calls == 0
    assert model.weight.grad is None


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

        # Planned accumulation must use the sum of the two DP-only
        # denominators while letting CP ranks contribute disjoint numerators.
        # The two calls deliberately have different weight sums, and DP ranks
        # deliberately own different amounts of supervision when dp_size=2.
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

        planned_model = _DDPWithCP(_DistributedCPModel())
        planned_optimizer = torch.optim.SGD(planned_model.parameters(), lr=0.1)
        planned_engine = Engine(
            planned_model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
            optimizers=planned_optimizer,
            max_grad_norm=None,
        )
        planned_engine.begin_accumulation([window_a, window_b])
        result_a = planned_engine.forward_backward(window_a, _identity_loss)
        result_b = planned_engine.forward_backward(window_b, _identity_loss)

        assert result_a.loss_sum.item() == pytest.approx(expected_a_sum)
        assert result_a.weight_sum.item() == pytest.approx(expected_a_weight)
        assert result_a.loss.item() == pytest.approx(expected_a_sum / expected_a_weight)
        assert result_b.loss_sum.item() == pytest.approx(expected_b_sum)
        assert result_b.weight_sum.item() == pytest.approx(expected_b_weight)
        assert result_b.loss.item() == pytest.approx(expected_b_sum / expected_b_weight)
        expected_global_mean = (expected_a_sum + expected_b_sum) / (expected_a_weight + expected_b_weight)
        assert planned_model.module.weight.grad.item() == pytest.approx(expected_global_mean)

        planned_engine.optim_step()
        assert planned_model.module.weight.item() == pytest.approx(1.0 - 0.1 * expected_global_mean)
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


def _planned_accumulation_validation_consensus_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        model = nn.parallel.DistributedDataParallel(ScaleModel())
        engine = Engine(
            model,
            device="cpu",
            optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
        )
        window = [_datum([rank + 1])]
        planned_windows = [window] if rank == 0 else [window, [_datum([rank + 2])]]

        with pytest.raises((RuntimeError, ValueError)):
            engine.begin_accumulation(planned_windows)
        assert model.module.forward_calls == 0
        dist.barrier()

        model = nn.parallel.DistributedDataParallel(ScaleModel())
        engine = Engine(
            model,
            device="cpu",
            optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
        )
        window = [_datum([rank + 1])]
        engine.begin_accumulation([window])
        if rank == 0:
            window[0].loss_fn_inputs["weights"].mul_(2.0)

        # Rank 1 must observe rank 0's local plan-validation failure instead of
        # entering DDP forward and hanging on a different collective.
        with pytest.raises((RuntimeError, ValueError)):
            engine.forward_backward(window, _identity_loss)
        assert model.module.forward_calls == 0
    finally:
        dist.destroy_process_group()


def _model_parallel_planned_output_error_worker(rank: int, world_size: int, init_file: str) -> None:
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
            ParallelismSizes(dp_size=1, tp_size=world_size),
            world_size=world_size,
        )
        model = ScaleModel()
        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
        )
        window = [_datum([1, 2])]
        engine.begin_accumulation([window])

        def loss_with_rank_local_output_error(output, _loss_inputs):
            return output, ([{}, {}] if rank == 0 else [{}])

        expected = "one mapping per Datum" if rank == 0 else "another model-parallel rank"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            engine.forward_backward(window, loss_with_rank_local_output_error)

        # Both model-parallel ranks complete backward, then enter the same
        # terminal state even though only rank 0 owns the bad output contract.
        assert model.weight.grad is not None
        assert engine._accumulation_state is not None
        assert engine._accumulation_state.status == "broken"
        with pytest.raises(RuntimeError, match="broken"):
            engine.optim_step()

        model = ScaleModel()
        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
        )
        window = [_datum([1, 2])]
        engine.begin_accumulation([window])

        def rank_local_loss_failure(output, _loss_inputs):
            if rank == 0:
                raise ValueError("rank-local loss failure")
            return output

        expected = "rank-local loss failure" if rank == 0 else "another model-parallel rank"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            engine.forward_backward(window, rank_local_loss_failure)

        # Loss/shape errors are agreed before backward, so no peer enters a
        # TP/EP/DP backward collective while another exits locally.
        assert model.weight.grad is None
        assert engine._accumulation_state is not None
        assert engine._accumulation_state.status == "broken"
    finally:
        dist.destroy_process_group()


def _model_parallel_optimizer_fence_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    real_finalize = engine_module.scale_grads_and_clip_grad_norm
    try:
        mesh_context = MeshContext.build(
            MegatronFSDPConfig(),
            ParallelismSizes(dp_size=1, tp_size=world_size),
            world_size=world_size,
        )
        steps = []

        class RecordingSGD(torch.optim.SGD):
            def step(self, closure=None):
                steps.append("step")
                return super().step(closure)

        model = ScaleModel()
        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            optimizers=RecordingSGD(model.parameters(), lr=0.1),
            max_grad_norm=None,
        )
        window = [_datum([1, 2])]
        engine.begin_accumulation([window])
        engine.forward_backward(window, _identity_loss)

        finalize_calls = []

        def recording_finalize(**_kwargs):
            finalize_calls.append("finalize")
            return torch.tensor(1.5)

        engine_module.scale_grads_and_clip_grad_norm = recording_finalize

        def rank_local_fence():
            if rank == 0:
                raise ValueError("rank-local fence failure")

        expected = "rank-local fence failure" if rank == 0 else "another model-parallel rank"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            engine.optim_step(before_optimizer_step=rank_local_fence)

        assert finalize_calls == ["finalize"]
        assert steps == []
        torch.testing.assert_close(model.weight, torch.tensor(1.0))
        assert model.weight.grad is not None

        result = engine.optim_step()
        assert finalize_calls == ["finalize"]
        assert steps == ["step"]
        torch.testing.assert_close(result.grad_norm, torch.tensor(1.5))
        torch.testing.assert_close(model.weight, torch.tensor(0.85))
    finally:
        engine_module.scale_grads_and_clip_grad_norm = real_finalize
        dist.destroy_process_group()


def _planned_cp_preflight_consensus_worker(rank: int, world_size: int, init_file: str) -> None:
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
            ParallelismSizes(dp_size=2, cp_size=2),
            world_size=world_size,
        )
        model = ScaleModel()
        engine = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            optimizers=torch.optim.SGD(model.parameters(), lr=0.1),
        )
        # Only one CP subgroup disagrees. The full-control error reduction must
        # stop all four ranks before any rank enters the later DP all-reduce.
        weight = 2.0 if rank == 0 else 1.0
        window = [_datum([rank + 1], [weight])]

        with pytest.raises((ValueError, RuntimeError), match="context-parallel"):
            engine.begin_accumulation([window])

        assert engine._accumulation_state is None
        assert model.forward_calls == 0
        assert model.weight.grad is None
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_data_parallel_window_uses_global_numerator_and_denominator(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(2, str(tmp_path / "engine_dist_init")),
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


def test_planned_accumulation_validation_errors_reach_every_data_rank(tmp_path):
    mp.spawn(
        _planned_accumulation_validation_consensus_worker,
        args=(2, str(tmp_path / "engine_planned_validation_init")),
        nprocs=2,
        join=True,
    )


def test_planned_output_error_breaks_every_model_parallel_rank_after_backward(tmp_path):
    mp.spawn(
        _model_parallel_planned_output_error_worker,
        args=(2, str(tmp_path / "engine_model_parallel_output_init")),
        nprocs=2,
        join=True,
    )


def test_optimizer_fence_failure_reaches_every_model_parallel_rank_and_can_retry(tmp_path):
    mp.spawn(
        _model_parallel_optimizer_fence_worker,
        args=(2, str(tmp_path / "engine_model_parallel_fence_init")),
        nprocs=2,
        join=True,
    )


def test_planned_cp_preflight_failure_reaches_all_dp_cp_ranks(tmp_path):
    mp.spawn(
        _planned_cp_preflight_consensus_worker,
        args=(4, str(tmp_path / "engine_planned_cp_preflight_init")),
        nprocs=4,
        join=True,
    )
