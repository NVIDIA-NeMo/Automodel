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

import sys
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

import nemo_automodel.engine as engine_module
from nemo_automodel import Datum as PublicDatum
from nemo_automodel import Engine as PublicEngine
from nemo_automodel.components.datasets.datum import Datum, collate_datums
from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    shard_batch_contiguous,
)
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.engine import Engine, collate_prebatched


class ScaleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        self.forward_calls += 1
        return input_ids.to(torch.float32) * self.weight


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


def test_forward_backward_uses_one_denominator_for_the_window():
    model = ScaleModel()
    initial_weight = model.weight.detach().clone()
    engine = Engine(model, device="cpu")

    loss, outputs = engine.forward_backward(
        [_datum([1, 2]), _datum([3])],
        _identity_loss,
    )

    assert loss.item() == pytest.approx(2.0)
    assert outputs == []
    assert model.weight.grad.item() == pytest.approx(2.0)
    assert torch.equal(model.weight, initial_weight)
    assert model.forward_calls == 2


def test_forward_backward_groups_flat_datums_by_microbatch_size():
    group_sizes = []

    def recording_collate(datums):
        group_sizes.append(len(datums))
        return collate_datums(datums)

    model = ScaleModel()
    loss, _ = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=recording_collate,
    ).forward_backward([_datum([value]) for value in range(1, 6)], _identity_loss)

    assert group_sizes == [2, 2, 1]
    assert model.forward_calls == 3
    assert loss.item() == pytest.approx(3.0)


def test_raw_thd_packed_collater_is_prepared_by_context_parallel_sharder():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    seen = {}

    def loss_fn(output, inputs):
        seen.update(inputs)
        assert inputs["weights"].shape == output.shape == (3,)
        return output

    loss, _ = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=partial(collate_datums, packed=True),
    ).forward_backward([_datum([1, 2]), _datum([3])], loss_fn)

    assert loss.item() == pytest.approx(2.0)
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

    loss, _ = engine.forward_backward([datum], loss_fn)

    assert loss.item() == pytest.approx(11 / 6)
    assert model.weight.grad.item() == pytest.approx(11 / 6)
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(2.0)
    assert not cp_context_active


def _model_ready_packed_collate(datums):
    model_inputs, loss_inputs = collate_datums(datums, packed=True)
    lengths = torch.tensor([datum.seq_len for datum in datums], dtype=torch.int32)
    model_inputs = {
        "input_ids": model_inputs["input_ids"].flatten(),
        "position_ids": model_inputs["position_ids"].flatten(),
        "cu_seqlens": F.pad(lengths.cumsum(0), (1, 0)),
        "max_seqlen": lengths.max(),
        "qkv_format": "thd",
    }
    loss_inputs = {
        key: value.flatten() if value.ndim == 2 and value.shape[0] == 1 else value for key, value in loss_inputs.items()
    }
    return model_inputs, loss_inputs


def test_packed_rl_callback_keeps_per_datum_sequence_boundaries():
    model = ScaleModel()
    first = _datum([1, 2])
    second = _datum([3])
    first.loss_fn_inputs["sequence_scale"] = torch.tensor(2.0)
    second.loss_fn_inputs["sequence_scale"] = torch.tensor(0.5)

    datums = [first, second]

    def sequence_loss(output, _loss_inputs):
        chunks = output.squeeze(0).split([datum.seq_len for datum in datums])
        losses = torch.cat([chunk * datum.loss_fn_inputs["sequence_scale"] for chunk, datum in zip(chunks, datums)])
        outputs = [
            {"sequence_sum": chunk.sum(), "sequence_length": datum.seq_len} for chunk, datum in zip(chunks, datums)
        ]
        return losses, outputs

    loss, outputs = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=_model_ready_packed_collate,
    ).forward_backward(datums, sequence_loss)

    assert loss.item() == pytest.approx(2.5)
    assert model.weight.grad.item() == pytest.approx(2.5)
    assert [item["sequence_length"] for item in outputs] == [2, 1]
    assert [item["sequence_sum"].item() for item in outputs] == pytest.approx([3.0, 3.0])


def test_weights_mask_loss_and_denominator():
    model = ScaleModel()
    loss, _ = Engine(model, device="cpu").forward_backward(
        [_datum([1, 100], [1.0, 0.0]), _datum([3, 5], [0.5, 1.0])],
        _identity_loss,
    )

    assert loss.item() == pytest.approx(3.0)
    assert model.weight.grad.item() == pytest.approx(3.0)


def test_fractional_weight_sum_below_one_is_not_clamped():
    model = ScaleModel()
    loss, _ = Engine(model, device="cpu").forward_backward(
        [_datum([2, 4], [0.2, 0.3])],
        _identity_loss,
    )

    assert loss.item() == pytest.approx(3.2)
    assert model.weight.grad.item() == pytest.approx(3.2)


def test_loss_fn_outputs_follow_datum_order_and_are_detached():
    model = ScaleModel()

    def loss_with_outputs(output, _loss_inputs):
        return output, [{"first_token": row.flatten()[0], "model_value": row.sum()} for row in output]

    _, outputs = Engine(model, device="cpu", microbatch_size=2).forward_backward(
        [_datum([1, 2]), _datum([3]), _datum([4])],
        loss_with_outputs,
    )

    assert [item["first_token"].item() for item in outputs] == [1, 3, 4]
    assert all(not item["model_value"].requires_grad for item in outputs)


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

    loss, outputs = Engine(model, device="cpu").forward_backward([datum], policy_loss)

    assert torch.isfinite(loss)
    assert torch.isfinite(outputs[0]["policy_sum"])
    assert not outputs[0]["policy_sum"].requires_grad
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

    loss, outputs = Engine(
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

    assert loss.item() == pytest.approx(4.5)
    assert model.weight.grad.item() == pytest.approx(4.5)
    assert outputs == []
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

    _, outputs = Engine(
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
    assert [item["metric"].item() for item in outputs] == [1.0, 2.0]


def test_pipeline_default_packed_collater_splits_flat_datums_inside_engine():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])
    datums = [_datum([1, 2]), _datum([3, 4]), _datum([5, 6]), _datum([7, 8])]

    def loss_with_outputs(output, _loss_inputs):
        return output, [
            {"pair_sum": output[..., :2].sum()},
            {"pair_sum": output[..., 2:].sum()},
        ]

    loss, outputs = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=4,
        collate_fn=partial(collate_datums, packed=True),
    ).forward_backward(datums, loss_with_outputs)

    assert loss.item() == pytest.approx(4.5)
    assert model.weight.grad.item() == pytest.approx(4.5)
    assert [item["pair_sum"].item() for item in outputs] == [3.0, 7.0, 11.0, 15.0]
    assert [item["input_ids"].shape for item in pipeline.prepared_inputs[0]] == [(1, 4), (1, 4)]


def test_pipeline_default_packed_collater_rejects_ambiguous_per_datum_loss_fields():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2)
    datums = [
        Datum(
            model_inputs={"input_ids": torch.tensor([index, index + 1])},
            loss_fn_inputs={"weights": torch.ones(2), "reward": torch.tensor(float(index))},
        )
        for index in (1, 3, 5, 7)
    ]

    with pytest.raises(NotImplementedError, match="per-Datum loss fields.*reward"):
        Engine(
            pipeline,
            device="cpu",
            mesh_context=_pipeline_mesh_context(),
            microbatch_size=4,
            collate_fn=partial(collate_datums, packed=True),
        ).forward_backward(datums, _identity_loss)


def test_pipeline_outputs_allow_uneven_datum_counts_at_final_thd_boundaries():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    pipeline = _FakeAutoPipeline(model, num_microbatches=2, callback_order=[1, 0])
    datums = [_datum([1]), _datum([2]), _datum([3]), _datum([4, 5, 6])]

    def final_thd_collate(items):
        lengths = [datum.seq_len for datum in items]
        tokens = torch.cat([datum.input_ids for datum in items])
        return (
            {
                "input_ids": tokens,
                "position_ids": torch.cat([torch.arange(length) for length in lengths]),
                "cu_seqlens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32),
                "max_seqlen": torch.tensor(max(lengths), dtype=torch.int32),
                "qkv_format": "thd",
            },
            {"weights": torch.cat([datum.loss_fn_inputs["weights"] for datum in items])},
        )

    def loss_with_outputs(output, _loss_inputs):
        first_token = int(output.reshape(-1)[0].item())
        ids = [1, 2, 3] if first_token == 1 else [4]
        return output, [{"datum_id": torch.tensor(datum_id)} for datum_id in ids]

    _, outputs = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=4,
        collate_fn=final_thd_collate,
    ).forward_backward(datums, loss_with_outputs)

    assert [item["datum_id"].item() for item in outputs] == [1, 2, 3, 4]


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
        objects[0] = expected

    monkeypatch.setattr(dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(dist, "get_rank", lambda *, group: 1)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, group_rank: 11 if group_rank == 0 else 12)
    monkeypatch.setattr(dist, "broadcast_object_list", fake_broadcast_object_list)

    result = engine._broadcast_pipeline_outputs([])

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

    assert engine._broadcast_pipeline_outputs([]) == []


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

    loss, _ = engine.forward_backward([datum], loss_fn)

    assert loss.item() == pytest.approx(18 / 8)
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

    loss, _ = Engine(
        pipeline,
        device="cpu",
        mesh_context=_pipeline_mesh_context(),
        microbatch_size=2,
    ).forward_backward([_datum([1]), _datum([2])], _identity_loss)

    assert loss.item() == pytest.approx(1.5)
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

    loss, _ = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward([datum], _identity_loss)

    assert loss.item() == pytest.approx(2.0)
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

    loss, _ = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward([datum], _identity_loss)

    assert torch.isfinite(loss)
    assert model.text.weight.grad is not None
    assert model.vision.weight.grad is not None


def test_scalar_loss_is_a_local_weighted_sum_numerator():
    model = ScaleModel()

    loss, _ = Engine(model, device="cpu").forward_backward(
        [_datum([1, 100], [1.0, 0.0]), _datum([3, 5], [0.5, 1.0])],
        lambda output, inputs: (output * inputs["weights"]).sum(),
    )

    assert loss.item() == pytest.approx(3.0)
    assert model.weight.grad.item() == pytest.approx(3.0)


def test_zero_weights_run_graph_connected_zero_backward():
    model = ScaleModel()
    loss, _ = Engine(model, device="cpu").forward_backward(
        [_datum([1, 2], [0.0, 0.0])],
        _identity_loss,
    )
    assert loss.item() == 0
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
        model = nn.parallel.DistributedDataParallel(ScaleModel())
        bad_window = [_datum([1])] if rank == 0 else [_datum([1]), _datum([2])]
        with pytest.raises(ValueError, match="same number of microbatches"):
            Engine(model, device="cpu").forward_backward(bad_window, _identity_loss)
        assert model.module.forward_calls == 0

        window = [_datum([1, 2]), _datum([3])] if rank == 0 else [_datum([4]), _datum([5, 6])]
        loss, outputs = Engine(model, device="cpu").forward_backward(window, _identity_loss)
        assert loss.item() == pytest.approx(3.5)
        assert outputs == []
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

        loss, _ = Engine(
            model,
            device="cpu",
            mesh_context=mesh_context,
            collate_fn=collate_prebatched,
        ).forward_backward(window, _identity_loss)

        assert loss.item() == pytest.approx(4.5)
        assert model.module.weight.grad.item() == pytest.approx(4.5)
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
