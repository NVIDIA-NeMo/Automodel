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
from nemo_automodel.components.loss.causal_lm import causal_lm_loss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
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


def _identity_loss(output, _loss_inputs, _datums, _model_inputs):
    return output


def test_engine_and_datum_are_lazy_top_level_exports():
    assert PublicEngine is Engine
    assert PublicDatum is Datum


def test_forward_backward_uses_one_denominator_for_the_window():
    model = ScaleModel()
    initial_weight = model.weight.detach().clone()
    engine = Engine(model, device="cpu")

    loss, outputs = engine.forward_backward(
        [[_datum([1, 2])], [_datum([3])]],
        _identity_loss,
    )

    assert loss.item() == pytest.approx(2.0)
    assert outputs == []
    assert model.weight.grad.item() == pytest.approx(2.0)
    assert torch.equal(model.weight, initial_weight)
    assert model.forward_calls == 2


def test_raw_thd_packed_collater_is_prepared_by_context_parallel_sharder():
    model = ScaleModel()
    model.backend = SimpleNamespace(attn="te")
    seen = {}

    def loss_fn(output, inputs, _datums, model_inputs):
        seen.update(model_inputs)
        assert inputs["weights"].shape == output.shape == (3,)
        return output

    loss, _ = Engine(
        model,
        device="cpu",
        collate_fn=partial(collate_datums, packed=True),
    ).forward_backward([[_datum([1, 2]), _datum([3])]], loss_fn)

    assert loss.item() == pytest.approx(2.0)
    assert "seq_lens" not in seen
    assert "seq_lens_padded" not in seen
    assert seen["qkv_format"] == "thd"
    assert seen["cu_seqlens"].tolist() == [0, 2, 3]


def test_raw_thd_requires_a_thd_capable_context_parallel_sharder():
    model = ScaleModel()

    with pytest.raises(ValueError, match="could not prepare raw THD inputs"):
        Engine(
            model,
            device="cpu",
            collate_fn=partial(collate_datums, packed=True),
        ).forward_backward([[_datum([1, 2]), _datum([3])]], _identity_loss)

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

        def forward(self, *args, **kwargs):
            assert cp_context_active
            return super().forward(*args, **kwargs)

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

    def loss_fn(output, inputs, _datums, model_inputs):
        assert cp_context_active
        assert model_inputs["input_ids"].tolist() == [[5, 6, 9, 9]]
        assert inputs["target_tokens"].tolist() == [[15, 16, 0, 0]]
        assert inputs["weights"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
        torch.testing.assert_close(inputs["advantages"], torch.tensor([[0.5, 0.6, 0.0, 0.0]]))
        return output

    loss, _ = engine.forward_backward([[datum]], loss_fn)

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

    def sequence_loss(output, _loss_inputs, datums, _model_inputs):
        chunks = output.squeeze(0).split([datum.seq_len for datum in datums])
        losses = torch.cat([chunk * datum.loss_fn_inputs["sequence_scale"] for chunk, datum in zip(chunks, datums)])
        outputs = [
            {"sequence_sum": chunk.sum(), "sequence_length": datum.seq_len} for chunk, datum in zip(chunks, datums)
        ]
        return losses, outputs

    loss, outputs = Engine(
        model,
        device="cpu",
        collate_fn=_model_ready_packed_collate,
    ).forward_backward([[first, second]], sequence_loss)

    assert loss.item() == pytest.approx(2.5)
    assert model.weight.grad.item() == pytest.approx(2.5)
    assert [item["sequence_length"] for item in outputs] == [2, 1]
    assert [item["sequence_sum"].item() for item in outputs] == pytest.approx([3.0, 3.0])


def test_weights_mask_loss_and_denominator():
    model = ScaleModel()
    loss, _ = Engine(model, device="cpu").forward_backward(
        [[_datum([1, 100], [1.0, 0.0])], [_datum([3, 5], [0.5, 1.0])]],
        _identity_loss,
    )

    assert loss.item() == pytest.approx(3.0)
    assert model.weight.grad.item() == pytest.approx(3.0)


def test_loss_fn_outputs_follow_datum_order_and_are_detached():
    model = ScaleModel()

    def loss_with_outputs(output, _loss_inputs, datums, _model_inputs):
        return output, [
            {"first_token": datum.input_ids[0], "model_value": output[index].sum()}
            for index, datum in enumerate(datums)
        ]

    _, outputs = Engine(model, device="cpu").forward_backward(
        [[_datum([1, 2]), _datum([3])], [_datum([4])]],
        loss_with_outputs,
    )

    assert [item["first_token"].item() for item in outputs] == [1, 3, 4]
    assert all(not item["model_value"].requires_grad for item in outputs)


def test_loss_fn_outputs_must_align_with_datums():
    model = ScaleModel()
    with pytest.raises(ValueError, match="one mapping per Datum"):
        Engine(model, device="cpu").forward_backward(
            [[_datum([1]), _datum([2])]],
            lambda output, _inputs, _datums, _model_inputs: (output, [{"only": "one"}]),
        )
    assert model.weight.grad is None


def test_loss_fn_outputs_must_be_consistent_across_the_window():
    def inconsistent_outputs(output, _inputs, datums, _model_inputs):
        if datums[0].input_ids[0].item() == 1:
            return output, [{"value": output.sum()}]
        return output

    with pytest.raises(ValueError, match="every microbatch or none"):
        Engine(ScaleModel(), device="cpu").forward_backward(
            [[_datum([1])], [_datum([2])]],
            inconsistent_outputs,
        )


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.output = nn.Linear(4, 8)

    def forward(self, input_ids, **_):
        return self.output(self.embedding(input_ids))


class TinyCausalLM(TinyLM):
    def forward(self, input_ids, **_):
        return SimpleNamespace(logits=super().forward(input_ids))


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

    def policy_loss(logits, inputs, datums, _model_inputs):
        assert len(datums) == 1
        new_logprobs = -F.cross_entropy(
            logits.flatten(0, 1),
            inputs["target_tokens"].flatten(),
            reduction="none",
        ).view_as(inputs["weights"])
        ratio = torch.exp(new_logprobs - inputs["logprobs"])
        losses = -(ratio * inputs["advantages"])
        return losses, [{"policy_sum": (losses * inputs["weights"]).sum()}]

    loss, outputs = Engine(model, device="cpu").forward_backward([[datum]], policy_loss)

    assert torch.isfinite(loss)
    assert torch.isfinite(outputs[0]["policy_sum"])
    assert not outputs[0]["policy_sum"].requires_grad
    assert model.embedding.weight.grad is not None
    assert model.output.weight.grad is not None


def test_causal_lm_loss_matches_a_manual_accumulation_window():
    torch.manual_seed(7)
    model = TinyCausalLM()
    reference = TinyCausalLM()
    reference.load_state_dict(model.state_dict())
    batches = [
        (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, -100]])),
        (torch.tensor([[4, 5]]), torch.tensor([[5, 6]])),
    ]
    loss_fn = MaskedCrossEntropy()
    mtp_config = SimpleNamespace(scaling_factor=None, ignore_index=-100)

    window = [
        [
            Datum(
                model_inputs={"input_ids": input_ids},
                loss_fn_inputs={"labels": labels, "weights": labels.ne(-100)},
            )
        ]
        for input_ids, labels in batches
    ]

    def engine_loss(output, inputs, _datums, _model_inputs):
        return causal_lm_loss(
            loss_fn,
            model,
            output,
            inputs["labels"],
            mtp_config,
            num_label_tokens=None,
            grad_reduce_group=None,
        )

    actual_loss, _ = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward(window, engine_loss)

    denominator = sum((labels != -100).sum() for _, labels in batches)
    reference_loss = (
        sum(
            F.cross_entropy(reference(input_ids).logits.flatten(0, 1), labels.flatten(), reduction="sum")
            for input_ids, labels in batches
        )
        / denominator
    )
    reference_loss.backward()

    torch.testing.assert_close(actual_loss, reference_loss.detach().to(actual_loss))
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected.grad)


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
        [[_datum([1])], [_datum([2])]],
        _identity_loss,
    )

    assert events == ["prepare", "sync:False", "after_first", "final", "sync:True"]


def test_forward_context_covers_forward_loss_and_backward():
    active = False

    @contextmanager
    def forward_context():
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

    def loss_fn(output, _inputs, _datums, _model_inputs):
        assert active
        return output

    Engine(model, device="cpu", context_fn=forward_context).forward_backward([[_datum([1, 2])]], loss_fn)
    assert not active


def test_window_sets_the_same_moe_aux_scale_as_the_recipes(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)

    Engine(ScaleModel(), device="cpu").forward_backward(
        [[_datum([1])], [_datum([2])]],
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
    Engine(model, device="cpu", collate_fn=_vlm_collate).forward_backward(
        [datums],
        lambda output, _inputs, _datums, _model_inputs: output,
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

    loss, _ = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward([[datum]], _identity_loss)

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

    loss, _ = Engine(model, device="cpu", collate_fn=collate_prebatched).forward_backward([[datum]], _identity_loss)

    assert torch.isfinite(loss)
    assert model.text.weight.grad is not None
    assert model.vision.weight.grad is not None


def test_scalar_loss_is_a_local_weighted_sum_numerator():
    model = ScaleModel()

    loss, _ = Engine(model, device="cpu").forward_backward(
        [[_datum([1, 100], [1.0, 0.0])], [_datum([3, 5], [0.5, 1.0])]],
        lambda output, inputs, _datums, _model_inputs: (output * inputs["weights"]).sum(),
    )

    assert loss.item() == pytest.approx(3.0)
    assert model.weight.grad.item() == pytest.approx(3.0)


def test_zero_weights_fail_before_forward():
    model = ScaleModel()
    with pytest.raises(ValueError, match="positive global weight sum"):
        Engine(model, device="cpu").forward_backward(
            [[_datum([1, 2], [0.0, 0.0])]],
            _identity_loss,
        )
    assert model.forward_calls == 0
    assert model.weight.grad is None


def test_pipeline_parallelism_fails_before_forward():
    model = ScaleModel()
    mesh_context = SimpleNamespace(pp_size=2, cp_size=1)

    with pytest.raises(NotImplementedError, match="pipeline"):
        Engine(model, device="cpu", mesh_context=mesh_context).forward_backward(
            [[_datum([1])]],
            _identity_loss,
        )

    assert model.forward_calls == 0


def test_megatron_fsdp_per_token_loss_mode_fails_before_forward():
    model = ScaleModel()
    model.calculate_per_token_loss = True

    with pytest.raises(NotImplementedError, match="calculate_per_token_loss=True"):
        Engine(model, device="cpu").forward_backward([[_datum([1])]], _identity_loss)

    assert model.forward_calls == 0
    assert model.weight.grad is None


def test_loss_shape_must_exactly_match_weights():
    model = ScaleModel()
    with pytest.raises(ValueError, match="exactly the same shape"):
        Engine(model, device="cpu").forward_backward(
            [[_datum([1, 2])]],
            lambda output, _inputs, _datums, _model_inputs: output[:, :1],
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
            [[_datum([1, 2])]],
            _identity_loss,
        )
    assert model.forward_calls == 0


def _distributed_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        model = nn.parallel.DistributedDataParallel(ScaleModel())
        bad_window = [[_datum([1])]] if rank == 0 else [[_datum([1])], [_datum([2])]]
        with pytest.raises(ValueError, match="same number of microbatches"):
            Engine(model, device="cpu").forward_backward(bad_window, _identity_loss)
        assert model.module.forward_calls == 0

        window = [[_datum([1, 2])], [_datum([3])]] if rank == 0 else [[_datum([4])], [_datum([5, 6])]]
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
                [
                    Datum(
                        model_inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
                        loss_fn_inputs={"weights": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
                    )
                ],
                [
                    Datum(
                        model_inputs={"input_ids": torch.tensor([[5, 6, 7, 8]])},
                        loss_fn_inputs={"weights": torch.tensor([[0.0, 0.0, 1.0, 1.0]])},
                    )
                ],
            ]
        else:
            dp_rank = get_flat_mesh(mesh_context.device_mesh, "dp").get_local_rank()
            first = dp_rank * 4 + 1
            window = [
                [
                    Datum(
                        model_inputs={"input_ids": torch.arange(first, first + 4).unsqueeze(0)},
                        loss_fn_inputs={"weights": torch.ones(1, 4)},
                    )
                ]
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
            ).forward_backward([[datum]], _identity_loss)

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
