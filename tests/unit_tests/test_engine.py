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
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.engine import Engine


class ScaleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, input_ids: torch.Tensor, **_) -> torch.Tensor:
        self.forward_calls += 1
        return input_ids.to(torch.float32) * self.weight


def _datum(values, weights=None) -> Datum:
    values = torch.tensor(values, dtype=torch.long)
    weights = torch.ones_like(values, dtype=torch.float32) if weights is None else torch.tensor(weights)
    return Datum(model_inputs={"input_ids": values}, loss_fn_inputs={"weights": weights})


def _identity_loss(output, _loss_inputs, _datums):
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


def test_padded_and_packed_windows_have_the_same_loss_and_gradient():
    padded_model = ScaleModel()
    packed_model = ScaleModel()
    window = [[_datum([1, 2]), _datum([3])]]

    padded_loss, _ = Engine(padded_model, device="cpu").forward_backward(window, _identity_loss)

    def packed_identity_loss(output, loss_inputs, datums):
        assert [datum.seq_len for datum in datums] == [2, 1]
        return _identity_loss(output, loss_inputs, datums)

    packed_loss, _ = Engine(
        packed_model,
        device="cpu",
        collate_fn=partial(collate_datums, packed=True),
    ).forward_backward(window, packed_identity_loss)

    torch.testing.assert_close(padded_loss, packed_loss)
    torch.testing.assert_close(padded_model.weight.grad, packed_model.weight.grad)


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

    def loss_with_outputs(output, _loss_inputs, datums):
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
            lambda output, _inputs, _datums: (output, [{"only": "one"}]),
        )
    assert model.weight.grad is None


def test_loss_fn_outputs_must_be_consistent_across_the_window():
    def inconsistent_outputs(output, _inputs, datums):
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

    def policy_loss(logits, inputs, datums):
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
        lambda output, _inputs, _datums: output,
    )

    assert model.text.weight.grad is not None
    assert model.text.weight.grad.abs().sum() > 0
    assert model.vision.weight.grad is not None
    assert model.vision.weight.grad.abs().sum() > 0


def test_zero_weights_fail_before_forward():
    model = ScaleModel()
    with pytest.raises(ValueError, match="positive global weight sum"):
        Engine(model, device="cpu").forward_backward(
            [[_datum([1, 2], [0.0, 0.0])]],
            _identity_loss,
        )
    assert model.forward_calls == 0
    assert model.weight.grad is None


@pytest.mark.parametrize(("pp_size", "cp_size", "name"), [(2, 1, "pipeline"), (1, 2, "context")])
def test_unsupported_parallelism_fails_before_forward(pp_size, cp_size, name):
    model = ScaleModel()
    mesh_context = SimpleNamespace(pp_size=pp_size, cp_size=cp_size)

    with pytest.raises(NotImplementedError, match=name):
        Engine(model, device="cpu", mesh_context=mesh_context).forward_backward(
            [[_datum([1])]],
            _identity_loss,
        )

    assert model.forward_calls == 0


def test_loss_shape_must_exactly_match_weights():
    model = ScaleModel()
    with pytest.raises(ValueError, match="exactly the same shape"):
        Engine(model, device="cpu").forward_backward(
            [[_datum([1, 2])]],
            lambda output, _inputs, _datums: output.sum(),
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


def test_data_parallel_window_uses_global_numerator_and_denominator(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(2, str(tmp_path / "engine_dist_init")),
        nprocs=2,
        join=True,
    )
