# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel import Datum, Engine, LossOutput
from nemo_automodel.components.datasets.datum import LossInputLayout, collate_vlm_datums


class TokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Scale one padded token batch.

        Args:
            input_ids: Token IDs of shape ``[batch, sequence]``.

        Returns:
            Floating token values of shape ``[batch, sequence]``.
        """
        return input_ids.to(torch.float32) * self.scale


def test_forward_needs_no_loss_fields_and_restores_model_mode() -> None:
    model = TokenModel()
    model.train()
    engine = Engine(model, device="cpu")

    outputs = engine.forward(
        [Datum(input_ids=[1, 2, 3]), Datum(input_ids=[4, 5])],
        lambda output, _inputs: output,
    )

    assert model.training
    torch.testing.assert_close(outputs[0], torch.tensor([2.0, 4.0, 6.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([8.0, 10.0]))


def test_forward_restores_model_mode_after_compute_failure() -> None:
    model = TokenModel()
    model.train()

    with pytest.raises(RuntimeError, match="failed"):
        Engine(model, device="cpu").forward(
            [Datum(input_ids=[1, 2])],
            lambda _output, _inputs: (_ for _ in ()).throw(RuntimeError("failed")),
        )

    assert model.training


def test_forward_restores_intentionally_mixed_child_modes() -> None:
    model = TokenModel()
    model.frozen_tower = nn.Linear(1, 1)
    model.train()
    model.frozen_tower.eval()

    Engine(model, device="cpu").forward([Datum(input_ids=[1, 2])], lambda output, _inputs: output)

    assert model.training
    assert not model.frozen_tower.training


def test_vlm_forward_uses_prediction_axis_without_fake_labels_or_weights() -> None:
    processor = SimpleNamespace(
        image_token_id=99,
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([1, 99, 2, 3]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "pixel_values": torch.ones(1, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )
    engine = Engine(
        TokenModel(),
        device="cpu",
        collate_fn=partial(collate_vlm_datums, processor=processor),
    )

    outputs = engine.forward([datum], lambda output, _inputs: output)

    torch.testing.assert_close(outputs[0], torch.tensor([2.0, 198.0, 4.0]))


def test_packed_vlm_forward_accepts_r3_routes_with_trailing_feature_axes() -> None:
    processor = SimpleNamespace(
        image_token_id=99,
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    datums = []
    for first_token in (1, 5):
        input_ids = torch.tensor([first_token, 99, first_token + 1, first_token + 2])
        datums.append(
            Datum(
                model_inputs={
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "pixel_values": torch.ones(1, 2),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                loss_fn_inputs={
                    "routed_experts": torch.arange(6, dtype=torch.int16).reshape(3, 2, 1),
                    "target_tokens": input_ids[1:].clone(),
                },
                loss_fn_input_layouts={
                    "routed_experts": LossInputLayout.PER_TOKEN,
                    "target_tokens": LossInputLayout.PER_TOKEN,
                },
                loss_fn_input_pad_values={"routed_experts": -1},
            )
        )
    model = TokenModel()
    model.backend = SimpleNamespace(attn="te")
    seen = {}

    def compute_output(output, inputs):
        """Return physical token values while recording packed side shapes.

        Args:
            output: Packed model values of shape ``[tokens]``.
            inputs: Side tensors with targets ``[tokens]`` and routed experts
                ``[tokens, layers, top_k]``.

        Returns:
            Model token values of shape ``[tokens]``.
        """
        seen["routes"] = inputs["routed_experts"].shape
        seen["targets"] = inputs["target_tokens"].shape
        return output

    outputs = Engine(
        model,
        device="cpu",
        collate_fn=partial(collate_vlm_datums, processor=processor, packed=True),
    ).forward(datums, compute_output)

    assert seen == {"routes": torch.Size([6, 2, 1]), "targets": torch.Size([6])}
    torch.testing.assert_close(outputs[0], torch.tensor([2.0, 198.0, 4.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([10.0, 198.0, 12.0]))


def test_forward_uses_model_tokens_as_anchor_when_routes_are_the_only_side_input() -> None:
    routes = torch.arange(12, dtype=torch.int16).reshape(3, 2, 2)
    datum = Datum(
        input_ids=[1, 2, 3],
        loss_fn_inputs={"routed_experts": routes},
        loss_fn_input_layouts={"routed_experts": LossInputLayout.PER_TOKEN},
        loss_fn_input_pad_values={"routed_experts": -1},
    )

    outputs = Engine(TokenModel(), device="cpu").forward(
        [datum],
        lambda output, inputs: output + inputs["routed_experts"][..., 0, 0],
    )

    torch.testing.assert_close(outputs[0], torch.tensor([2.0, 8.0, 14.0]))


def test_forward_custom_collater_needs_no_weights() -> None:
    datum = Datum(input_ids=[1, 2, 3])

    def collate(datums):
        """Return padded ``[batch, sequence]`` model and task tensors."""
        assert datums == [datum]
        return {"input_ids": torch.tensor([[1, 2, 3]])}, {"bias": torch.ones(1, 3)}

    outputs = Engine(TokenModel(), device="cpu", collate_fn=collate).forward(
        [datum],
        lambda output, inputs: output + inputs["bias"],
    )

    torch.testing.assert_close(outputs[0], torch.tensor([3.0, 5.0, 7.0]))


@pytest.mark.parametrize("axis", ["context-parallel", "pipeline"])
def test_forward_rejects_inconsistent_model_parallel_datum_layouts(monkeypatch, axis) -> None:
    engine = Engine(TokenModel(), device="cpu")
    group = object()
    if axis == "context-parallel":
        engine._cp_group_and_size = lambda: (group, 2)
    else:
        engine._pp_group_and_size = lambda: (group, 2)

    def gather_layouts(gathered, local, *, group):
        """Report one peer with a different scoring layout digest."""
        gathered.copy_(torch.cat((local, local)))
        gathered[-1] += 1

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather_layouts)

    with pytest.raises(ValueError, match=rf"{axis} ranks must use the same scoring Datum"):
        engine.forward([Datum(input_ids=[1, 2])], lambda output, _inputs: output)


def test_forward_backward_preserves_explicit_batch_boundaries() -> None:
    model = TokenModel()
    engine = Engine(model, device="cpu")
    batches = [
        [
            Datum(
                input_ids=[1, 2],
                loss_fn_inputs={"weights": torch.ones(2)},
                loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
            ),
            Datum(
                input_ids=[3],
                loss_fn_inputs={"weights": torch.ones(1)},
                loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
            ),
        ],
        [
            Datum(
                input_ids=[4],
                loss_fn_inputs={"weights": torch.ones(1)},
                loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
            )
        ],
    ]

    def loss(output, inputs):
        """Return a scalar numerator and one physical token stream.

        Args:
            output: Padded model values of shape ``[batch, sequence]``.
            inputs: Loss tensors whose weights have shape
                ``[batch, sequence]``.

        Returns:
            A scalar weighted numerator and token output matching ``output``.
        """
        return LossOutput(loss_sum=(output * inputs["weights"]).sum(), token_outputs={"scaled": output})

    result = engine.forward_backward(batches, loss)

    assert [len(batch["scaled"]) for batch in result.token_outputs] == [2, 1]
    assert result.batch_outputs == [None, None]
    torch.testing.assert_close(result.token_outputs[0]["scaled"][0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(result.token_outputs[1]["scaled"][0], torch.tensor([8.0]))


def test_evaluate_keeps_batch_outputs_separate_from_token_outputs() -> None:
    datum = Datum(input_ids=[1, 2], loss_fn_inputs={"weights": torch.ones(2)})

    result = Engine(TokenModel(), device="cpu").evaluate(
        [[datum]],
        lambda output, inputs: LossOutput(
            loss_sum=(output * inputs["weights"]).sum(),
            batch_output={"metric": output.mean()},
        ),
    )

    assert result.token_outputs == [{}]
    torch.testing.assert_close(result.batch_outputs[0]["metric"], torch.tensor(3.0))
    assert not result.batch_outputs[0]["metric"].requires_grad


def test_forward_backward_rejects_a_flat_datum_sequence() -> None:
    datum = Datum(input_ids=[1], loss_fn_inputs={"weights": torch.ones(1)})

    with pytest.raises((TypeError, ValueError), match="Datum"):
        Engine(TokenModel(), device="cpu").forward_backward([datum], lambda output, _inputs: output.sum())
