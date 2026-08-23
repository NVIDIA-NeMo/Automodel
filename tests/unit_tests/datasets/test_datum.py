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

import pickle
from copy import copy, deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.datasets.datum import (
    CROSS_ENTROPY_IGNORE_IDX,
    CollatedLossInputs,
    Datum,
    LossInputLayout,
    collate_datums,
    collate_vlm_datums,
)


def _toy_datums():
    return [
        Datum(
            model_inputs={"input_ids": torch.tensor([10, 11, 12])},
            loss_fn_inputs={
                "target_tokens": torch.tensor([11, 12, 13]),
                "weights": torch.tensor([1.0, 1.0, 0.0]),
                "advantages": torch.tensor([0.5, 0.5, 0.5]),
            },
        ),
        Datum(
            model_inputs={"input_ids": torch.tensor([20, 21])},
            loss_fn_inputs={
                "target_tokens": torch.tensor([21, 22]),
                "weights": torch.tensor([1.0, 1.0]),
                "advantages": torch.tensor([0.9, 0.9]),
            },
        ),
    ]


def _routed_datums():
    """Build ragged Datums with per-token routing metadata.

    Returns:
        Two Datums whose routed-expert tensors use the
        ``[tokens, layers, topk]`` layout and ``torch.int16`` dtype.
    """
    first_routes = torch.arange(3 * 2 * 2, dtype=torch.int16).reshape(3, 2, 2)
    second_routes = torch.arange(100, 100 + 2 * 2 * 2, dtype=torch.int16).reshape(2, 2, 2)
    return [
        Datum(
            input_ids=torch.tensor([10, 11, 12]),
            loss_fn_inputs={"weights": torch.ones(3), "routed_experts": first_routes},
            loss_fn_input_layouts={"routed_experts": LossInputLayout.PER_TOKEN},
            loss_fn_input_pad_values={"routed_experts": -1},
        ),
        Datum(
            input_ids=torch.tensor([20, 21]),
            loss_fn_inputs={"weights": torch.ones(2), "routed_experts": second_routes},
            loss_fn_input_layouts={"routed_experts": LossInputLayout.PER_TOKEN},
            loss_fn_input_pad_values={"routed_experts": -1},
        ),
    ]


def _vlm_datum(input_ids, weights, **media_inputs):
    """Build one unshifted VLM item for collater tests.

    Args:
        input_ids: Token IDs with shape ``[sequence]``.
        weights: Boolean supervision values with shape ``[sequence]``.
        **media_inputs: Processor tensors with shape ``[media, ...]``.

    Returns:
        A Datum whose token fields have shape ``[sequence]`` and whose media
        fields preserve their input shapes.
    """
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.bool)
    labels = input_ids.clone().masked_fill(~weights, CROSS_ENTROPY_IGNORE_IDX)
    return Datum(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            **media_inputs,
        },
        loss_fn_inputs={"labels": labels, "weights": weights},
        loss_fn_input_layouts={
            "labels": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"labels": CROSS_ENTROPY_IGNORE_IDX},
    )


def _vlm_rl_datums():
    """Build ragged VLM Datums with prediction-aligned RL side channels.

    Returns:
        Two Datums whose target scores use the unshifted
        ``[sequence, objectives]`` axis, routed experts use the already-shifted
        ``[sequence - 1, layers, topk]`` axis, and scalar metadata uses
        per-Datum or replicated layouts.
    """
    datums = [
        _vlm_datum([1, 2, 3, 4], [0, 1, 1, 1]),
        _vlm_datum([5, 6, 7], [0, 1, 1]),
    ]
    target_scores = [
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
        torch.arange(20, 26, dtype=torch.float32).reshape(3, 2),
    ]
    routed_experts = [
        torch.arange(12, dtype=torch.int16).reshape(3, 2, 2),
        torch.arange(100, 108, dtype=torch.int16).reshape(2, 2, 2),
    ]
    sequence_scores = [torch.tensor(0.25), torch.tensor(0.75)]
    sampling_policy = torch.tensor([0.9, 0.1])
    for datum, scores, routes, sequence_score in zip(
        datums,
        target_scores,
        routed_experts,
        sequence_scores,
    ):
        datum.loss_fn_inputs.update(
            {
                "target_scores": scores,
                "routed_experts": routes,
                "sequence_score": sequence_score,
                "sampling_policy": sampling_policy.clone(),
            }
        )
        datum.loss_fn_input_layouts.update(
            {
                "target_scores": LossInputLayout.PER_TOKEN,
                "routed_experts": LossInputLayout.PER_TOKEN,
                "sequence_score": LossInputLayout.PER_DATUM,
                "sampling_policy": LossInputLayout.REPLICATED,
            }
        )
        datum.loss_fn_input_pad_values.update(
            {
                "target_scores": -7.5,
                "routed_experts": -1,
            }
        )
    return datums


# ── Datum ─────────────────────────────────────────────────────────────────


def _clone_instead_of_pinning(tensor: torch.Tensor, _device: str | None = None) -> torch.Tensor:
    """Return an observable CPU-safe stand-in for ``Tensor.pin_memory``.

    Args:
        tensor: Tensor of arbitrary shape and dtype.
        _device: Optional accelerator identifier accepted by PyTorch's pinning
            dispatcher. It does not affect this CPU-safe test double.

    Returns:
        Tensor of the same shape and dtype with independent storage.
    """
    return tensor.clone()


def test_datum_keeps_old_input_ids_convenience():
    d = Datum(input_ids=[1, 2, 3], loss_fn_inputs={"weights": [1, 1, 0]})
    assert isinstance(d.input_ids, torch.Tensor)
    assert d.input_ids.dtype == torch.long
    assert d.seq_len == 3
    assert isinstance(d.loss_fn_inputs["weights"], torch.Tensor)

    positional = Datum([4, 5], {"weights": [1, 0]})
    assert positional.input_ids.tolist() == [4, 5]


def test_datum_allows_prebatched_inputs_only_with_a_custom_collater():
    datum = Datum(model_inputs={"input_ids": torch.zeros(2, 3, dtype=torch.long)})
    assert datum.model_inputs["input_ids"].shape == (2, 3)
    with pytest.raises(ValueError, match="default collater requires 1-D"):
        collate_datums([datum])


def test_datum_accepts_model_specific_inputs():
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([1, 2]),
            "pixel_values": torch.randn(1, 3, 4, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        loss_fn_inputs={"weights": torch.ones(2)},
    )
    assert datum.model_inputs["pixel_values"].shape == (1, 3, 4, 4)
    assert datum.seq_len == 2


def test_datum_pin_memory_handles_molt_prebatched_inputs(monkeypatch):
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[10, 11, 12], [20, 21, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 1]]),
        },
        loss_fn_inputs={
            "labels": torch.tensor([[11, 12, -100], [21, -100, -100]]),
            "weights": torch.tensor([[True, True, False], [True, False, False]]),
        },
        loss_fn_input_layouts={
            "labels": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"labels": -100},
    )
    original_model_tensors = dict(datum.model_inputs)
    original_loss_tensors = dict(datum.loss_fn_inputs)
    original_layouts = datum.loss_fn_input_layouts
    original_pad_values = datum.loss_fn_input_pad_values
    monkeypatch.setattr(torch.Tensor, "pin_memory", _clone_instead_of_pinning)

    result = datum.pin_memory()

    assert result is datum
    for key, original in original_model_tensors.items():
        assert datum.model_inputs[key] is not original
        torch.testing.assert_close(datum.model_inputs[key], original)
    for key, original in original_loss_tensors.items():
        assert datum.loss_fn_inputs[key] is not original
        torch.testing.assert_close(datum.loss_fn_inputs[key], original)
    assert datum.loss_fn_input_layouts is original_layouts
    assert datum.loss_fn_input_layouts == {
        "labels": LossInputLayout.PER_TOKEN,
        "weights": LossInputLayout.PER_TOKEN,
    }
    assert datum.loss_fn_input_pad_values is original_pad_values
    assert datum.loss_fn_input_pad_values == {"labels": -100}


def test_datum_pin_memory_uses_dataloader_recursion_for_model_inputs(monkeypatch):
    custom_leaf = Mock()
    pinned_custom_leaf = object()
    custom_leaf.pin_memory.return_value = pinned_custom_leaf
    image = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    grid = torch.tensor([[1, 2, 3]])
    auxiliary = torch.tensor([4.0, 5.0])
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "media": {
                "images": [image, None],
                "details": (grid, {"auxiliary": auxiliary}),
                "custom": custom_leaf,
            },
            "batch_size": 1,
            "qkv_format": "bshd",
        },
        loss_fn_inputs={"weights": torch.ones(1, 3)},
        loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", _clone_instead_of_pinning)

    result = datum.pin_memory()

    assert result is datum
    assert isinstance(datum.model_inputs["media"], dict)
    assert isinstance(datum.model_inputs["media"]["images"], list)
    assert isinstance(datum.model_inputs["media"]["details"], tuple)
    pinned_image = datum.model_inputs["media"]["images"][0]
    pinned_grid = datum.model_inputs["media"]["details"][0]
    pinned_auxiliary = datum.model_inputs["media"]["details"][1]["auxiliary"]
    for pinned, original in ((pinned_image, image), (pinned_grid, grid), (pinned_auxiliary, auxiliary)):
        assert pinned is not original
        torch.testing.assert_close(pinned, original)
    assert datum.model_inputs["media"]["images"][1] is None
    assert datum.model_inputs["batch_size"] == 1
    assert datum.model_inputs["qkv_format"] == "bshd"
    assert datum.model_inputs["media"]["custom"] is pinned_custom_leaf
    custom_leaf.pin_memory.assert_called_once_with()


def test_datum_pin_memory_does_not_partially_commit_on_failure(monkeypatch):
    model_tensor = torch.tensor([[1, 2, 3]])
    loss_tensor = torch.ones(1, 3)
    datum = Datum(
        model_inputs={"input_ids": model_tensor, "metadata": {"source": "molt"}},
        loss_fn_inputs={"weights": loss_tensor},
        loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
    )
    original_model_inputs = datum.model_inputs
    original_loss_fn_inputs = datum.loss_fn_inputs
    original_layouts = datum.loss_fn_input_layouts

    def fail_for_loss_tensor(tensor: torch.Tensor, _device: str | None = None) -> torch.Tensor:
        """Fail after model-input pinning reaches the loss-input mapping.

        Args:
            tensor: Tensor of arbitrary shape and dtype.
            _device: Optional accelerator identifier accepted by PyTorch's
                pinning dispatcher.

        Returns:
            An independent tensor when ``tensor`` is a model input.

        Raises:
            RuntimeError: If ``tensor`` is the selected loss input.
        """
        if tensor is loss_tensor:
            raise RuntimeError("injected pin-memory failure")
        return tensor.clone()

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_for_loss_tensor)

    with pytest.raises(RuntimeError, match="injected pin-memory failure"):
        datum.pin_memory()

    assert datum.model_inputs is original_model_inputs
    assert datum.loss_fn_inputs is original_loss_fn_inputs
    assert datum.model_inputs["input_ids"] is model_tensor
    assert datum.model_inputs["metadata"] == {"source": "molt"}
    assert datum.loss_fn_inputs["weights"] is loss_tensor
    assert datum.loss_fn_input_layouts is original_layouts
    assert datum.loss_fn_input_layouts == {"weights": LossInputLayout.PER_TOKEN}


def test_datum_accepts_optional_loss_input_layouts():
    datum = Datum(
        input_ids=torch.tensor([1, 2]),
        loss_fn_inputs={"weights": torch.ones(2)},
        loss_fn_input_layouts={"weights": LossInputLayout.PER_TOKEN},
    )
    assert datum.loss_fn_input_layouts == {"weights": LossInputLayout.PER_TOKEN}


def test_datum_rejects_invalid_loss_input_layouts():
    with pytest.raises(ValueError, match="unknown loss inputs"):
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"weights": torch.ones(1)},
            loss_fn_input_layouts={"missing": LossInputLayout.PER_TOKEN},
        )
    with pytest.raises(TypeError, match="must be a LossInputLayout"):
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"weights": torch.ones(1)},
            loss_fn_input_layouts={"weights": "per_token"},  # type: ignore[dict-item]
        )


def test_datum_validates_loss_input_pad_value_metadata():
    with pytest.raises(ValueError, match="unknown loss inputs"):
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"weights": torch.ones(1)},
            loss_fn_input_pad_values={"missing": -1},
        )
    with pytest.raises(TypeError, match="bool, int, or float"):
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"weights": torch.ones(1)},
            loss_fn_input_pad_values={"weights": object()},  # type: ignore[dict-item]
        )


def test_to_features_applies_masking_convention():
    feats = _toy_datums()[0].to_features()
    assert feats["input_ids"] == [10, 11, 12]
    # weights==0 at position 2 -> ignore_index in labels.
    assert feats["labels"] == [11, 12, CROSS_ENTROPY_IGNORE_IDX]


def test_to_features_omits_labels_without_targets():
    feats = Datum(input_ids=torch.tensor([1, 2, 3])).to_features()
    assert feats == {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_to_features_native_python_ints():
    # Collaters call torch.LongTensor(...) on these, so they must be plain ints.
    feats = _toy_datums()[0].to_features()
    assert all(isinstance(x, int) for x in feats["input_ids"])
    assert all(isinstance(x, int) for x in feats["labels"])


# ── collate_datums delegates to the canonical collaters ─────────────────────


def test_collate_vlm_datums_delegates_padding_and_preserves_processor_fields():
    processor = SimpleNamespace(
        image_token_id=99,
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=9),
    )
    datums = [
        _vlm_datum(
            [1, 99, 3],
            [0, 1, 1],
            pixel_values=torch.tensor([[1.0, 2.0]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            image_flags=torch.tensor([[1]]),
            imgs_sizes=torch.tensor([[32, 64]]),
        ),
        _vlm_datum([4, 5], [0, 1]),
    ]

    model_inputs, loss_inputs = collate_vlm_datums(datums, processor=processor)

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[1, 99], [4, 5]]))
    torch.testing.assert_close(model_inputs["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    torch.testing.assert_close(loss_inputs["labels"], torch.tensor([[99, 3], [5, -100]]))
    torch.testing.assert_close(loss_inputs["weights"], torch.tensor([[True, True], [True, False]]))
    torch.testing.assert_close(model_inputs["image_flags"], torch.tensor([[1]]))
    torch.testing.assert_close(model_inputs["imgs_sizes"], torch.tensor([[32, 64]]))
    assert model_inputs["pixel_values"].dtype == torch.bfloat16
    assert loss_inputs.item_to_datum == (0, 1)
    assert loss_inputs.layouts == {
        "labels": LossInputLayout.PER_TOKEN,
        "weights": LossInputLayout.PER_TOKEN,
    }
    assert loss_inputs.pad_values == {"labels": CROSS_ENTROPY_IGNORE_IDX}


@pytest.mark.parametrize("packed", [False, True])
def test_collate_vlm_datums_pads_variable_resolution_pixel_tensors(packed):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    first_pixels = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    second_pixels = torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2) + 100
    datums = [
        _vlm_datum([1, 2, 3], [0, 1, 1], pixel_values=first_pixels),
        _vlm_datum([4, 5, 6], [0, 1, 1], pixel_values=second_pixels),
    ]

    model_inputs, _ = collate_vlm_datums(datums, processor=processor, packed=packed)

    pixel_values = model_inputs["pixel_values"]
    assert pixel_values.shape == (3, 3, 4, 4)
    assert pixel_values.dtype == torch.bfloat16
    torch.testing.assert_close(pixel_values[0, :, :2, :], first_pixels[0].to(torch.bfloat16))
    assert not bool(pixel_values[0, :, 2:, :].any())
    torch.testing.assert_close(pixel_values[1:, :, :, :2], second_pixels.to(torch.bfloat16))
    assert not bool(pixel_values[1:, :, :, 2:].any())


def test_collate_vlm_datums_indexed_mask_preserves_media_and_rl_side_channels():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = _vlm_rl_datums()
    first_pixels = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    second_pixels = torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2) + 100
    datums[0].model_inputs["pixel_values"] = first_pixels
    datums[1].model_inputs["pixel_values"] = second_pixels

    model_inputs, loss_inputs = collate_vlm_datums(
        datums,
        processor=processor,
        packing_layout="indexed_mask",
    )

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[1, 2, 3, 5, 6]]))
    torch.testing.assert_close(model_inputs["attention_mask"], torch.tensor([[1, 1, 1, 2, 2]]))
    torch.testing.assert_close(model_inputs["_packed_seq_ids"], model_inputs["attention_mask"])
    assert "_engine_packing_layout" not in model_inputs
    assert loss_inputs._engine_packing_layout == "indexed_mask"
    torch.testing.assert_close(model_inputs["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
    assert "qkv_format" not in model_inputs
    assert model_inputs["pixel_values"].shape == (3, 3, 4, 4)
    torch.testing.assert_close(model_inputs["pixel_values"][0, :, :2, :], first_pixels[0].to(torch.bfloat16))
    torch.testing.assert_close(model_inputs["pixel_values"][1:, :, :, :2], second_pixels.to(torch.bfloat16))
    torch.testing.assert_close(
        loss_inputs["target_scores"],
        torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [22.0, 23.0], [24.0, 25.0]]]),
    )
    torch.testing.assert_close(
        loss_inputs["routed_experts"][0],
        torch.cat([datum.loss_fn_inputs["routed_experts"] for datum in datums]),
    )
    torch.testing.assert_close(loss_inputs["sequence_score"], torch.tensor([0.25, 0.75]))
    assert loss_inputs.item_to_datum == (0, 1)


def test_collate_vlm_datums_indexed_mask_rejects_thd_alignment():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    with pytest.raises(ValueError, match="indexed_mask.*sequence_alignment=1"):
        collate_vlm_datums(
            _vlm_rl_datums(),
            processor=processor,
            packing_layout="indexed_mask",
            sequence_alignment=2,
        )


def test_collate_vlm_datums_single_indexed_document_keeps_explicit_layout_metadata():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    model_inputs, loss_inputs = collate_vlm_datums(
        [_vlm_datum([1, 2, 3], [0, 1, 1])],
        processor=processor,
        packing_layout="indexed_mask",
    )

    torch.testing.assert_close(model_inputs["_packed_seq_ids"], torch.ones(1, 2, dtype=torch.long))
    assert "_engine_packing_layout" not in model_inputs
    assert loss_inputs._engine_packing_layout == "indexed_mask"


def test_collate_vlm_datums_pads_s_and_s_minus_one_side_channels_with_layouts():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = _vlm_rl_datums()

    _, loss_inputs = collate_vlm_datums(datums, processor=processor)

    torch.testing.assert_close(
        loss_inputs["target_scores"],
        torch.tensor(
            [
                [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                [[22.0, 23.0], [24.0, 25.0], [-7.5, -7.5]],
            ]
        ),
    )
    assert loss_inputs["routed_experts"].shape == (2, 3, 2, 2)
    torch.testing.assert_close(loss_inputs["routed_experts"][0], datums[0].loss_fn_inputs["routed_experts"])
    torch.testing.assert_close(loss_inputs["routed_experts"][1, :2], datums[1].loss_fn_inputs["routed_experts"])
    assert torch.equal(loss_inputs["routed_experts"][1, 2], torch.full((2, 2), -1, dtype=torch.int16))
    torch.testing.assert_close(loss_inputs["sequence_score"], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(loss_inputs["sampling_policy"], torch.tensor([0.9, 0.1]))
    assert loss_inputs.layouts == {
        "labels": LossInputLayout.PER_TOKEN,
        "routed_experts": LossInputLayout.PER_TOKEN,
        "sampling_policy": LossInputLayout.REPLICATED,
        "sequence_score": LossInputLayout.PER_DATUM,
        "target_scores": LossInputLayout.PER_TOKEN,
        "weights": LossInputLayout.PER_TOKEN,
    }
    assert loss_inputs.pad_values == {
        "labels": CROSS_ENTROPY_IGNORE_IDX,
        "routed_experts": -1,
        "target_scores": -7.5,
    }
    assert loss_inputs.item_to_datum == (0, 1)


def test_collate_vlm_datums_keeps_prediction_aligned_s_minus_one_weights():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = [
        _vlm_datum([1, 2, 3, 4], [0, 1, 1, 1]),
        _vlm_datum([5, 6, 7], [0, 1, 1]),
    ]
    datums[0].loss_fn_inputs["weights"] = torch.tensor([True, False, True])
    datums[1].loss_fn_inputs["weights"] = torch.tensor([True, True])

    _, loss_inputs = collate_vlm_datums(datums, processor=processor)

    torch.testing.assert_close(
        loss_inputs["weights"],
        torch.tensor([[True, False, True], [True, True, False]]),
    )
    assert int(loss_inputs["weights"].sum()) == 4


def test_collate_vlm_datums_packs_side_channels_with_each_document_alignment():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=9))
    datums = _vlm_rl_datums()

    model_inputs, loss_inputs = collate_vlm_datums(
        datums,
        processor=processor,
        packed=True,
        sequence_alignment=4,
    )

    torch.testing.assert_close(model_inputs["seq_lens"], torch.tensor([[3, 2]]))
    torch.testing.assert_close(model_inputs["seq_lens_padded"], torch.tensor([[4, 4]]))
    torch.testing.assert_close(
        loss_inputs["target_scores"],
        torch.tensor(
            [
                [
                    [2.0, 3.0],
                    [4.0, 5.0],
                    [6.0, 7.0],
                    [-7.5, -7.5],
                    [22.0, 23.0],
                    [24.0, 25.0],
                    [-7.5, -7.5],
                    [-7.5, -7.5],
                ]
            ]
        ),
    )
    expected_routes = torch.cat(
        [
            torch.cat([datums[0].loss_fn_inputs["routed_experts"], torch.full((1, 2, 2), -1, dtype=torch.int16)]),
            torch.cat([datums[1].loss_fn_inputs["routed_experts"], torch.full((2, 2, 2), -1, dtype=torch.int16)]),
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(loss_inputs["routed_experts"], expected_routes)
    torch.testing.assert_close(loss_inputs["sequence_score"], torch.tensor([0.25, 0.75]))
    torch.testing.assert_close(loss_inputs["sampling_policy"], torch.tensor([0.9, 0.1]))
    assert loss_inputs.item_to_datum == (0, 1)


@pytest.mark.parametrize("field", ["mm_token_type_ids", "token_type_ids"])
@pytest.mark.parametrize("packed", [False, True])
def test_collate_vlm_datums_aligns_token_type_model_inputs(field, packed):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=9))
    datums = [
        _vlm_datum([1, 2, 3, 4], [0, 1, 1, 1], **{field: torch.tensor([0, 1, 2, 0])}),
        _vlm_datum([5, 6, 7], [0, 1, 1], **{field: torch.tensor([2, 1, 0])}),
    ]

    model_inputs, _ = collate_vlm_datums(
        datums,
        processor=processor,
        packed=packed,
        sequence_alignment=4 if packed else 1,
    )

    if packed:
        expected = torch.tensor([[0, 1, 2, 0, 2, 1, 0, 0]])
    else:
        expected = torch.tensor([[0, 1, 2], [2, 1, 0]])
    torch.testing.assert_close(model_inputs[field], expected)
    assert model_inputs[field].shape == model_inputs["input_ids"].shape


@pytest.mark.parametrize("layout", ["padded", "thd", "indexed"])
def test_collate_vlm_datums_infers_media_token_types_for_forward_datums(layout):
    processor = SimpleNamespace(
        image_token_id=99,
        video_token_id=88,
        image_processor=SimpleNamespace(merge_size=2),
        video_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    datum = Datum(
        model_inputs={
            "input_ids": torch.tensor([1, 99, 88, 2]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )

    model_inputs, loss_inputs = collate_vlm_datums(
        [datum],
        processor=processor,
        packed=layout == "thd",
        packing_layout="indexed_mask" if layout == "indexed" else None,
    )

    torch.testing.assert_close(model_inputs["mm_token_type_ids"], torch.tensor([[0, 1, 2]]))
    assert model_inputs["mm_token_type_ids"].shape == model_inputs["input_ids"].shape
    assert loss_inputs == {}


def test_collate_vlm_datums_rejects_invalid_unshifted_contract():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    with pytest.raises(ValueError, match="at least one"):
        collate_vlm_datums([], processor=processor)
    with pytest.raises(ValueError, match="target position zero"):
        collate_vlm_datums([_vlm_datum([1, 2], [1, 1])], processor=processor)


def test_collate_vlm_datums_rejects_per_token_side_channel_outside_s_or_s_minus_one():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datum = _vlm_datum([1, 2, 3, 4], [0, 1, 1, 1])
    datum.loss_fn_inputs["advantages"] = torch.ones(2)
    datum.loss_fn_input_layouts["advantages"] = LossInputLayout.PER_TOKEN

    with pytest.raises(ValueError, match=r"advantages.*S or S-1"):
        collate_vlm_datums([datum], processor=processor)


@pytest.mark.parametrize("field", ["labels", "weights"])
def test_collate_vlm_datums_requires_token_layout_for_canonical_loss_fields(field):
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datum = _vlm_datum([1, 2, 3], [0, 1, 1])
    datum.loss_fn_input_layouts[field] = LossInputLayout.REPLICATED
    if field == "labels":
        datum.loss_fn_input_pad_values.pop("labels")

    with pytest.raises(ValueError, match=rf"VLM {field} must use the PER_TOKEN layout"):
        collate_vlm_datums([datum], processor=processor)


def test_collate_vlm_datums_rejects_mixed_s_and_s_minus_one_conventions_for_one_field():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = _vlm_rl_datums()
    datums[1].loss_fn_inputs["target_scores"] = datums[1].loss_fn_inputs["target_scores"][1:]

    with pytest.raises(ValueError, match="target_scores"):
        collate_vlm_datums(datums, processor=processor)


def test_collate_vlm_datums_rejects_explicit_padding_in_packed_input():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datum = _vlm_datum([1, 2, 0], [0, 1, 0])
    datum.model_inputs["attention_mask"][-1] = 0

    with pytest.raises(ValueError, match=r"packed VLM Datums.*attention_mask padding"):
        collate_vlm_datums([datum], processor=processor, packed=True)


@pytest.mark.parametrize("packed", [False, True])
def test_collate_vlm_datums_rejects_resolvable_media_token_mismatch(packed):
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_token_id=99,
        image_processor=SimpleNamespace(merge_size=2),
    )
    datum = _vlm_datum(
        [1, 99, 2],
        [0, 1, 1],
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        pixel_values=torch.ones(4, 2),
    )

    with pytest.raises(ValueError, match=r"media token mismatch.*image tokens=1, expected=4"):
        collate_vlm_datums([datum], processor=processor, packed=packed)


@pytest.mark.parametrize("packed", [False, True])
def test_collate_vlm_datums_rejects_media_removed_by_autoregressive_shift(packed):
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_token_id=99,
        image_processor=SimpleNamespace(merge_size=2),
    )
    datum = _vlm_datum(
        [1, 2, 99],
        [0, 1, 1],
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        pixel_values=torch.ones(1, 2),
    )

    with pytest.raises(ValueError, match="media token mismatch"):
        collate_vlm_datums([datum], processor=processor, packed=packed)


def test_collate_vlm_datums_rejects_unconcatenable_processor_media_field():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = [
        _vlm_datum([1, 2, 3], [0, 1, 1], auxiliary_media=torch.ones(1, 2)),
        _vlm_datum([4, 5, 6], [0, 1, 1], auxiliary_media=torch.ones(1, 3)),
    ]

    with pytest.raises(ValueError, match=r"processor field 'auxiliary_media'.*cannot be concatenated"):
        collate_vlm_datums(datums, processor=processor)


def test_collate_vlm_datums_packs_aligned_thd_documents():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=9))
    datums = [
        _vlm_datum(
            [1, 2, 3],
            [0, 1, 1],
            pixel_values=torch.tensor([[1.0, 2.0]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            image_flags=torch.tensor([[1]]),
        ),
        _vlm_datum([4, 5], [0, 1]),
    ]

    model_inputs, loss_inputs = collate_vlm_datums(
        datums,
        processor=processor,
        packed=True,
        sequence_alignment=4,
    )

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[1, 2, 9, 9, 4, 9, 9, 9]]))
    torch.testing.assert_close(model_inputs["position_ids"], torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))
    torch.testing.assert_close(model_inputs["seq_lens"], torch.tensor([[2, 1]]))
    torch.testing.assert_close(model_inputs["seq_lens_padded"], torch.tensor([[4, 4]]))
    torch.testing.assert_close(loss_inputs["labels"], torch.tensor([[2, 3, -100, -100, 5, -100, -100, -100]]))
    torch.testing.assert_close(
        loss_inputs["weights"],
        torch.tensor([[True, True, False, False, True, False, False, False]]),
    )
    torch.testing.assert_close(model_inputs["image_flags"], torch.tensor([[1]]))
    assert model_inputs["qkv_format"] == "thd"
    assert loss_inputs.item_to_datum == (0, 1)


def test_collate_vlm_datums_packs_mrope_only_without_cp_alignment():
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    datums = [_vlm_datum([1, 2, 3], [0, 1, 1]), _vlm_datum([4, 5], [0, 1])]

    def get_rope_index(input_ids, **_kwargs):
        sequence = input_ids.shape[-1]
        positions = torch.arange(sequence).expand(3, 1, sequence)
        return positions, torch.zeros(1)

    model_inputs, _ = collate_vlm_datums(
        datums,
        processor=processor,
        packed=True,
        get_rope_index=get_rope_index,
    )

    assert model_inputs["position_ids"].shape == (3, 1, 3)
    with pytest.raises(NotImplementedError, match="multi-axis mRoPE"):
        collate_vlm_datums(
            datums,
            processor=processor,
            packed=True,
            get_rope_index=get_rope_index,
            sequence_alignment=2,
        )


def test_collate_padded_uses_default_collater_schema():
    batch, loss_inputs = collate_datums(_toy_datums())
    assert isinstance(loss_inputs, CollatedLossInputs)
    assert loss_inputs.layouts == {
        "advantages": LossInputLayout.PER_TOKEN,
        "target_tokens": LossInputLayout.PER_TOKEN,
        "weights": LossInputLayout.PER_TOKEN,
    }
    assert loss_inputs.item_to_datum == (0, 1)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [20, 21, 0]  # right-pad
    assert "labels" not in batch
    assert loss_inputs["target_tokens"][0].tolist() == [11, 12, 13]
    assert loss_inputs["target_tokens"][1].tolist() == [21, 22, 0]
    # padding_mask is produced by default_collater, not by us.
    assert "padding_mask" in batch


def test_collate_packed_concatenates_into_one_thd_row():
    batch, loss_inputs = collate_datums(_toy_datums(), packed=True)
    # All datums share one flat [1, total_tokens] pack.
    assert batch["qkv_format"] == "thd"
    assert batch["input_ids"].shape == (1, 5)
    assert batch["input_ids"][0].tolist() == [10, 11, 12, 20, 21]
    assert "labels" not in batch
    assert loss_inputs["target_tokens"][0].tolist() == [11, 12, 13, 21, 22]
    # position_ids restart at every sequence boundary (RoPE resets).
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1]
    assert batch["seq_lens"][0].tolist() == [3, 2]
    assert batch["seq_lens_padded"][0].tolist() == [3, 2]


def test_collate_indexed_mask_uses_one_row_with_document_ids_and_reset_positions():
    batch, loss_inputs = collate_datums(_toy_datums(), packing_layout="indexed_mask")

    torch.testing.assert_close(batch["input_ids"], torch.tensor([[10, 11, 12, 20, 21]]))
    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1, 1, 2, 2]]))
    torch.testing.assert_close(batch["_packed_seq_ids"], batch["attention_mask"])
    assert "_engine_packing_layout" not in batch
    assert loss_inputs._engine_packing_layout == "indexed_mask"
    torch.testing.assert_close(batch["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
    assert "qkv_format" not in batch
    assert "seq_lens" not in batch
    torch.testing.assert_close(loss_inputs["weights"], torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]]))
    torch.testing.assert_close(loss_inputs["advantages"], torch.tensor([[0.5, 0.5, 0.5, 0.9, 0.9]]))
    assert loss_inputs.item_to_datum == (0, 1)


def test_collate_single_indexed_document_keeps_explicit_layout_metadata():
    batch, loss_inputs = collate_datums([_toy_datums()[0]], packing_layout="indexed_mask")

    torch.testing.assert_close(batch["_packed_seq_ids"], torch.ones(1, 3, dtype=torch.long))
    assert "_engine_packing_layout" not in batch
    assert loss_inputs._engine_packing_layout == "indexed_mask"


def test_collate_indexed_mask_rejects_conflicting_or_padded_layout_options():
    datums = _toy_datums()

    with pytest.raises(ValueError, match="packed and packing_layout"):
        collate_datums(datums, packed=True, packing_layout="indexed_mask")
    with pytest.raises(ValueError, match="pad_seq_len_divisible"):
        collate_datums(datums, packing_layout="indexed_mask", pad_seq_len_divisible=8)


def test_collate_packed_side_inputs_ride_the_flat_axis():
    batch, loss_inputs = collate_datums(_toy_datums(), packed=True)
    # Per-token floats are concatenated in datum order, aligned with input_ids.
    assert set(batch).isdisjoint({"advantages", "weights", "target_tokens"})
    assert loss_inputs["advantages"].shape == (1, 5)
    assert loss_inputs["advantages"][0].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.9, 0.9])
    assert loss_inputs["weights"][0].tolist() == pytest.approx([1.0, 1.0, 0.0, 1.0, 1.0])
    # seq_lens allows splitting flat outputs back per datum.
    lens = [n for n in batch["seq_lens"][0].tolist() if n > 0]
    split = torch.split(loss_inputs["advantages"][0], lens)
    assert [t.tolist() for t in split] == [pytest.approx([0.5, 0.5, 0.5]), pytest.approx([0.9, 0.9])]


def test_collate_padded_per_token_trailing_dims_use_the_declared_pad_value():
    datums = _routed_datums()

    _, loss_inputs = collate_datums(datums)

    routes = loss_inputs["routed_experts"]
    assert routes.shape == (2, 3, 2, 2)
    assert routes.dtype == torch.int16
    torch.testing.assert_close(routes[0], datums[0].loss_fn_inputs["routed_experts"])
    torch.testing.assert_close(routes[1, :2], datums[1].loss_fn_inputs["routed_experts"])
    assert torch.equal(routes[1, 2], torch.full((2, 2), -1, dtype=torch.int16))
    assert loss_inputs.layouts["routed_experts"] is LossInputLayout.PER_TOKEN
    assert loss_inputs.pad_values == {"routed_experts": -1}


def test_collate_requires_explicit_layout_for_trailing_token_features():
    datums = _routed_datums()
    for datum in datums:
        datum.loss_fn_input_layouts.clear()

    with pytest.raises(ValueError, match="declare an explicit layout for 'routed_experts'"):
        collate_datums(datums)


def test_collate_packed_per_token_trailing_dims_preserve_token_order_and_metadata():
    datums = _routed_datums()

    model_inputs, loss_inputs = collate_datums(datums, packed=True)

    routes = loss_inputs["routed_experts"]
    assert model_inputs["input_ids"].tolist() == [[10, 11, 12, 20, 21]]
    assert routes.shape == (1, 5, 2, 2)
    torch.testing.assert_close(
        routes[0],
        torch.cat([datum.loss_fn_inputs["routed_experts"] for datum in datums]),
    )
    assert loss_inputs.layouts["routed_experts"] is LossInputLayout.PER_TOKEN
    assert loss_inputs.pad_values == {"routed_experts": -1}


def test_collate_packed_per_sample_side_input_is_one_per_datum():
    datums = [
        Datum(model_inputs={"input_ids": torch.tensor([1, 2])}, loss_fn_inputs={"advantages": torch.tensor([0.5])}),
        Datum(
            model_inputs={"input_ids": torch.tensor([3, 4, 5])},
            loss_fn_inputs={"advantages": torch.tensor([0.9])},
        ),
    ]
    batch, loss_inputs = collate_datums(datums, packed=True)
    assert batch["input_ids"].shape == (1, 5)
    assert loss_inputs["advantages"].tolist() == pytest.approx([0.5, 0.9])
    assert loss_inputs.layouts == {"advantages": LossInputLayout.PER_DATUM}
    # Logical THD items are the two valid sequences, not the one physical row.
    assert loss_inputs.item_to_datum == (0, 1)


def test_collated_loss_inputs_copy_preserves_side_channel_and_dict_compatibility():
    result = collate_datums(_toy_datums(), packing_layout="indexed_mask")
    loss_inputs = result[1]

    assert isinstance(result, tuple)
    for copied in (
        loss_inputs.copy(),
        copy(loss_inputs),
        deepcopy(loss_inputs),
        pickle.loads(pickle.dumps(loss_inputs)),  # noqa: S301 - trusted in-process round trip
    ):
        assert isinstance(copied, CollatedLossInputs)
        assert copied.keys() == loss_inputs.keys()
        assert all(torch.equal(copied[key], loss_inputs[key]) for key in loss_inputs)
        assert copied.layouts == loss_inputs.layouts
        assert copied.item_to_datum == loss_inputs.item_to_datum
        assert copied._engine_packing_layout == "indexed_mask"


def test_collated_loss_inputs_copy_preserves_read_only_pad_values():
    loss_inputs = collate_datums(_routed_datums())[1]

    for copied in (
        loss_inputs.copy(),
        copy(loss_inputs),
        deepcopy(loss_inputs),
        pickle.loads(pickle.dumps(loss_inputs)),  # noqa: S301 - trusted in-process round trip
    ):
        assert isinstance(copied, CollatedLossInputs)
        assert copied.pad_values == {"routed_experts": -1}
        with pytest.raises(TypeError):
            copied.pad_values["routed_experts"] = 0  # type: ignore[index]


def test_collated_loss_inputs_requires_complete_read_only_layouts():
    with pytest.raises(ValueError, match="exactly"):
        CollatedLossInputs(
            {"weights": torch.ones(2)},
            layouts={},
            item_to_datum=(0,),
        )

    loss_inputs = CollatedLossInputs(
        {"weights": torch.ones(2)},
        layouts={"weights": LossInputLayout.PER_TOKEN},
        item_to_datum=(index for index in [0]),  # type: ignore[arg-type]
    )
    assert loss_inputs.item_to_datum == (0,)
    with pytest.raises(TypeError):
        loss_inputs.layouts["weights"] = LossInputLayout.REPLICATED  # type: ignore[index]


@pytest.mark.parametrize(
    ("layouts", "pad_values", "error", "message"),
    [
        (
            {"weights": LossInputLayout.PER_TOKEN},
            {"missing": -1},
            ValueError,
            "unknown loss inputs",
        ),
        (
            {"weights": LossInputLayout.PER_DATUM},
            {"weights": -1},
            ValueError,
            "only valid for PER_TOKEN",
        ),
        (
            {"weights": LossInputLayout.PER_TOKEN},
            {"weights": object()},
            TypeError,
            "bool, int, or float",
        ),
    ],
)
def test_collated_loss_inputs_validates_pad_value_metadata(layouts, pad_values, error, message):
    with pytest.raises(error, match=message):
        CollatedLossInputs(
            {"weights": torch.ones(1)},
            layouts=layouts,
            item_to_datum=(0,),
            pad_values=pad_values,
        )


def test_collate_requires_consistent_per_token_pad_value_metadata():
    partial = _routed_datums()
    partial[1].loss_fn_input_pad_values.clear()
    with pytest.raises(ValueError, match="every Datum must declare the pad value"):
        collate_datums(partial)

    conflicting = _routed_datums()
    conflicting[1].loss_fn_input_pad_values["routed_experts"] = -2
    with pytest.raises(ValueError, match="same pad value"):
        collate_datums(conflicting)

    non_token = Datum(
        input_ids=torch.tensor([1]),
        loss_fn_inputs={"sample_id": torch.tensor(7)},
        loss_fn_input_layouts={"sample_id": LossInputLayout.PER_DATUM},
        loss_fn_input_pad_values={"sample_id": -1},
    )
    with pytest.raises(ValueError, match="does not use the PER_TOKEN layout"):
        collate_datums([non_token])


def test_collate_explicit_per_datum_overrides_single_token_shape_inference():
    datum = Datum(
        input_ids=torch.tensor([7]),
        loss_fn_inputs={"advantage": torch.tensor([0.5])},
        loss_fn_input_layouts={"advantage": LossInputLayout.PER_DATUM},
    )

    _, loss_inputs = collate_datums([datum])

    assert loss_inputs["advantage"].shape == (1,)
    assert loss_inputs.layouts == {"advantage": LossInputLayout.PER_DATUM}


@pytest.mark.parametrize(
    ("layout", "value", "message"),
    [
        (LossInputLayout.PER_TOKEN, torch.tensor(1.0), "PER_TOKEN"),
        (LossInputLayout.PER_DATUM, torch.ones(2), "PER_DATUM"),
    ],
)
def test_collate_validates_explicit_loss_input_layout(layout, value, message):
    datum = Datum(
        input_ids=torch.tensor([1, 2]),
        loss_fn_inputs={"field": value},
        loss_fn_input_layouts={"field": layout},
    )

    with pytest.raises(ValueError, match=message):
        collate_datums([datum])


def test_collate_rejects_conflicting_explicit_layouts():
    datums = [
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"field": torch.tensor([0.5])},
            loss_fn_input_layouts={"field": LossInputLayout.PER_TOKEN},
        ),
        Datum(
            input_ids=torch.tensor([2]),
            loss_fn_inputs={"field": torch.tensor([0.9])},
            loss_fn_input_layouts={"field": LossInputLayout.PER_DATUM},
        ),
    ]

    with pytest.raises(ValueError, match="same explicit layout"):
        collate_datums(datums)


def test_collate_rejects_partially_declared_layouts():
    datums = [
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"field": torch.tensor([0.5])},
            loss_fn_input_layouts={"field": LossInputLayout.PER_TOKEN},
        ),
        Datum(
            input_ids=torch.tensor([2]),
            loss_fn_inputs={"field": torch.tensor([0.9])},
        ),
    ]

    with pytest.raises(ValueError, match="every Datum must declare"):
        collate_datums(datums)


def test_collate_replicated_input_keeps_one_identical_value():
    shared = torch.tensor([0.1, 0.2])
    datums = [
        Datum(
            input_ids=torch.tensor([1, 2]),
            loss_fn_inputs={"coefficients": shared},
            loss_fn_input_layouts={"coefficients": LossInputLayout.REPLICATED},
        ),
        Datum(
            input_ids=torch.tensor([3]),
            loss_fn_inputs={"coefficients": shared.clone()},
            loss_fn_input_layouts={"coefficients": LossInputLayout.REPLICATED},
        ),
    ]

    _, loss_inputs = collate_datums(datums, packed=True)

    assert loss_inputs["coefficients"] is shared
    assert loss_inputs.layouts == {"coefficients": LossInputLayout.REPLICATED}
    assert loss_inputs.item_to_datum == (0, 1)


def test_collate_replicated_input_requires_equal_values():
    datums = [
        Datum(
            input_ids=torch.tensor([1]),
            loss_fn_inputs={"coefficient": torch.tensor(0.1)},
            loss_fn_input_layouts={"coefficient": LossInputLayout.REPLICATED},
        ),
        Datum(
            input_ids=torch.tensor([2]),
            loss_fn_inputs={"coefficient": torch.tensor(0.2)},
            loss_fn_input_layouts={"coefficient": LossInputLayout.REPLICATED},
        ),
    ]

    with pytest.raises(ValueError, match="same value"):
        collate_datums(datums)


def test_collate_carries_per_token_float_side_inputs():
    # advantages is float per-token -> padded to collated width and stacked.
    batch, loss_inputs = collate_datums(_toy_datums())
    assert "advantages" not in batch
    assert loss_inputs["advantages"].dtype == torch.float
    assert loss_inputs["advantages"].shape == (2, 3)
    assert loss_inputs["advantages"][1].tolist() == pytest.approx([0.9, 0.9, 0.0])  # right-pad with 0


def test_collate_per_sample_scalar_side_input():
    datums = [
        Datum(model_inputs={"input_ids": torch.tensor([1, 2])}, loss_fn_inputs={"advantages": torch.tensor([0.5])}),
        Datum(
            model_inputs={"input_ids": torch.tensor([3, 4, 5])},
            loss_fn_inputs={"advantages": torch.tensor([0.9])},
        ),
    ]
    _, loss_inputs = collate_datums(datums)
    # length-1 != seq_len -> treated as per-sample, one value per datum.
    assert loss_inputs["advantages"].shape == (2,)
    assert loss_inputs["advantages"].tolist() == pytest.approx([0.5, 0.9])


def test_collate_pad_seq_len_divisible():
    batch, loss_inputs = collate_datums(_toy_datums(), pad_seq_len_divisible=8)
    assert batch["input_ids"].shape == (2, 8)
    # float side-inputs follow the collated width.
    assert loss_inputs["advantages"].shape == (2, 8)


def test_collate_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        collate_datums([])


def test_to_features_without_weights_leaves_labels_unmasked():
    d = Datum(input_ids=torch.tensor([1, 2, 3]), loss_fn_inputs={"target_tokens": torch.tensor([2, 3, 4])})
    assert d.to_features()["labels"] == [2, 3, 4]


def test_collate_rejects_inconsistent_loss_input_keys():
    datums = [
        Datum(input_ids=torch.tensor([1, 2]), loss_fn_inputs={"advantages": torch.zeros(2), "weights": torch.ones(2)}),
        Datum(input_ids=torch.tensor([3, 4]), loss_fn_inputs={"advantages": torch.zeros(2)}),
    ]
    # Silently intersecting the keys would drop the loss mask for the whole batch.
    with pytest.raises(ValueError, match="same loss_fn_inputs keys"):
        collate_datums(datums)


def test_collate_preserves_additional_token_model_inputs():
    datums = [
        Datum(
            model_inputs={"input_ids": torch.tensor([1, 2]), "token_type_ids": torch.tensor([0, 1])},
            loss_fn_inputs={"weights": torch.ones(2)},
        ),
        Datum(
            model_inputs={"input_ids": torch.tensor([3]), "token_type_ids": torch.tensor([1])},
            loss_fn_inputs={"weights": torch.ones(1)},
        ),
    ]
    model_inputs, loss_inputs = collate_datums(datums)
    assert model_inputs["token_type_ids"].tolist() == [[0, 1], [1, 1]]
    assert "weights" not in model_inputs
    assert loss_inputs["weights"].tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_default_packed_collater_rejects_model_specific_inputs():
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([1]), "pixel_values": torch.randn(1, 3, 2, 2)},
        loss_fn_inputs={"weights": torch.ones(1)},
    )
    with pytest.raises(ValueError, match="model-specific collate_fn"):
        collate_datums([datum], packed=True)


def test_default_packed_collater_rejects_explicitly_padded_datum():
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([1, 2, 0]), "attention_mask": torch.tensor([1, 1, 0])},
        loss_fn_inputs={"weights": torch.tensor([1.0, 1.0, 0.0])},
    )
    with pytest.raises(ValueError, match="only real tokens"):
        collate_datums([datum], packed=True)


def test_default_collater_rejects_float_token_model_inputs_instead_of_casting_them():
    datum = Datum(
        model_inputs={"input_ids": torch.tensor([1, 2]), "token_scores": torch.tensor([0.2, 1.7])},
        loss_fn_inputs={"weights": torch.ones(2)},
    )
    with pytest.raises(ValueError, match="cannot preserve floating-point"):
        collate_datums([datum])


def test_collate_reads_a_length_one_entry_on_a_single_token_sequence_as_per_token():
    datums = [Datum(input_ids=torch.tensor([7]), loss_fn_inputs={"advantages": torch.tensor([0.5])})]
    assert collate_datums(datums)[1]["advantages"].shape == (1, 1)


def test_collate_padding_mask_does_not_misread_a_real_pad_valued_token():
    # A Datum holds only real tokens, so the mask must come from its length, not
    # from matching the pad id -- id 0 is a real token here (and pad_token_id ==
    # eos_token_id is a common config).
    datums = [Datum(input_ids=torch.tensor([5, 0, 7])), Datum(input_ids=torch.tensor([9, 9]))]
    batch, _ = collate_datums(datums)
    assert batch["padding_mask"].tolist() == [[False, False, False], [False, False, True]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
