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

import pytest
import torch

from nemo_automodel.components.datasets.datum import (
    CROSS_ENTROPY_IGNORE_IDX,
    CollatedLossInputs,
    Datum,
    LossInputLayout,
    collate_datums,
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


# ── Datum ─────────────────────────────────────────────────────────────────


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
    result = collate_datums(_toy_datums())
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
