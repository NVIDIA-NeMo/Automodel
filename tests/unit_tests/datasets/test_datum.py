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

import pytest
import torch

from nemo_automodel.components.datasets.datum import (
    CROSS_ENTROPY_IGNORE_IDX,
    Datum,
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
